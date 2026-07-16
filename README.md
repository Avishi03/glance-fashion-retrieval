# Multimodal Fashion & Context Retrieval

Natural-language search over a fashion image corpus: "a red tie and a white shirt
in a formal setting" returns the right images, and — the part that matters — does
*not* return the same images for "a white tie and a red shirt".

## The problem this is built around

CLIP is a strong zero-shot baseline and a poor compositional one. A dual encoder
pools an image into a single vector, so `red shirt + blue pants` and
`blue shirt + red pants` land in nearly the same place: the model registers that
`{red, blue, shirt, pants}` are present but loses **which colour attaches to
which garment**. This bag-of-words behaviour is well documented in contrastive
vision-language models, and it is the failure the brief's hint points at.

Everything below exists to fix binding.

## Architecture

Modelled on **LookSync** (Glance's production visual product search system,
[arXiv:2511.00072](https://arxiv.org/abs/2511.00072)), which decomposes a look
into layer-wise garment descriptions, retrieves from a vector DB, and reranks.
Same four stages, adapted from *product retrieval given an AI-generated look* to
*scene retrieval given natural language*.

```
query ──► decompose ──► clause routing ──► ANN shortlist ──► fusion ──► top-k
              │              │                  │                │
        syntactic split   text-encoder      ChromaDB        one-to-one
        (no vocabulary)   anchors           (global +       assignment
                          (zero-shot)       region vectors)
```

### Part A — Indexer (`indexer/`)

Two vector kinds per image:

| Vector | Source | Answers |
|---|---|---|
| `global` | whole frame | scene ("inside a modern office"), vibe ("casual weekend") |
| `region` | one per garment box | which colour belongs to which garment |

A crop of the shirt contains *only* the shirt, so "red" has nothing else to bind
to. Region boxes come from Fashionpedia annotations, filtered to the 27 real
garment/accessory categories — **46% of its boxes are parts** (`sleeve`,
`neckline`, `zipper`, `rivet`) and a crop of a zipper is index noise.

There is deliberately no third "scene vector": with one backbone that is just the
global vector under another name.

Storage is **ChromaDB** — the brief says to pick the most convenient vector DB
rather than build one. The ML logic is in *what we embed*, not in the store.

### Part B — Retriever (`retriever/`)

**Decomposition is zero-shot by construction.** The brief grades handling of
"descriptions it hasn't seen explicitly in a training label", which rules out the
obvious implementation: a parser with a hardcoded colour list and garment
taxonomy passes the five graded queries and silently drops any term outside its
vocabulary — closed-vocabulary retrieval in a zero-shot costume.

So we never enumerate colours or garments:

1. **Split** on conjunctions/commas — purely syntactic, no word list.
2. **Route** each clause by comparing it to *anchor prompts* describing what a
   garment is versus what a place is, using the text encoder itself. The encoder
   knows "windbreaker" even though we never wrote it down. Unsure clauses fall
   back to the global vector, so the system degrades to CLIP rather than to
   nothing.

**Binding via one-to-one assignment.** Region scoring alone is insufficient, and
this is the subtle part. If each clause independently takes its best region, then
for an image with a **red shirt** and a **white tie**, the clause "a red tie"
matches the red *shirt* crop — colour similarity dominates garment identity, the
swapped image still scores well, and nothing is fixed.

Instead garment clauses are assigned to regions **one-to-one**, greedily by
descending similarity. Clauses compete: once "a red tie" claims the red crop,
"a white shirt" cannot have it too. An image genuinely containing a red tie and a
white shirt satisfies both clauses with two different crops; the swapped image
satisfies neither well. This needs no category lookup, so it stays open-vocabulary.

Fusion z-normalises per clause. Raw scores are not comparable across clauses or
backbones — SigLIP's sigmoid loss puts its cosines on a completely different
scale from CLIP's InfoNCE — so absolute thresholds would be meaningless.

## Dataset

**1,788 images**: 1,200 Fashionpedia + 588 COCO. Built, not sampled blind.

Two audits drove this, and both changed the design:

**Fashionpedia's natural distribution starves the graded queries.** `tie` appears
in 3 of 1,158 val images — the compositional query would have had no ground truth
at all. So the corpus is **stratified from train** with a quota per graded query
(200 tie+shirt, 200 formal, 180 outerwear, 160 shirt, 200 casual, 260 random
distractors), filled rarest-first so a tie+shirt image is never consumed by the
distractor bucket.

**Fashionpedia cannot serve the environment axis.** Zero-shot scene tagging found
it is **85% red carpet, runway and studio**; the four environments the brief
requires total 14.2%, with 19 office images and 4 home images. Three of five
graded queries are environment-driven.

COCO fixes that and does something a generic scene dataset could not: **its object
labels give scene ground truth not derived from CLIP**. A `bench` box is a human
annotation, so grading a CLIP retriever against it is not circular — which
CLIP-generated scene tags could never be.

| Scene | Rule (person always required) | Images |
|---|---|---|
| office | `laptop`/`keyboard`/`mouse` | 89 |
| home | `couch`/`bed`/`dining table`/`tv` | 175 |
| park | `bench` | 149 |
| street | `car`/`traffic light`/`bus`/`bicycle` | 175 |

## Evaluation

Fashionpedia's HF release ships **no colour or scene attributes**, so there are no
labels to compute Precision@5 against. LookSync hit the same wall and fell back to
human Mean Opinion Score.

So the headline metric is **colour-swap overlap**, which needs no labels:

> Retrieve top-5 for "a red tie and a white shirt", then for "a white tie and a
> red shirt". Both queries contain an *identical bag of words*. A system that
> ignores binding returns nearly the same images for both (overlap → 5/5). A
> system that binds returns different ones (overlap → 0/5).

**Lower is better.** It is rank-based (no score comparability needed) and
self-supervised (no ground truth needed). It measures *sensitivity to binding* —
precisely the property the brief's hint calls out.

## Scalability to 1M images

Candidate generation is an ANN lookup in ChromaDB over the whole corpus; only the
shortlist is scored densely. Cost is O(log N) in corpus size, so the logic is
unchanged at 1M images. Region vectors multiply the index ~4×, which is the real
cost — mitigable by indexing regions only for queries that contain garment
clauses, or by product quantisation.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python data/prepare.py           # stratified Fashionpedia sample
python data/prepare_coco.py      # COCO scene top-up
python -m indexer.build_index --backbone fashion-siglip
python -m eval.run_eval
```

## Layout

```
data/       corpus construction + audits    (logic separated from data)
indexer/    Part A: encoders, region crops, vector storage
retriever/  Part B: decomposition, routing, assignment, fusion
eval/       graded queries + compositional stress test
```
