"""Generate the submission PDF."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/Avishi03/glance-fashion-retrieval"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceBefore=14,
                    spaceAfter=8, textColor=colors.HexColor("#111111"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=11,
                    spaceAfter=5, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9.5, leading=13.5,
                      alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, leading=11.5,
                       textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=8,
                      leading=10.5, backColor=colors.HexColor("#f4f4f4"))


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, style=BODY):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=10) for i in items],
        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=7,
    )


def table(data, widths, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def build():
    res = json.loads((ROOT / "eval" / "results.json").read_text())
    story = []

    # ---------------------------------------------------------------- title
    story += [
        P("Multimodal Fashion &amp; Context Retrieval", ParagraphStyle(
            "T", parent=ss["Title"], fontSize=20, spaceAfter=2)),
        P("Compositional text-to-image retrieval over a fashion corpus", SMALL),
        Spacer(1, 8),
    ]

    # ------------------------------------------------------------- framing
    story += [
        P("1. The problem", H1),
        P("CLIP is a strong zero-shot retrieval baseline and a weak compositional one. "
          "A dual encoder pools an image into a single vector, so <b>red shirt + blue pants</b> "
          "and <b>blue shirt + red pants</b> land in nearly the same place: the model registers "
          "that {red, blue, shirt, pants} are present but loses <b>which colour attaches to which "
          "garment</b>. This bag-of-words behaviour is well documented in contrastive "
          "vision-language models, and it is the failure the brief's hint points at."),
        P("Every design decision below exists to fix attribute binding. The system is measured "
          "on whether it can tell a query from its colour-swapped twin — a property vanilla CLIP "
          "largely does not have."),
    ]

    # ----------------------------------------------------------- approaches
    story += [
        P("2. Approaches considered", H1),
        P("Five viable architectures, in rough order of cost:"),
        table([
            ["Approach", "How it works", "Good when", "Weakness"],
            ["A. Vanilla CLIP,\nglobal vector",
             "One embedding per image; cosine ANN against the query.",
             "Fast to build, genuinely zero-shot, scales trivially. The right "
             "default for coarse semantic search.",
             "No attribute binding. Fine-grained fashion vocabulary is weak "
             "(trained on web alt-text, not garments)."],
            ["B. Domain-tuned\nbackbone",
             "Swap in a fashion-fine-tuned encoder (FashionCLIP, FashionSigLIP).",
             "One-line change, large gain on garment vocabulary and texture/fabric "
             "terms. Always worth doing.",
             "Still a dual encoder — still pools the frame, so binding is unfixed. "
             "Domain shift if the corpus is not catalogue-like."],
            ["C. Region multi-\nvector (chosen)",
             "Embed each garment crop separately; assign query clauses to distinct "
             "regions.",
             "Directly attacks binding: a shirt crop contains only the shirt, so "
             "'red' cannot bind elsewhere. Still zero-shot.",
             "Index grows ~3–4×. Needs boxes — annotations here, a detector at "
             "scale. Crops lose global context."],
            ["D. Cross-encoder\nrerank",
             "Joint image-text encoder (BLIP-ITM) with cross-attention over a "
             "shortlist.",
             "Cross-attention resolves binding properly. Cheap if confined to "
             "top-k (k≈50).",
             "Orders of magnitude slower per pair — cannot touch the full corpus. "
             "Latency scales with k."],
            ["E. Fine-tune with\nhard negatives",
             "Train on colour-swapped negatives to force binding into the "
             "embedding.",
             "Best ceiling; pushes binding into the vector so retrieval stays a "
             "single ANN lookup.",
             "Needs labelled attribute data (absent here), GPU budget, and "
             "sacrifices some zero-shot generality."],
        ], [26 * mm, 40 * mm, 51 * mm, 51 * mm]),
        Spacer(1, 4),
        P("<b>Chosen: B + C</b>, with D scoped as the next increment. B is free. C is where the "
          "brief's stated failure mode actually lives. D is deliberately deferred rather than "
          "dropped — it is the strongest precision lever left, and the shortlist architecture is "
          "already built to accept it. E is out of reach: the dataset ships no attribute labels "
          "to train on (see §4).", SMALL),
    ]

    story.append(PageBreak())

    # --------------------------------------------------------- architecture
    story += [
        P("3. Chosen architecture", H1),
        P("Modelled on <b>LookSync</b>, Glance's production visual product search system "
          "(arXiv:2511.00072): decompose the look into layer-wise garment descriptions → "
          "vectorise → vector-DB retrieval → rerank. The same four stages, adapted from "
          "<i>product retrieval given an AI-generated look</i> to <i>scene retrieval given "
          "natural language</i>."),
        P("Notably, LookSync's own bake-off put vanilla CLIP ViT-H/14 <i>above</i> FashionCLIP "
          "and Fashion-SigLIP, by 3–7% MOS — on catalogue product retrieval. That result did not "
          "transfer here, which is why the backbone was chosen by measurement rather than "
          "assumption (§5).", SMALL),

        P("Part A — Indexer", H2),
        P("Two vector kinds per image:"),
        table([
            ["Vector", "Source", "Answers"],
            ["global", "whole frame", "scene ('inside a modern office'), vibe ('casual weekend')"],
            ["region", "one per garment box", "which colour belongs to which garment"],
        ], [24 * mm, 34 * mm, 110 * mm]),
        Spacer(1, 5),
        P("Region boxes are filtered to the 27 real garment/accessory categories. "
          "<b>46% of Fashionpedia's boxes are parts</b> — sleeve, neckline, zipper, rivet — and a "
          "crop of a zipper is index noise. There is deliberately no third 'scene vector': with a "
          "single backbone that is just the global vector under another name. Storage is ChromaDB, "
          "per the brief's instruction to pick a convenient store rather than build one; the ML "
          "logic is in <i>what</i> is embedded, not the store."),

        P("Part B — Retriever", H2),
        P("<b>Decomposition is zero-shot by construction.</b> The brief grades handling of "
          "'descriptions it hasn't seen explicitly in a training label', which rules out the "
          "obvious implementation. A parser with a hardcoded colour list and garment taxonomy "
          "passes all five graded queries and silently drops any term outside its vocabulary — "
          "closed-vocabulary retrieval wearing a zero-shot costume. So no colours or garments are "
          "ever enumerated:"),
        bullets([
            "<b>Split</b> on conjunctions and commas — purely syntactic, no word list.",
            "<b>Route</b> each clause by comparing it against <i>anchor prompts</i> describing "
            "what a garment is versus what a place is, using the text encoder itself. The encoder "
            "knows 'windbreaker' even though we never wrote it down. Unsure clauses fall back to "
            "the global vector, so the system degrades to CLIP rather than to nothing.",
        ]),
        Spacer(1, 3),
        P("<b>Binding via one-to-one assignment.</b> Region scoring alone is <i>not</i> enough, and "
          "this is the subtle part. If each clause independently takes its best-matching region, "
          "then for an image with a <b>red shirt</b> and a <b>white tie</b>, the clause 'a red tie' "
          "happily matches the red <i>shirt</i> crop — colour similarity dominates garment "
          "identity, the swapped image still scores well, and nothing has been fixed."),
        P("Instead garment clauses are assigned to regions <b>one-to-one</b>, greedily by "
          "descending similarity. Clauses compete: once 'a red tie' claims the red crop, 'a white "
          "shirt' cannot also have it. An image genuinely containing a red tie and a white shirt "
          "satisfies both clauses with two different crops; the colour-swapped image satisfies "
          "neither well. This needs no category lookup, so it remains open-vocabulary."),
        P("Fusion z-normalises per clause. Raw scores are not comparable across clauses or "
          "backbones — SigLIP's pairwise sigmoid loss puts its cosines on a completely different "
          "scale from CLIP's InfoNCE (a random image scores +0.18 under CLIP, −0.05 under "
          "SigLIP) — so absolute thresholds would be meaningless.", SMALL),
    ]

    story.append(PageBreak())

    # -------------------------------------------------------------- dataset
    story += [
        P("4. Dataset: built, not sampled", H1),
        P("<b>1,788 images</b> = 1,200 Fashionpedia + 588 COCO. Two audits drove this, and both "
          "changed the design."),

        P("Audit 1 — Fashionpedia's distribution starves the graded queries", H2),
        P("<b>tie appears in 3 of 1,158 val images.</b> The compositional query — the one the "
          "entire architecture exists to win — would have had essentially no ground truth. Train "
          "is 39× larger and holds 1,455 tie images and 1,402 tie+shirt images, so the corpus is "
          "<b>stratified from train with a quota per graded query</b> (200 tie+shirt, 200 formal, "
          "180 outerwear, 160 shirt, 200 casual, 260 random distractors), filled rarest-first so a "
          "tie+shirt image is never consumed by the distractor bucket."),

        P("Audit 2 — Fashionpedia cannot serve the environment axis", H2),
        P("Zero-shot scene tagging over the sample found it is <b>85% red carpet, runway and "
          "studio</b>. The four environments the brief requires total <b>14.2%</b> — 19 office "
          "images and 4 home images out of 1,200 — while three of the five graded queries are "
          "environment-driven."),
        table([
            ["Scene", "Share"],
            ["red carpet", "36.2%"],
            ["runway / catwalk", "29.8%"],
            ["studio backdrop", "19.1%"],
            ["urban street / park / office / home", "14.2%  ← the required axis"],
        ], [60 * mm, 60 * mm]),
        Spacer(1, 5),
        P("COCO fixes that, and does one thing a generic scene dataset could not: <b>its object "
          "labels give scene ground truth that is not derived from CLIP</b>. A 'bench' box is a "
          "human annotation, so grading a CLIP retriever against it is not circular — which "
          "CLIP-generated scene tags could never be. Scene is inferred from co-present objects with "
          "person always required: office ← laptop/keyboard/mouse (89), home ← couch/bed/tv (175), "
          "park ← bench (149), street ← car/traffic light/bus (175)."),
        P("Note the honest limit: COCO rows carry no garment annotations, so they contribute a "
          "global vector only. They are scene evidence and distractors, not garment evidence.",
          SMALL),
    ]

    # -------------------------------------------------------------- results
    story += [
        P("5. Results", H1),
        P("<b>The metric.</b> Fashionpedia's HF release ships no colour or scene attributes, so "
          "there are no labels to compute Precision@5 against. LookSync hit the same wall and fell "
          "back to human Mean Opinion Score. So the headline metric needs no labels:"),
        P("Retrieve top-5 for <i>'a red tie and a white shirt'</i>, then for <i>'a white tie and a "
          "red shirt'</i>. Both queries contain an <b>identical bag of words</b>. A system that "
          "ignores binding returns nearly the same images for both (overlap → 1.0). A system that "
          "binds returns different ones (overlap → 0). <b>Lower is better.</b> The metric is "
          "rank-based (no score comparability needed) and self-supervised (no ground truth "
          "needed): it measures <i>sensitivity to binding</i>, exactly the property the brief's "
          "hint calls out. Averaged over 4 swap pairs, k=5.", SMALL),
        Spacer(1, 4),
        Image(str(ROOT / "eval" / "ablation.png"), width=140 * mm, height=81 * mm),
        Spacer(1, 4),
        table([
            ["Configuration", "Swap overlap ↓", "Reading"],
            ["vanilla CLIP (global only)", f"{res['vanilla CLIP (global only)']['mean_swap_overlap']:.2f}",
             "Returns 65% of the same images either way — it cannot tell the queries apart."],
            ["FashionSigLIP (global only)", f"{res['FashionSigLIP (global only)']['mean_swap_overlap']:.2f}",
             "Domain vocabulary helps, but pooling still destroys binding."],
            ["FashionSigLIP + regions", f"{res['FashionSigLIP + regions']['mean_swap_overlap']:.2f}",
             "Region scoring + assignment does the heavy lifting: 0.50 → 0.30."],
        ], [44 * mm, 24 * mm, 100 * mm]),
        Spacer(1, 5),
        P("The backbone swap buys 0.65 → 0.50; <b>the mechanism buys 0.50 → 0.30</b>. That split "
          "matters: the gain is attributable to the architecture, not to having picked a fancier "
          "model. Zero-shot decomposition routes correctly with no vocabulary — 'A red tie' → "
          "garment, 'a white shirt' → garment, 'in a formal setting' → scene."),

        P("Relevance: swap overlap alone is not sufficient", H2),
        P("Swap overlap measures whether a system is <i>sensitive</i> to binding, not whether it is "
          "<i>right</i>. A retriever returning five random images would score a perfect 0.00. "
          "Adding region vectors adds noise, and noise also lowers overlap — so the result above is "
          "consistent with better binding <i>and</i> with simply being noisier. The two must be "
          "disentangled, and the only way to do that here is to look at the output: the corpus "
          "ships no colour or scene labels to score against. This is the same wall LookSync hit, "
          "and they answered it the same way — human judgement."),
        P("Precision@5, judged by inspection of the top-5 for each graded query "
          "(contact sheets for both systems are in the repo under <font face='Courier' size='8'>"
          "eval/</font>):", SMALL),
        table([
            ["Graded query", "Ours", "Vanilla CLIP"],
            ["1. A person in a bright yellow raincoat", "2/5", "1/5"],
            ["2. Professional business attire inside a modern office", "4/5", "2/5"],
            ["3. Someone wearing a blue shirt sitting on a park bench", "3/5", "1/5"],
            ["4. Casual weekend outfit for a city walk", "5/5", "4/5"],
            ["5. A red tie and a white shirt in a formal setting", "5/5", "0/5"],
            ["Precision@5", "0.76", "0.32"],
        ], [92 * mm, 22 * mm, 26 * mm]),
        Spacer(1, 5),
        P("So relevance moves the same way as binding: the region system is not merely different "
          "from CLIP, it is better. The compositional query is the clearest case — and it is worth "
          "seeing rather than reading:"),
        Spacer(1, 3),
        Image(str(ROOT / "eval" / "comparison_compositional.png"), width=168 * mm, height=97 * mm),
        Spacer(1, 3),
        P("Vanilla CLIP returns <b>zero red ties</b>: a white bow tie, an olive tie, a striped tie, "
          "a black bow tie, and — the diagnostic case — <b>a white tie on a gold shirt</b>. The "
          "colour-swapped image is retrieved as a top match, because a pooled vector registers "
          "{red, white, tie, shirt} without binding. This is the brief's hint, reproduced exactly.",
          SMALL),

        P("Known limitations", H2),
        bullets([
            "<b>Query 1 is the weakest (2/5).</b> Fashionpedia has no 'raincoat' category — only "
            "coat/jacket/cape — so 'raincoat' degrades to 'yellow garment' and the results drift to "
            "yellow tops and an orange dress. The top hit is a genuine yellow raincoat; the tail is "
            "not. A vocabulary gap in the corpus, not in the model.",
            "<b>Precision@5 was judged by the author, on 25 image-query pairs.</b> That is a "
            "small, non-blind sample. LookSync used a panel and reported MOS with the same caveat; "
            "at production scale this needs multiple raters and a held-out set.",
            "<b>4 swap pairs is a small sample.</b> The ordering is consistent and the gap is "
            "wide, but the absolute values carry meaningful variance.",
            "<b>Greedy assignment, not Hungarian.</b> Optimal for ≤5 clauses in practice, but not "
            "guaranteed.",
            "<b>Region boxes come from annotations.</b> At 1M images this needs a detector "
            "(LookSync uses SAM v2 + Florence as its fallback path), which introduces "
            "detector error the current numbers do not include.",
            "<b>Query 4 does not decompose</b> — 'Casual weekend outfit for a city walk' routes as "
            "a single garment clause, since the splitter has no rule for 'for a…'. It scores 5/5 "
            "anyway, because the whole-query prior still carries the city context. Worth noting as "
            "luck rather than design: the fallback path saved it.",
        ], SMALL),
    ]

    story.append(PageBreak())

    # ---------------------------------------------------------- scalability
    story += [
        P("6. Scalability to 1M images", H1),
        P("Candidate generation is an ANN lookup in ChromaDB over the whole corpus; only the "
          "shortlist is scored densely. Cost is O(log N) in corpus size, so the retrieval logic is "
          "unchanged at 1M images. Three real costs:"),
        bullets([
            "<b>Index size grows ~3.4×</b> (6,036 vectors for 1,788 images). At 1M images that is "
            "~3.4M vectors × 768 dims × 4 bytes ≈ 10 GB — past comfortable RAM. Fix: product "
            "quantisation, or index regions only for garment-bearing images.",
            "<b>Dense scoring is O(candidates × clauses × regions)</b>, but candidates are capped "
            "at ~150/clause, so this is constant in N.",
            "<b>Chroma is the right call at this scale and the wrong one at 1M.</b> The brief says "
            "to pick the convenient store; the migration path is Qdrant or Vertex AI Vector Search "
            "(LookSync serves 18M+ products at P90 &lt; 1s with an offline-index / online-serve "
            "split plus aggressive caching).",
        ]),

        P("7. Future work", H1),
        P("(a) Adding locations (cities, places) and weather", H2),
        P("The architecture already has the right seam: <b>scene clauses route to the global "
          "vector</b>, so location and weather are a third clause role rather than a redesign."),
        bullets([
            "<b>Add a role, not a model.</b> Extend the anchor set with 'a city or geographic "
            "location' and 'the weather or time of day'. Routing is similarity-based, so this is "
            "additive — no retraining, no vocabulary.",
            "<b>Cheap metadata first.</b> EXIF GPS and capture time give city and season for free "
            "where present; both become Chroma metadata filters, which cut the candidate set "
            "<i>before</i> any vector work and so cost nothing at query time.",
            "<b>Where labels are absent, use a dedicated encoder.</b> A geo-aware model (StreetCLIP "
            "was trained for exactly this) tags city/region; weather is well within zero-shot CLIP "
            "('an overcast day', 'rain-soaked street'). Store as a fourth vector kind + metadata.",
            "<b>Weather is a garment prior, not just a filter</b> — the commercially interesting "
            "part. 'What to wear in Bangalore in July' is a monsoon query: it should surface "
            "raincoats and closed shoes. That is a learned correlation between weather and garment "
            "category, not a retrieval constraint, and it is where a recommendation layer would sit "
            "on top of this retriever.",
        ]),
        P("(b) Improving precision", H2),
        bullets([
            "<b>Cross-encoder rerank (approach D) is the biggest lever left.</b> BLIP-ITM over the "
            "top-50: cross-attention resolves binding properly rather than approximating it with "
            "assignment. Confined to a shortlist it is affordable, and the architecture already "
            "produces that shortlist. This is the first thing I would build next.",
            "<b>Hard-negative mining.</b> Colour-swapped and garment-swapped negatives are exactly "
            "what the swap metric measures — mine them automatically from the corpus and fine-tune "
            "(approach E). This pushes binding into the embedding so it survives at ANN time "
            "instead of being reconstructed at scoring time.",
            "<b>Replace the syntactic splitter with an LLM decomposer</b>, as LookSync does. It "
            "fixes the query-4 failure directly and handles possessives, negation ('not black') "
            "and implicit garments ('business attire' → blazer + button-down) that regex cannot.",
            "<b>Learn the fusion weights.</b> W_PRIOR and the clause weights are hand-set. With "
            "even a few hundred judged pairs they should be fit, not guessed.",
            "<b>Detector-based regions.</b> Removes the annotation dependency and lets region "
            "scoring work on arbitrary images — required for this to run on real traffic.",
        ]),

        P("8. Codebase", H1),
        P(f'<link href="{GITHUB_URL}" color="blue">{GITHUB_URL}</link>'),
        P("<font face='Courier' size='8'>data/</font> corpus construction + audits &nbsp;·&nbsp; "
          "<font face='Courier' size='8'>indexer/</font> Part A: encoders, crops, vector storage "
          "&nbsp;·&nbsp; <font face='Courier' size='8'>retriever/</font> Part B: decomposition, "
          "routing, assignment, fusion &nbsp;·&nbsp; <font face='Courier' size='8'>eval/</font> "
          "graded queries + stress test. Logic is separated from data throughout: the corpus is "
          "JSON + images, every model is swappable by a flag, and the eval harness runs any "
          "backbone/config combination without touching retrieval code.", SMALL),
    ]

    doc = SimpleDocTemplate(
        str(ROOT / "REPORT.pdf"), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Multimodal Fashion & Context Retrieval",
    )
    doc.build(story)
    print(f"wrote {ROOT / 'REPORT.pdf'}")


if __name__ == "__main__":
    build()
