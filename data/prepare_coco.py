"""Top up the corpus with real-world scenes from COCO.

Why this exists: the scene audit found Fashionpedia is 85% red carpet, runway
and studio -- only 14% of it sits in the four environments the brief requires,
with 19 office images and 4 home images out of 1200. Three of the five graded
queries are environment-driven, so Fashionpedia alone cannot serve them.

COCO fixes that, and does one thing better than a generic scene dataset would:
its object labels give us scene ground truth that is NOT derived from CLIP. A
'bench' box is a human annotation, so grading a CLIP retriever against it is not
circular -- which the zero-shot scene tags could never be.

Scene is inferred from co-present objects, person always required (the queries
are all about what someone is wearing):
    office <- laptop / keyboard / mouse
    home   <- couch / bed / dining table / tv
    park   <- bench
    street <- car / traffic light / bus / bicycle

Streaming, deliberately: load_dataset() resolves every split before selecting
one, so asking for COCO's 5k-image val split pulls the 118k-image train split
too (~19GB) to use 700 images. Streaming fetches only the shards we actually
consume and stops as soon as the quotas fill.

Appends to data/corpus.json with source='coco' and a scene label. COCO has no
garment annotations, so these rows carry no garment boxes -- they are scene
evidence and honest distractors, not garment evidence.
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

MAX_SIDE = 640
QUOTA_PER_SCENE = 175

OUT_DIR = Path(__file__).parent
IMG_DIR = OUT_DIR / "images"

# Ordered rarest-first: a bench+car image should land in 'park', not 'street'.
SCENE_RULES = [
    ("office", {"laptop", "keyboard", "mouse"}),
    ("home", {"couch", "bed", "dining table", "tv"}),
    ("park", {"bench"}),
    ("street", {"car", "traffic light", "bus", "bicycle"}),
]


def resize(img):
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return img
    s = MAX_SIDE / max(w, h)
    return img.resize((round(w * s), round(h * s)))


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
    names = ds.features["objects"]["category"].feature.names

    need = {o for _, objs in SCENE_RULES for o in objs} | {"person"}
    if missing := need - set(names):
        print(f"!! COCO category names not found: {missing}")
        print(f"   available: {names}")
        return

    counts = Counter()
    rows = []
    seen = 0

    for row in ds:
        seen += 1
        cats = {names[c] for c in row["objects"]["category"]}
        if "person" not in cats:
            continue

        scene = next((s for s, objs in SCENE_RULES if objs & cats), None)
        if scene is None or counts[scene] >= QUOTA_PER_SCENE:
            continue

        img = resize(row["image"].convert("RGB"))
        iid = f"coco_{row['image_id']}"
        img.save(IMG_DIR / f"{iid}.jpg", quality=92)
        rows.append({
            "image_id": iid,
            "file": f"{iid}.jpg",
            "source": "coco",
            "scene": scene,          # human-annotated -> usable as eval ground truth
            "stratum": f"scene_{scene}",
            "width": img.size[0],
            "height": img.size[1],
            "objects": [],           # COCO has no garment boxes
        })
        counts[scene] += 1
        print(f"  seen {seen:5d} | " + " ".join(f"{s}={counts[s]}" for s, _ in SCENE_RULES), end="\r")

        if all(counts[s] >= QUOTA_PER_SCENE for s, _ in SCENE_RULES):
            break

    print(f"\n\nstreamed {seen} images, kept {len(rows)}")
    for s, _ in SCENE_RULES:
        print(f"  {s:8s} {counts[s]}")

    corpus_path = OUT_DIR / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    corpus = [c for c in corpus if c.get("source") != "coco"]   # idempotent re-run
    for c in corpus:
        c.setdefault("source", "fashionpedia")
        c.setdefault("scene", None)      # unknown; Fashionpedia has no scene labels
    corpus.extend(rows)
    corpus_path.write_text(json.dumps(corpus, indent=1))

    n_fp = sum(1 for c in corpus if c["source"] == "fashionpedia")
    print(f"\ncorpus now {len(corpus)} images ({n_fp} fashionpedia + {len(rows)} coco)")


if __name__ == "__main__":
    main()
