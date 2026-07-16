"""Build the image corpus: a stratified sample of Fashionpedia train.

Why stratified rather than random: the graded queries lean on categories that
are rare in Fashionpedia's natural distribution. A blind 1k sample of train
yields ~32 ties and would leave the compositional query ("a red tie and a white
shirt") with almost no ground truth. Quotas guarantee every graded query has a
pool to retrieve from.

The quotas are defined per query, not per category, so the corpus is built to
exercise the evaluation rather than to mirror Fashionpedia's own priors. The
remainder is filled randomly to keep the corpus diverse and to ensure the
retriever has genuine distractors to rank against.

Writes:
  data/images/{image_id}.jpg      resized, max side 640
  data/corpus.json                per-image annotations (garment boxes only)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 0
TARGET = 1200
MAX_SIDE = 640

OUT_DIR = Path(__file__).parent
IMG_DIR = OUT_DIR / "images"

# Categories worth cropping: whole garments and accessories that a person can
# describe in a query. Parts (sleeve, neckline, pocket, zipper, rivet...) are
# 46% of Fashionpedia's boxes but carry no standalone retrievable semantics --
# a crop of a zipper is noise in a fashion index.
MAIN_GARMENTS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress",
    "jumpsuit", "cape",
}
ACCESSORIES = {
    "glasses", "hat", "headband, head covering, hair accessory", "tie",
    "glove", "watch", "belt", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella", "hood",
}
KEEP = MAIN_GARMENTS | ACCESSORIES

# One quota per graded query. Predicates run against the set of category names
# present in an image.
QUOTAS = [
    # q5 compositional: needs both garments co-present to bind colour correctly
    ("q5_tie_shirt", 200, lambda c: {"tie", "shirt, blouse"} <= c),
    # q2 business attire
    ("q2_formal", 200, lambda c: "jacket" in c and ({"shirt, blouse"} & c or "pants" in c)),
    # q1 outerwear
    ("q1_outerwear", 180, lambda c: bool({"coat", "jacket", "cape"} & c)),
    # q3 shirt
    ("q3_shirt", 160, lambda c: "shirt, blouse" in c),
    # q4 casual
    ("q4_casual", 200, lambda c: bool({"top, t-shirt, sweatshirt", "shorts"} & c)),
    # distractors: anything, keeps the index honest
    ("distractor", 260, lambda c: True),
]


def resize(img):
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return img
    s = MAX_SIDE / max(w, h)
    return img.resize((round(w * s), round(h * s)))


def main():
    random.seed(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("loading annotations (images not decoded yet)...")
    ann = load_dataset("detection-datasets/fashionpedia", split="train").select_columns(["objects"])
    names = ann.features["objects"]["category"].feature.names

    # Pass 1: category sets per image, no JPEG decoding.
    cats_per_img = [
        {names[c] for c in row["objects"]["category"]}
        for row in tqdm(ann, desc="scanning")
    ]

    # Pass 2: fill quotas in order. Earlier (rarer) quotas get first claim on an
    # image, so a tie+shirt image is never consumed by the distractor bucket.
    chosen, taken = {}, set()
    for name, quota, pred in QUOTAS:
        pool = [i for i, c in enumerate(cats_per_img) if i not in taken and pred(c)]
        random.shuffle(pool)
        picked = pool[:quota]
        for i in picked:
            chosen[i] = name
            taken.add(i)
        print(f"{name:16s} quota {quota:4d}  pool {len(pool):6d}  took {len(picked):4d}")

    print(f"\ntotal selected: {len(chosen)} (target {TARGET})")

    # Pass 3: decode and write only the selected images.
    full = load_dataset("detection-datasets/fashionpedia", split="train")
    corpus = []
    for idx in tqdm(sorted(chosen), desc="writing"):
        row = full[idx]
        img = resize(row["image"].convert("RGB"))
        image_id = row["image_id"]
        img.save(IMG_DIR / f"{image_id}.jpg", quality=92)

        sx, sy = img.size[0] / row["width"], img.size[1] / row["height"]
        objects = [
            {
                "category": names[c],
                # rescale boxes to the resized image
                "bbox": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
            }
            for c, b in zip(row["objects"]["category"], row["objects"]["bbox"])
            if names[c] in KEEP
        ]
        corpus.append({
            "image_id": image_id,
            "file": f"{image_id}.jpg",
            "stratum": chosen[idx],
            "width": img.size[0],
            "height": img.size[1],
            "objects": objects,
        })

    (OUT_DIR / "corpus.json").write_text(json.dumps(corpus, indent=1))
    n_obj = sum(len(c["objects"]) for c in corpus)
    print(f"\nwrote {len(corpus)} images, {n_obj} garment boxes -> {OUT_DIR/'corpus.json'}")


if __name__ == "__main__":
    main()
