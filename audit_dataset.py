"""Audit Fashionpedia coverage before committing to it as the corpus.

Two questions:
  1. Does it span the three required axes (environment, clothing type, color)?
  2. Do the five evaluation queries have plausible ground-truth targets?

Question 2 is the important one: a query with no targets in the corpus fails at
demo time regardless of retrieval quality.
"""

from collections import Counter

from datasets import load_dataset

# Fashionpedia's 46 categories split into three bands. Only the first two are
# worth cropping and embedding -- a bbox around a zipper or a single sleeve
# carries no retrievable garment semantics and would pollute the index.
MAIN_GARMENTS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress",
    "jumpsuit", "cape",
}
ACCESSORIES = {
    "glasses", "hat", "headband, head covering, hair accessory", "tie",
    "glove", "watch", "belt", "leg warmer", "tights, stockings", "sock",
    "shoe", "bag, wallet", "scarf", "umbrella", "hood",
}

# The five graded queries, reduced to the category evidence each one needs.
EVAL_QUERY_NEEDS = {
    "1. bright yellow raincoat": ["coat", "jacket", "cape"],
    "2. business attire in office": ["jacket", "shirt, blouse", "pants", "tie"],
    "3. blue shirt on park bench": ["shirt, blouse"],
    "4. casual weekend city walk": ["top, t-shirt, sweatshirt", "shorts", "skirt"],
    "5. red tie and white shirt": ["tie", "shirt, blouse"],
}


def main():
    ds = load_dataset("detection-datasets/fashionpedia", split="val")
    names = ds.features["objects"]["category"].feature.names
    print(f"val split: {len(ds)} images, {len(names)} categories\n")

    cat_counts = Counter()
    imgs_with_cat = Counter()
    garments_per_img = []

    for row in ds:
        cats = [names[c] for c in row["objects"]["category"]]
        cat_counts.update(cats)
        for c in set(cats):
            imgs_with_cat[c] += 1
        garments_per_img.append(sum(1 for c in cats if c in MAIN_GARMENTS))

    band = lambda c: (
        "garment" if c in MAIN_GARMENTS else
        "accessory" if c in ACCESSORIES else "part"
    )

    print("=== category counts (images containing / total boxes) ===")
    for name, count in cat_counts.most_common():
        print(f"{imgs_with_cat[name]:5d} imgs {count:6d} boxes  [{band(name):9s}] {name}")

    # How much of the annotation budget is spent on un-croppable parts?
    by_band = Counter()
    for name, count in cat_counts.items():
        by_band[band(name)] += count
    total = sum(by_band.values())
    print("\n=== annotation budget by band ===")
    for b, c in by_band.most_common():
        print(f"  {b:9s} {c:6d} boxes ({100*c/total:.1f}%)")

    n_img = len(garments_per_img)
    print(f"\nmain garments per image: mean {sum(garments_per_img)/n_img:.2f}, "
          f"images with 0: {sum(1 for g in garments_per_img if g == 0)}, "
          f"with 2+: {sum(1 for g in garments_per_img if g >= 2)}")

    print("\n=== eval query category support (images containing) ===")
    for query, needed in EVAL_QUERY_NEEDS.items():
        parts = [f"{n}={imgs_with_cat.get(n, 0)}" for n in needed]
        ok = all(imgs_with_cat.get(n, 0) > 0 for n in needed)
        print(f"[{'OK ' if ok else 'GAP'}] {query:30s} {'  '.join(parts)}")

    # Co-occurrence for the compositional query -- the stress test needs images
    # where two colourable garments appear together.
    both = sum(
        1 for row in ds
        if {"tie", "shirt, blouse"} <= {names[c] for c in row["objects"]["category"]}
    )
    print(f"\nimages containing BOTH tie and shirt/blouse: {both}")


if __name__ == "__main__":
    main()
