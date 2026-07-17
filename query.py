"""Part B entrypoint: accept a natural-language string, return the top-k images.

    python query.py "A red tie and a white shirt in a formal setting."
    python query.py "someone in a bright yellow raincoat" --k 10
    python query.py "burnt sienna windbreaker" --show          # writes a contact sheet
    python query.py "a blue shirt in a park" --backbone vanilla-clip --no-regions

--backbone / --no-regions exist so the ablation in the report is reproducible
from the command line: run the same query through vanilla CLIP and through the
region system and compare what comes back.
"""

import argparse
from pathlib import Path

from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser(description="Natural-language fashion image search.")
    ap.add_argument("query", help="natural language description, in quotes")
    ap.add_argument("--k", type=int, default=5, help="number of images to return")
    ap.add_argument("--backbone", default="fashion-siglip",
                    choices=["fashion-siglip", "vanilla-clip"])
    ap.add_argument("--no-regions", dest="regions", action="store_false",
                    help="ablate region scoring -- score against whole-frame vectors only")
    ap.add_argument("--show", action="store_true",
                    help="write the results to a contact sheet PNG")
    args = ap.parse_args()

    r = Retriever(args.backbone)
    hits = r.search(args.query, k=args.k, use_regions=args.regions)

    if not hits:
        print("no results")
        return

    # Show the decomposition: it is the part worth seeing, since the clause roles
    # determine which vectors each part of the query was scored against.
    print(f'\nquery: "{args.query}"')
    print(f"backbone: {args.backbone}   regions: {args.regions}\n")
    print("decomposition:")
    for text, role, conf in hits[0].detail["clauses"]:
        print(f"   [{role:7s}] {text!r}   (routing margin {conf:+.4f})")

    print(f"\ntop {len(hits)}:")
    print(f"  {'rank':<5} {'score':>7}  {'image':<16} path")
    for i, h in enumerate(hits, 1):
        path = DATA / "images" / f"{h.image_id}.jpg"
        print(f"  {i:<5} {h.score:>7.3f}  {h.image_id:<16} {path}")

    if args.show:
        from PIL import Image, ImageDraw

        CELL, PAD = 240, 6
        sheet = Image.new("RGB", (PAD + len(hits) * (CELL + PAD), CELL + 34), "white")
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 8), f'"{args.query}"  ({args.backbone}, regions={args.regions})', fill="black")
        for i, h in enumerate(hits):
            im = Image.open(DATA / "images" / f"{h.image_id}.jpg").convert("RGB")
            im.thumbnail((CELL, CELL))
            sheet.paste(im, (PAD + i * (CELL + PAD), 30))
        out = ROOT / "query_result.png"
        sheet.save(out)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
