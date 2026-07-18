"""The money figure: vanilla CLIP vs ours on the compositional query.

Side by side, the failure the brief's hint describes is visible without any
metric: vanilla CLIP returns zero red ties for "a red tie and a white shirt",
including a white tie on a gold shirt -- the colour-swapped case retrieved as a
top match, because a pooled vector sees {white, red, tie, shirt} and cannot tell
which attaches to which.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from eval.queries import EVAL_QUERIES
from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

QUERY = EVAL_QUERIES[4][1]      # the compositional one
CELL, PAD, LABEL_H = 200, 5, 26
K = 5

ROWS = [
    ("vanilla CLIP (global only)  -  0/5 red ties", "vanilla-clip", False),
    ("FashionSigLIP + regions (ours)  -  4/5", "fashion-siglip", True),
]


def main():
    panels = []
    for title, backbone, regions in ROWS:
        r = Retriever(backbone)
        hits = r.search(QUERY, k=K, use_regions=regions)
        panels.append((title, [h.image_id for h in hits]))
        print(f"{backbone:15s} {[h.image_id for h in hits]}")

    W = PAD + K * (CELL + PAD)
    H = LABEL_H + PAD + len(panels) * (LABEL_H + CELL + PAD)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 6), f'query:  "{QUERY}"', fill="black")

    y = LABEL_H + PAD
    for title, ids in panels:
        d.text((PAD, y + 6), title, fill="black")
        y += LABEL_H
        for i, iid in enumerate(ids):
            im = Image.open(DATA / "images" / f"{iid}.jpg").convert("RGB")
            im.thumbnail((CELL, CELL))
            sheet.paste(im, (PAD + i * (CELL + PAD), y))
        y += CELL + PAD

    out = ROOT / "eval" / "comparison_compositional.png"
    sheet.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
