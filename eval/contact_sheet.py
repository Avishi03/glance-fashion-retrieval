"""Render top-k results per graded query as a contact sheet, so a human can
actually look at what the system returns.

This exists because the swap-overlap metric measures sensitivity to binding, not
relevance. A retriever returning random images scores a perfect 0.0 overlap. The
metric is necessary but nowhere near sufficient, and the only honest check on
relevance -- given the corpus ships no colour or scene labels -- is to look.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from eval.queries import EVAL_QUERIES
from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CELL = 220
PAD = 6
LABEL_H = 34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="fashion-siglip")
    ap.add_argument("--regions", action="store_true", default=True)
    ap.add_argument("--no-regions", dest="regions", action="store_false")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    r = Retriever(args.backbone)
    rows = []
    for kind, q in EVAL_QUERIES:
        hits = r.search(q, k=args.k, use_regions=args.regions)
        rows.append((f"[{kind}] {q}", [h.image_id for h in hits]))
        print(f"{kind:14s} {[h.image_id for h in hits]}")

    W = PAD + args.k * (CELL + PAD)
    H = sum(LABEL_H + CELL + PAD for _ in rows) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)

    y = PAD
    for title, ids in rows:
        draw.text((PAD, y + 8), title, fill="black")
        y += LABEL_H
        for i, iid in enumerate(ids):
            im = Image.open(DATA / "images" / f"{iid}.jpg").convert("RGB")
            im.thumbnail((CELL, CELL))
            x = PAD + i * (CELL + PAD)
            sheet.paste(im, (x, y))
            draw.text((x + 3, y + 3), iid, fill="yellow")
        y += CELL + PAD

    tag = "regions" if args.regions else "global"
    out = ROOT / "eval" / f"contact_{args.backbone}_{tag}.png"
    sheet.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
