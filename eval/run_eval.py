"""Evaluation: ablation over the five graded queries + the compositional test.

The ablation isolates one change at a time, so the PDF can attribute any gain to
a specific mechanism rather than to "we used a fancier model":

  1. vanilla-clip,   global only   -- the baseline the brief asks us to beat
  2. fashion-siglip, global only   -- what the domain backbone alone buys
  3. fashion-siglip, regions       -- what region scoring + assignment adds

COMPOSITIONAL METRIC. We report top-5 OVERLAP between a query and its
colour-swapped twin ("a red tie and a white shirt" vs "a white tie and a red
shirt"). Both queries contain an identical bag of words, so:

  - a system that ignores binding retrieves nearly the SAME images for both
    -> overlap near 5/5
  - a system that binds colour to garment retrieves DIFFERENT images
    -> overlap near 0/5

Lower is better. Overlap is rank-based, so it needs no score comparability
across queries (fused scores are z-normalised per query and are not comparable)
and no ground-truth colour labels -- which matters, because Fashionpedia's HF
release ships none. It measures sensitivity to binding, not correctness of any
single result, and that is exactly the property the brief's hint calls out.
"""

import argparse
import json
from pathlib import Path

from eval.queries import EVAL_QUERIES, SWAP_PAIRS
from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    ("vanilla CLIP (global only)", "vanilla-clip", False),
    ("FashionSigLIP (global only)", "fashion-siglip", False),
    ("FashionSigLIP + regions", "fashion-siglip", True),
]
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    results = {}
    for label, backbone, use_regions in CONFIGS:
        print(f"\n{'='*66}\n{label}\n{'='*66}")
        r = Retriever(backbone)

        # --- the five graded queries -------------------------------------
        per_query = {}
        for kind, q in EVAL_QUERIES:
            hits = r.search(q, k=args.k, use_regions=use_regions)
            per_query[q] = [h.image_id for h in hits]
            clauses = hits[0].detail["clauses"] if hits else []
            print(f"\n[{kind}] {q}")
            print(f"  clauses: {clauses}")
            print(f"  top{args.k}: {[h.image_id for h in hits]}")

        # --- compositional stress test -----------------------------------
        overlaps = []
        print(f"\n  --- colour-swap overlap (lower = better binding) ---")
        for q, q_swapped in SWAP_PAIRS:
            a = {h.image_id for h in r.search(q, k=args.k, use_regions=use_regions)}
            b = {h.image_id for h in r.search(q_swapped, k=args.k, use_regions=use_regions)}
            ov = len(a & b) / args.k
            overlaps.append(ov)
            print(f"  {ov*args.k:.0f}/{args.k}  {q[:44]:44s} vs swapped")

        mean_ov = sum(overlaps) / len(overlaps)
        print(f"\n  mean swap overlap: {mean_ov:.3f}")

        results[label] = {
            "backbone": backbone,
            "use_regions": use_regions,
            "queries": per_query,
            "swap_overlaps": overlaps,
            "mean_swap_overlap": mean_ov,
        }

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(results, indent=1))

    print(f"\n\n{'='*66}\nSUMMARY -- mean colour-swap overlap (lower = better)\n{'='*66}")
    for label, r in results.items():
        print(f"  {r['mean_swap_overlap']:.3f}   {label}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
