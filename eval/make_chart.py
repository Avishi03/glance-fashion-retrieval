"""Ablation chart for the report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
results = json.loads((ROOT / "eval" / "results.json").read_text())

labels = list(results)
vals = [results[l]["mean_swap_overlap"] for l in labels]
short = ["vanilla CLIP\n(global only)", "FashionSigLIP\n(global only)", "FashionSigLIP\n+ regions"]
colors = ["#c0392b", "#e67e22", "#27ae60"]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
bars = ax.bar(short, vals, color=colors, width=0.58)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
            ha="center", fontsize=12, fontweight="bold")

ax.set_ylabel("colour-swap overlap @5   (lower = better)", fontsize=10)
ax.set_title("Compositional binding: do 'red tie + white shirt' and\n"
             "'white tie + red shirt' return the same images?", fontsize=11, pad=12)
ax.set_ylim(0, 0.78)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out = ROOT / "eval" / "ablation.png"
fig.savefig(out)
print(f"wrote {out}")
