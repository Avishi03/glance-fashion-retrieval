"""Measure the environment axis of the sampled corpus.

Fashionpedia ships no scene labels, so we estimate coverage with zero-shot CLIP
over a fixed set of scene prompts.

Caveat, and it matters: these tags are a PROXY for corpus composition, not
evaluation ground truth. Grading a CLIP-based retriever against CLIP-generated
labels would be circular. We use vanilla OpenAI CLIP here (not the FashionSigLIP
retrieval backbone) and only to answer one question: do office / park / street
images exist in this corpus at all, or is it mostly studio and runway shots?

The 'studio backdrop' and 'runway' prompts are the control: if they dominate,
Fashionpedia cannot serve the environment axis and the corpus needs a top-up.
"""

import json
from collections import Counter
from pathlib import Path

import open_clip
import torch
from PIL import Image

DATA = Path(__file__).parent / "data"
SCENES = [
    "an office interior",
    "an urban street",
    "a public park",
    "a home interior, a living room",
    "a plain studio backdrop, a product photo",
    "a fashion runway, a catwalk show",
    "a red carpet event",
    "a beach",
]
REQUIRED = SCENES[:4]  # the four environments the brief names
BATCH = 32


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    with torch.no_grad():
        toks = tokenizer([f"a photo of a person in {s}" for s in SCENES]).to(device)
        tfeat = model.encode_text(toks)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    corpus = json.loads((DATA / "corpus.json").read_text())
    counts, margins = Counter(), []

    for i in range(0, len(corpus), BATCH):
        chunk = corpus[i : i + BATCH]
        batch = torch.stack([
            preprocess(Image.open(DATA / "images" / c["file"]).convert("RGB"))
            for c in chunk
        ]).to(device)
        with torch.no_grad():
            ifeat = model.encode_image(batch)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims = ifeat @ tfeat.T

        top2 = sims.topk(2, dim=-1)
        for j in range(len(chunk)):
            counts[SCENES[top2.indices[j, 0].item()]] += 1
            margins.append((top2.values[j, 0] - top2.values[j, 1]).item())
        print(f"  tagged {min(i+BATCH, len(corpus))}/{len(corpus)}", end="\r")

    n = len(corpus)
    print("\n\n=== estimated scene distribution (zero-shot CLIP -- PROXY, not ground truth) ===")
    for scene, c in counts.most_common():
        print(f"{c:5d}  ({100*c/n:5.1f}%)  {scene}")

    real = sum(counts[s] for s in REQUIRED)
    print(f"\nfour REQUIRED environments: {real}/{n} ({100*real/n:.1f}%)")
    for s in REQUIRED:
        print(f"    {counts[s]:4d}  {s}")
    print(f"\nmean top1-top2 margin: {sum(margins)/len(margins):.4f} "
          f"(small => CLIP is unsure => treat these tags as weak)")


if __name__ == "__main__":
    main()
