"""Backbone wrappers.

Everything goes through open_clip so the bake-off is apples-to-apples: one
library, one embedding convention, one normalisation. (transformers 5.x reshuffled
CLIP's projection internals and `get_text_features` no longer returns a
joint-space vector, which is an easy way to silently produce garbage
similarities.)

Note the two backbones are NOT score-comparable. SigLIP trains with a pairwise
sigmoid loss rather than CLIP's softmax-over-batch InfoNCE, so its cosine
similarities occupy a completely different range -- vanilla CLIP scores a random
image around +0.18 where FashionSigLIP gives -0.05. Never compare raw scores
across backbones, and never threshold on an absolute value. Fusion normalises
per query (see retriever/search.py).
"""

from dataclasses import dataclass

import numpy as np
import open_clip
import torch

BACKBONES = {
    # name            arch                                    pretrained
    "vanilla-clip":   ("ViT-B-32", "openai"),
    "fashion-siglip": ("hf-hub:Marqo/marqo-fashionSigLIP", None),
    # patrickjohncyh/fashion-clip is deliberately absent: it ships in
    # transformers format only, with no open_clip_pytorch_model.bin, so it
    # cannot join this bake-off without a second code path. Marqo's published
    # numbers already place FashionSigLIP above it (R@1 0.121 vs 0.077).
}


@dataclass
class Encoder:
    name: str
    device: str = "cuda"
    batch_size: int = 32

    def __post_init__(self):
        if self.name not in BACKBONES:
            raise ValueError(f"unknown backbone {self.name!r}; have {list(BACKBONES)}")
        arch, pretrained = BACKBONES[self.name]
        if not torch.cuda.is_available():
            self.device = "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)

    @property
    def dim(self) -> int:
        """Derived empirically rather than read off the model.

        open_clip exposes the embedding width inconsistently across towers --
        SigLIP's vision side is a TimmModel with no .output_dim -- so encoding a
        throwaway string is the one method that works for every backbone.
        """
        if getattr(self, "_dim", None) is None:
            self._dim = int(self.encode_texts(["x"]).shape[-1])
        return self._dim

    @torch.no_grad()
    def encode_images(self, images) -> np.ndarray:
        """PIL images -> L2-normalised embeddings, (n, dim) float32."""
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([
                self.preprocess(im) for im in images[i : i + self.batch_size]
            ]).to(self.device)
            f = self.model.encode_image(batch).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    @torch.no_grad()
    def encode_texts(self, texts) -> np.ndarray:
        """Strings -> L2-normalised embeddings, (n, dim) float32."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            toks = self.tokenizer(texts[i : i + self.batch_size]).to(self.device)
            f = self.model.encode_text(toks).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)
