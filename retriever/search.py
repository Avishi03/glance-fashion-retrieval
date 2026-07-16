"""Part B: the retriever.

Scoring, and specifically how attribute binding is enforced.

The problem, from the brief's own hint: CLIP struggles to tell "red shirt with
blue pants" from "blue shirt with red pants". A dual encoder pools the frame into
one vector, so both images land in nearly the same place -- the model knows
{red, blue, shirt, pants} are present but not which colour belongs to which
garment. This is the documented bag-of-words behaviour of contrastive VLMs.

The fix has two parts.

1. REGION SCORING. Each garment clause is scored against garment crops rather
   than the whole frame. A crop of the shirt contains only the shirt, so "red"
   has nothing else to bind to.

2. ONE-TO-ONE ASSIGNMENT. Region scoring alone is not enough, and this is the
   subtle bit. If each clause independently takes its best-matching region, then
   for an image with a RED SHIRT and a WHITE TIE, the clause "a red tie" happily
   matches the red *shirt* crop -- colour similarity dominates garment identity,
   so the swapped image still scores well and we have fixed nothing.

   So garment clauses are assigned to regions one-to-one, greedily by descending
   similarity. Clauses compete: once "a red tie" claims the red crop, "a white
   shirt" cannot also have it. An image genuinely containing a red tie and a
   white shirt satisfies both clauses with two different crops; the colour-swapped
   image cannot satisfy either well. That asymmetry is what the compositional
   stress test measures.

   This works because a garment crop encodes colour AND garment type together,
   and it needs no category lookup -- so it stays zero-shot and open-vocabulary.

Scene clauses score against the global vector: "inside a modern office" is a
property of the frame, not of any garment.

Fusion normalises per clause (z-score across candidates) before combining.
Raw scores are not comparable across clauses or backbones -- SigLIP's sigmoid
loss puts its cosines on a different scale entirely -- so absolute thresholds
would be meaningless.

Scalability: candidate generation is an ANN lookup in Chroma over the whole
corpus, and only the shortlist is scored densely. Cost is O(log N) in the corpus
size, so the same logic holds at 1M images.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from indexer.encoders import Encoder
from retriever.decompose import Role, decompose

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"

CANDIDATES_PER_CLAUSE = 150     # ANN shortlist width per clause
W_PRIOR = 0.5                   # weight on the whole-query global similarity


@dataclass
class Hit:
    image_id: str
    score: float
    detail: dict


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-6 else np.zeros_like(x)


def _greedy_assign(sim: np.ndarray) -> float:
    """sim is (n_clauses, n_regions). Return mean similarity under a one-to-one
    greedy assignment of clauses to distinct regions.

    Greedy rather than Hungarian: queries have <=5 garment clauses, where greedy
    and optimal agree almost always, and it avoids a scipy dependency.
    """
    if sim.size == 0:
        return 0.0
    n_c, n_r = sim.shape
    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    used_c, used_r, total = set(), set(), []
    for c, r in order:
        if c in used_c or r in used_r:
            continue
        used_c.add(int(c))
        used_r.add(int(r))
        total.append(sim[c, r])
        if len(used_c) == n_c or len(used_r) == n_r:
            break
    # clauses with no region left to claim (image has too few garments) score 0
    return float(sum(total) / n_c)


class Retriever:
    def __init__(self, backbone: str = "fashion-siglip"):
        self.enc = Encoder(backbone)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.coll = client.get_collection(f"fashion_{backbone.replace('-', '_')}")

    def _candidates(self, query: str, clauses) -> list:
        """ANN shortlist: union of nearest neighbours for the full query and each
        clause. Done in Chroma over the whole index -- this is the part that stays
        O(log N) at 1M images."""
        texts = [query] + [c.text for c in clauses]
        embs = self.enc.encode_texts(texts)
        ids = set()
        for e in embs:
            res = self.coll.query(
                query_embeddings=[e.tolist()],
                n_results=CANDIDATES_PER_CLAUSE,
                include=["metadatas"],
            )
            ids.update(m["image_id"] for m in res["metadatas"][0])
        return sorted(ids)

    def _fetch(self, image_ids):
        """Pull every stored vector for the shortlisted images."""
        res = self.coll.get(
            where={"image_id": {"$in": list(image_ids)}},
            include=["embeddings", "metadatas"],
        )
        per_image = {i: {"global": None, "regions": []} for i in image_ids}
        for emb, meta in zip(res["embeddings"], res["metadatas"]):
            e = np.asarray(emb, dtype=np.float32)
            if meta["kind"] == "global":
                per_image[meta["image_id"]]["global"] = e
            else:
                per_image[meta["image_id"]]["regions"].append(e)
        return per_image

    def search(self, query: str, k: int = 5, use_regions: bool = True):
        clauses = decompose(query, self.enc)
        garment = [c for c in clauses if c.role == Role.GARMENT]
        scene = [c for c in clauses if c.role == Role.SCENE]

        ids = self._candidates(query, clauses)
        if not ids:
            return []
        store = self._fetch(ids)

        q_emb = self.enc.encode_texts([query])[0]
        g_emb = self.enc.encode_texts([c.text for c in garment]) if garment else None
        s_emb = self.enc.encode_texts([c.text for c in scene]) if scene else None

        prior, g_scores, s_scores = [], [], []
        for i in ids:
            glob = store[i]["global"]
            regions = store[i]["regions"]
            prior.append(float(q_emb @ glob) if glob is not None else 0.0)

            if garment and use_regions and regions:
                R = np.stack(regions)                  # (n_regions, dim)
                g_scores.append(_greedy_assign(g_emb @ R.T))
            elif garment:
                # no regions (COCO rows) or ablation: fall back to the frame.
                # Degrades to CLIP behaviour rather than dropping the image.
                g_scores.append(float((g_emb @ glob).mean()) if glob is not None else 0.0)
            else:
                g_scores.append(0.0)

            s_scores.append(
                float((s_emb @ glob).mean()) if scene and glob is not None else 0.0
            )

        prior = _zscore(np.asarray(prior))
        total = W_PRIOR * prior
        if garment:
            total = total + _zscore(np.asarray(g_scores))
        if scene:
            total = total + _zscore(np.asarray(s_scores))

        order = np.argsort(-total)[:k]
        return [
            Hit(
                image_id=ids[j],
                score=float(total[j]),
                detail={
                    "prior": float(prior[j]),
                    "garment": float(g_scores[j]),
                    "scene": float(s_scores[j]),
                    "clauses": [(c.text, c.role.value, round(c.confidence, 4)) for c in clauses],
                },
            )
            for j in order
        ]
