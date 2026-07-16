"""Query decomposition -- LookSync stage 1, done zero-shot.

Glance's LookSync (arXiv:2511.00072) decomposes a look into layer-wise garment
descriptions with an LLM before retrieval. We need the same decomposition from a
natural-language query, but with a hard constraint: the brief grades zero-shot
capability -- "how well does the system handle descriptions it hasn't seen
explicitly in a training label?"

That rules out the obvious implementation. A parser with a hardcoded colour list
and garment taxonomy would look like it works on the five graded queries and
quietly fail on "a burnt sienna windbreaker": any term outside the vocabulary is
silently dropped. That is closed-vocabulary retrieval wearing a zero-shot
costume.

So we never enumerate colours or garments. Two steps, both open-vocabulary:

  1. SPLIT on conjunctions and commas. Purely syntactic -- no word list. "a red
     tie and a white shirt in a formal setting" -> ["a red tie", "a white shirt",
     "in a formal setting"].

  2. ROUTE each clause with the text encoder itself, by comparing it to anchor
     prompts describing what a garment is and what a place is. The encoder has
     seen "windbreaker" even if we never wrote it down. Clauses the router is
     unsure about fall back to GLOBAL rather than being dropped -- an unrouted
     clause still contributes through the whole-frame vector, so the system
     degrades to CLIP rather than to nothing.

Routing is a similarity comparison, not a lookup, so the vocabulary is whatever
the backbone knows -- which is the entire point of zero-shot retrieval.
"""

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Role(str, Enum):
    GARMENT = "garment"   # score against region (garment-crop) vectors
    SCENE = "scene"       # score against the global (whole-frame) vector


# Anchors describe the two roles in general terms. They name no specific garment
# and no specific place, so adding a new garment type needs no code change.
ANCHORS = {
    Role.GARMENT: [
        "a piece of clothing",
        "a garment worn by a person",
        "an item of apparel or a fashion accessory",
    ],
    Role.SCENE: [
        "a place or a location",
        "the setting or environment where a photo was taken",
        "a description of a situation, occasion or mood",
    ],
}

# Split points: coordinating conjunctions, commas, and the prepositions that
# introduce a setting. Syntax only -- no semantics, no vocabulary.
SPLIT_RE = re.compile(
    r"\s*(?:,|\band\b|\bwith\b|\bwhile\b|(?=\bin\b)|(?=\bon\b)|(?=\bat\b)|(?=\binside\b))\s*",
    flags=re.I,
)

STOP_CLAUSE = re.compile(r"^(?:a|an|the|someone|somebody|a person|person)?\s*$", re.I)


@dataclass
class Clause:
    text: str
    role: Role
    confidence: float   # margin between the winning role and the other


def split_clauses(query: str):
    """Syntactic split. No word lists, so any vocabulary survives."""
    parts = [p.strip(" .,") for p in SPLIT_RE.split(query)]
    return [p for p in parts if p and not STOP_CLAUSE.match(p)]


def route(clauses, encoder, min_confidence: float = 0.005):
    """Assign each clause a role by similarity to the role anchors.

    Ties and near-ties go to SCENE, which is scored against the global vector --
    the conservative fallback, equivalent to plain CLIP behaviour for that clause.
    """
    if not clauses:
        return []

    roles = list(ANCHORS)
    anchor_texts = [t for r in roles for t in ANCHORS[r]]
    anchor_emb = encoder.encode_texts(anchor_texts)
    # mean-pool each role's anchors into one direction, then renormalise
    sizes = [len(ANCHORS[r]) for r in roles]
    centroids, off = [], 0
    for n in sizes:
        c = anchor_emb[off : off + n].mean(0)
        centroids.append(c / np.linalg.norm(c))
        off += n
    centroids = np.stack(centroids)

    clause_emb = encoder.encode_texts(clauses)
    sims = clause_emb @ centroids.T          # (n_clauses, n_roles)

    out = []
    for text, row in zip(clauses, sims):
        order = np.argsort(-row)
        best, second = order[0], order[1]
        margin = float(row[best] - row[second])
        role = roles[best] if margin >= min_confidence else Role.SCENE
        out.append(Clause(text=text, role=role, confidence=margin))
    return out


def decompose(query: str, encoder):
    """Natural-language query -> routed clauses. Never returns empty."""
    clauses = split_clauses(query)
    if not clauses:
        return [Clause(text=query, role=Role.SCENE, confidence=0.0)]
    return route(clauses, encoder)
