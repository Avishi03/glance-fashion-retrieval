"""The five graded queries, plus the compositional stress test.

EVAL_QUERIES are verbatim from the brief.

The stress test is the load-bearing experiment. Precision@5 on five queries is
too small a sample to separate two systems -- and with no colour or scene labels
in Fashionpedia, scoring it needs human judgement anyway (LookSync hit the same
wall and fell back to human Mean Opinion Score, arXiv:2511.00072).

So we measure compositionality directly instead, with a counterfactual that
needs no labels at all:

    For a query Q = "a red tie and a white shirt" and its colour-swapped twin
    Q' = "a white tie and a red shirt", a system that binds attributes to
    garments must rank an image differently under Q than under Q'. A bag-of-words
    system sees the same word set in both and ranks them nearly identically.

The metric is SWAP MARGIN: mean(score under Q) - mean(score under Q') over the
top-k retrieved for Q. Vanilla CLIP should sit near zero -- it cannot tell the
two apart. Region scoring with one-to-one assignment should be clearly positive.

This is self-supervised: it needs no ground-truth colour labels, because it asks
whether the system is *sensitive* to binding, not whether any given image is
correct. That sidesteps the missing-annotation problem entirely.
"""

# Verbatim from the brief, section 4.
EVAL_QUERIES = [
    ("attribute", "A person in a bright yellow raincoat."),
    ("contextual", "Professional business attire inside a modern office."),
    ("semantic", "Someone wearing a blue shirt sitting on a park bench."),
    ("style", "Casual weekend outfit for a city walk."),
    ("compositional", "A red tie and a white shirt in a formal setting."),
]

# (query, colour-swapped twin). Swapping only the colour words keeps the bag of
# words identical, so any score difference is attributable to binding alone.
SWAP_PAIRS = [
    ("A red tie and a white shirt in a formal setting.",
     "A white tie and a red shirt in a formal setting."),
    ("A person in a black jacket and white pants.",
     "A person in a white jacket and black pants."),
    ("A woman in a blue top and a red skirt.",
     "A woman in a red top and a blue skirt."),
    ("Someone wearing a green shirt and brown shoes.",
     "Someone wearing a brown shirt and green shoes."),
]
