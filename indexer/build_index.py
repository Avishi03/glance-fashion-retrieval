"""Part A: the indexer.

Feature extraction + vector storage. Two vectors kinds per image:

  global  -- the whole frame. Carries scene ("inside a modern office") and
             overall vibe ("casual weekend outfit"), which are properties of the
             image, not of any one garment.
  region  -- one per garment box. This is what fixes compositionality. A dual
             encoder pools the whole frame into a single vector, so "red shirt,
             blue pants" and "blue shirt, red pants" collapse to nearly the same
             point: the model registers {red, blue, shirt, pants} but loses which
             colour attaches to which garment. Embedding the shirt crop alone
             means "red" can only ever bind to the shirt.

No separate "scene vector": with one backbone that would just be the global
vector under another name.

Region boxes come from Fashionpedia's annotations, filtered to real garments --
46% of its boxes are parts (sleeve, neckline, zipper, rivet) and a crop of a
zipper is index noise. COCO rows have no garment boxes and contribute a global
vector only; they are scene evidence and distractors.

Storage is ChromaDB: the brief says to pick the most convenient vector DB rather
than build one, and the ML logic here is in what we embed, not in the store.

Usage:  python -m indexer.build_index --backbone fashion-siglip
"""

import argparse
import json
from pathlib import Path

import chromadb
from PIL import Image
from tqdm import tqdm

from indexer.encoders import Encoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"

MIN_BOX_PX = 32     # smaller crops carry no recognisable garment
BOX_PAD = 0.08      # a little context around the crop helps the encoder


def crop(img, bbox):
    """bbox is [x0, y0, x1, y1]; pad slightly and clamp to the frame."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    px, py = w * BOX_PAD, h * BOX_PAD
    return img.crop((
        max(0, x0 - px), max(0, y0 - py),
        min(img.width, x1 + px), min(img.height, y1 + py),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="fashion-siglip")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    corpus = json.loads((DATA / "corpus.json").read_text())
    enc = Encoder(args.backbone, batch_size=args.batch)
    print(f"backbone {args.backbone} | dim {enc.dim} | device {enc.device}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll_name = f"fashion_{args.backbone.replace('-', '_')}"
    if coll_name in [c.name for c in client.list_collections()]:
        client.delete_collection(coll_name)     # rebuild is idempotent
    coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

    ids, metas, images = [], [], []
    n_global = n_region = 0

    for row in tqdm(corpus, desc="cropping"):
        img = Image.open(DATA / "images" / row["file"]).convert("RGB")

        ids.append(f"{row['image_id']}::global")
        metas.append({
            "image_id": str(row["image_id"]),
            "kind": "global",
            "category": "",
            "source": row["source"],
            "scene": row["scene"] or "",
            "stratum": row["stratum"],
        })
        images.append(img)
        n_global += 1

        for k, obj in enumerate(row["objects"]):
            x0, y0, x1, y1 = obj["bbox"]
            if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
                continue
            ids.append(f"{row['image_id']}::region{k}")
            metas.append({
                "image_id": str(row["image_id"]),
                "kind": "region",
                "category": obj["category"],
                "source": row["source"],
                "scene": row["scene"] or "",
                "stratum": row["stratum"],
            })
            images.append(crop(img, obj["bbox"]))
            n_region += 1

    print(f"encoding {len(images)} crops ({n_global} global + {n_region} region)...")
    embs = enc.encode_images(images)

    CHUNK = 2000
    for i in tqdm(range(0, len(ids), CHUNK), desc="writing"):
        coll.add(
            ids=ids[i : i + CHUNK],
            embeddings=embs[i : i + CHUNK].tolist(),
            metadatas=metas[i : i + CHUNK],
        )

    print(f"\ncollection '{coll_name}': {coll.count()} vectors "
          f"over {len(corpus)} images")


if __name__ == "__main__":
    main()
