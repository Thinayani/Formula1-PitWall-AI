import os
import uuid
from typing import Iterator
from pathlib import Path

from dotenv import load_dotenv
# from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PASSAGES_DIR     = Path("data/passages")
COLLECTION_NAME  = "pitwall_f1"
EMBED_MODEL    = "all-MiniLM-L6-v2"
EMBED_DIM      = 384
CHUNK_SIZE       = 600    # characters — tuned for F1 race passages
CHUNK_OVERLAP    = 80
BATCH_SIZE       = 50     # upsert this many vectors at once

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")

# Clients

embed_model = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(path="data/qdrant_storage")
splitter      = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Qdrant collection setup

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
        existing = []
    if COLLECTION_NAME not in existing:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        # Create payload indexes for fast filtering
        qdrant.create_payload_index(COLLECTION_NAME, "season",    PayloadSchemaType.INTEGER)
        qdrant.create_payload_index(COLLECTION_NAME, "race_name", PayloadSchemaType.KEYWORD)
        qdrant.create_payload_index(COLLECTION_NAME, "data_type", PayloadSchemaType.KEYWORD)
        print("  Collection created with indexes on season, race_name, data_type.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

# Metadata extraction
def extract_metadata(filename: str) -> dict:
    """
    Derive metadata from filename.
    Expected pattern: <season>_<round>_<race_slug>[_telemetry].txt
    e.g. 2021_22_abu_dhabi_grand_prix.txt
         2021_22_abu_dhabi_grand_prix_telemetry.txt
    """
    stem = Path(filename).stem  # strip .txt
    is_telemetry = stem.endswith("_telemetry")
    if is_telemetry:
        stem = stem[:-len("_telemetry")]

    parts = stem.split("_", 2)   # season, round, rest
    season     = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    round_num  = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    race_slug  = parts[2].replace("_", " ").title() if len(parts) > 2 else filename

    return {
        "season":    season,
        "round":     round_num,
        "race_name": race_slug,
        "data_type": "telemetry" if is_telemetry else "race_result",
        "source":    filename,
    }

# Embedding

def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed_model.encode(texts, show_progress_bar=False).tolist()


def chunk_file(path: Path) -> list[dict]:
    """
    Read a passage file, split into chunks, return list of
    {text, metadata} dicts ready for embedding.
    """
    text     = path.read_text(encoding="utf-8").strip()
    chunks   = splitter.split_text(text)
    metadata = extract_metadata(path.name)

    return [
        {"text": chunk, "metadata": {**metadata, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]

# Batched upsert

def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def upsert_chunks(chunks: list[dict]):
    """Embed and upsert a list of chunks to Qdrant."""
    texts    = [c["text"] for c in chunks]
    vectors  = embed_texts(texts)

    points = [
        PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vec,
            payload = {**chunk["metadata"], "text": chunk["text"]},
        )
        for chunk, vec in zip(chunks, vectors)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# Main 

def main():
    ensure_collection()

    passage_files = sorted(PASSAGES_DIR.glob("*.txt"))
    if not passage_files:
        print(f"No passage files found in {PASSAGES_DIR}. Run the ingest scripts first.")
        return

    print(f"\nFound {len(passage_files)} passage files.")
    print(f"Chunk size: {CHUNK_SIZE} chars  |  Overlap: {CHUNK_OVERLAP}  |  Batch: {BATCH_SIZE}\n")

    total_chunks   = 0
    total_upserted = 0

    for i, path in enumerate(passage_files, 1):
        print(f"[{i}/{len(passage_files)}] {path.name}", end=" ... ", flush=True)

        chunks = chunk_file(path)
        total_chunks += len(chunks)

        for batch in batched(chunks, BATCH_SIZE):
            try:
                upsert_chunks(batch)
                total_upserted += len(batch)
            except Exception as e:
                import traceback
                print(f"\n  [!] Upsert error: {e}")
                traceback.print_exc()
                break

        print(f"{len(chunks)} chunks")

    # Final collection info
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\nDone.")
    print(f"  Chunks indexed : {total_upserted}")
    print(f"  Qdrant vectors : {info.points_count}")
    print("Next: run python main.py to start the API server")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("[!] OPENAI_API_KEY not set. Add it to your .env file.")
    else:
        main()
