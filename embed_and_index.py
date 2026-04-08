import os

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

PASSAGES_DIR     = Path("data/passages")
COLLECTION_NAME  = "pitwall_f1"
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 1536
CHUNK_SIZE       = 600    # characters — tuned for F1 race passages
CHUNK_OVERLAP    = 80
BATCH_SIZE       = 50     # upsert this many vectors at once

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY   = os.getenv("", "")

# Clients

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant        = QdrantClient(url=QDRANT_URL)
splitter      = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Qdrant collection setup

def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
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