# retriever.py
# PitWall AI — Script 4/6
# Core RAG retrieval logic:
#   1. Embed the user query
#   2. Semantic search in Qdrant (top-K candidates)
#   3. Rerank with a cross-encoder for precision
#   4. Return top-N results with metadata
#
# This module is imported by main.py — not run directly.

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import CrossEncoder

load_dotenv()

# Config

COLLECTION_NAME   = "pitwall_f1"
EMBED_MODEL       = "text-embedding-3-small"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K       = 20   # how many candidates to retrieve before reranking
FINAL_TOP_N       = 5    # how many to return after reranking

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# Client

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(path="data/qdrant_storage")

# Cross-encoder loaded once at import time (CPU-friendly model, ~25 MB)
reranker = CrossEncoder(RERANKER_MODEL)


# Data models

@dataclass
class RetrievedChunk:
    text:       str
    score:      float          # reranker score (higher = more relevant)
    season:     Optional[int]
    round:      Optional[int]
    race_name:  Optional[str]
    data_type:  Optional[str]  # "race_result" or "telemetry"
    source:     Optional[str]

    def to_context_string(self) -> str:
        """Format for injection into the LLM prompt."""
        header = f"[{self.race_name} {self.season} — {self.data_type}]"
        return f"{header}\n{self.text}"


# Query embedding

def embed_query(query: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    return response.data[0].embedding


# Qdrant search

def build_filter(season: Optional[int] = None, data_type: Optional[str] = None) -> Optional[Filter]:
    """
    Build optional Qdrant payload filter.
    Useful for scoped queries like 'only 2021 season' or 'telemetry only'.
    """
    conditions = []

    if season is not None:
        conditions.append(
            FieldCondition(key="season", match=MatchValue(value=season))
        )

    if data_type is not None:
        conditions.append(
            FieldCondition(key="data_type", match=MatchValue(value=data_type))
        )

    if not conditions:
        return None

    from qdrant_client.models import Filter as QFilter, Must
    return QFilter(must=conditions)


def vector_search(
    query_vector: list[float],
    k: int = CANDIDATE_K,
    season: Optional[int] = None,
    data_type: Optional[str] = None,
) -> list[dict]:
    """Run cosine similarity search in Qdrant, return raw hits."""
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k,
        query_filter=build_filter(season, data_type),
        with_payload=True,
        score_threshold=0.30,   # discard very weak matches early
    )
    return [
        {
            "text":      hit.payload.get("text", ""),
            "payload":   hit.payload,
            "raw_score": hit.score,
        }
        for hit in results
    ]


# Reranking

def rerank(query: str, candidates: list[dict], top_n: int = FINAL_TOP_N) -> list[dict]:
    """
    Score each candidate against the query using a cross-encoder.
    Cross-encoders attend to both query and document together — much more
    accurate than cosine similarity alone, but too slow to run on all vectors.
    That's why we first narrow down to CANDIDATE_K with vector search.
    """
    if not candidates:
        return []

    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


# Public interface

def retrieve(
    query: str,
    top_n: int = FINAL_TOP_N,
    season: Optional[int] = None,
    data_type: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline: embed → vector search → rerank → return chunks.

    Args:
        query:     Natural language question from the user.
        top_n:     Number of final results to return.
        season:    Optional filter — restrict to a specific F1 season.
        data_type: Optional filter — "race_result" or "telemetry".

    Returns:
        List of RetrievedChunk objects, sorted by relevance (best first).
    """
    query_vector = embed_query(query)
    candidates   = vector_search(query_vector, k=CANDIDATE_K, season=season, data_type=data_type)

    if not candidates:
        return []

    reranked = rerank(query, candidates, top_n=top_n)

    return [
        RetrievedChunk(
            text      = r["text"],
            score     = r["rerank_score"],
            season    = r["payload"].get("season"),
            round     = r["payload"].get("round"),
            race_name = r["payload"].get("race_name"),
            data_type = r["payload"].get("data_type"),
            source    = r["payload"].get("source"),
        )
        for r in reranked
    ]


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Join retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labelled with its source for traceability.
    """
    if not chunks:
        return "No relevant data found in the PitWall database."
    return "\n\n---\n\n".join(c.to_context_string() for c in chunks)
