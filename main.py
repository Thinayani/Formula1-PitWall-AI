import os
import json
from typing import Optional, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests as http_requests

from retriever import retrieve, build_context_block, RetrievedChunk

load_dotenv()

OLLAMA_URL  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
MAX_TOKENS  = 1024

SYSTEM_PROMPT = """You are PitWall AI, an expert Formula 1 strategy analyst.
You have access to a database of F1 race results, qualifying data, fastest laps,
pit stop strategies, and lap-by-lap telemetry from 2010 to 2023.

When answering:
- Be precise and specific, cite the race, year, driver, and team when relevant.
- If the context doesn't contain enough information to answer, say so clearly.
  Do not invent statistics or results.
- For strategy questions, explain the reasoning behind decisions when possible.
- Keep answers concise but complete. Use bullet points for multi-part answers.
- You may reference F1 knowledge from your training, but always prioritise
  the retrieved context over general knowledge for specific facts and figures.
"""

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PitWall AI",
    description="F1 Strategy Intelligence powered by RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    season:    Optional[int] = None
    data_type: Optional[str] = None
    top_n:     int           = 5
    stream:    bool          = False


class SourceChunk(BaseModel):
    race_name:    Optional[str]
    season:       Optional[int]
    data_type:    Optional[str]
    score:        float
    text_preview: str


class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceChunk]


# ── RAG helpers ───────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str) -> str:
    return f"""Use the following retrieved F1 data to answer the question.
If the data doesn't contain a direct answer, say so — do not guess.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def chunks_to_sources(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    return [
        SourceChunk(
            race_name    = c.race_name,
            season       = c.season,
            data_type    = c.data_type,
            score        = round(c.score, 4),
            text_preview = c.text[:200] + ("…" if len(c.text) > 200 else ""),
        )
        for c in chunks
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        r = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_status = "online" if r.ok else "offline"
    except Exception:
        ollama_status = "offline"
    return {"status": "ok", "model": OLLAMA_MODEL, "ollama": ollama_status}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks  = retrieve(req.question, top_n=req.top_n, season=req.season, data_type=req.data_type)
    context = build_context_block(chunks)

    try:
        response = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": f"{SYSTEM_PROMPT}\n\n{build_prompt(req.question, context)}",
                "stream": False,
                "options": {"num_predict": MAX_TOKENS},
            },
            timeout=120,
        ).json()
        answer = response.get("response", "No answer generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

    return QueryResponse(
        answer  = answer,
        sources = chunks_to_sources(chunks),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks  = retrieve(req.question, top_n=req.top_n, season=req.season, data_type=req.data_type)
    context = build_context_block(chunks)

    async def token_generator() -> AsyncIterator[str]:
        try:
            with http_requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{build_prompt(req.question, context)}",
                    "stream": True,
                    "options": {"num_predict": MAX_TOKENS},
                },
                stream=True,
                timeout=120,
            ) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            yield chunk["response"]
                        if chunk.get("done"):
                            break
        except Exception as e:
            yield f"Error: {e}"

        sources_json = json.dumps([s.model_dump() for s in chunks_to_sources(chunks)])
        yield f"\n\n__SOURCES__{sources_json}"

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/stats")
def stats():
    try:
        from qdrant_client import QdrantClient as QC
        q    = QC(path="data/qdrant_storage")
        info = q.get_collection("pitwall_f1")
        return {"vectors_indexed": info.points_count, "collection": "pitwall_f1"}
    except Exception as e:
        return {"error": str(e)}