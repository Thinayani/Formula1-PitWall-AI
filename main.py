import os
import json
from typing import Optional, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

from retriever import retrieve, build_context_block, RetrievedChunk

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL      = "gemini-1.5-pro"
MAX_TOKENS     = 1024

genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai.GenerativeModel(
    model_name=LLM_MODEL,
    system_instruction="""You are PitWall AI, an expert Formula 1 strategy analyst.
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
)

# App setup

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

# Request/response models

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


# RAG helpers

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


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks  = retrieve(req.question, top_n=req.top_n, season=req.season, data_type=req.data_type)
    context = build_context_block(chunks)

    response = gemini_client.generate_content(
        build_prompt(req.question, context),
        generation_config=genai.types.GenerationConfig(max_output_tokens=MAX_TOKENS),
    )

    answer = response.text if response.text else "No answer generated."

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
        response = gemini_client.generate_content(
            build_prompt(req.question, context),
            generation_config=genai.types.GenerationConfig(max_output_tokens=MAX_TOKENS),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

        # Emit sources footer after the answer
        sources_json = json.dumps([s.model_dump() for s in chunks_to_sources(chunks)])
        yield f"\n\n__SOURCES__{sources_json}"

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/stats")
def stats():
    try:
        from qdrant_client import QdrantClient as QC
        q    = QC(path="data/qdrant_storage")
        info = q.get_collection("pitwall_f1")
        return {"vectors_indexed": info.vectors_count, "collection": "pitwall_f1"}
    except Exception as e:
        return {"error": str(e)}