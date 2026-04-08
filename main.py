import os
from typing import Optional, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic

from retriever import retrieve, build_context_block, RetrievedChunk

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-opus-4-6"
MAX_TOKENS        = 1024

SYSTEM_PROMPT = """You are PitWall AI, an expert Formula 1 strategy analyst.
You have access to a database of F1 race results, qualifying data, fastest laps,
pit stop strategies, and lap-by-lap telemetry from 2010 to 2023.

When answering:
- Be precise and specific — cite the race, year, driver, and team when relevant.
- If the context doesn't contain enough information to answer, say so clearly.
  Do not invent statistics or results.
- For strategy questions, explain the reasoning behind decisions when possible.
- Keep answers concise but complete. Use bullet points for multi-part answers.
- You may reference F1 knowledge from your training, but always prioritise
  the retrieved context over general knowledge for specific facts and figures.
"""

# App setup
app = FastAPI(
    title="PitWall AI",
    description="F1 Strategy Intelligence powered by RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Request/response models

class QueryRequest(BaseModel):
    question:  str
    season:    Optional[int]   = None   # filter to a specific year
    data_type: Optional[str]   = None   # "race_result" or "telemetry"
    top_n:     int             = 5
    stream:    bool            = False

class SourceChunk(BaseModel):
    race_name:  Optional[str]
    season:     Optional[int]
    data_type:  Optional[str]
    score:      float
    text_preview: str          # first 200 chars of the chunk

class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceChunk]

# RAG

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
            race_name     = c.race_name,
            season        = c.season,
            data_type     = c.data_type,
            score         = round(c.score, 4),
            text_preview  = c.text[:200] + ("…" if len(c.text) > 200 else ""),
        )
        for c in chunks
    ]

# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}
