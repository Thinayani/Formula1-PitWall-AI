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

#App setup
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