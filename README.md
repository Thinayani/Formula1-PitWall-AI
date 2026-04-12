# PitWall-AI-F1

F1 strategy intelligence powered by RAG (Retrieval-Augmented Generation).  
Ask anything about race results, tyre strategies, qualifying, pit stops, and lap data from 2010–2023.

---

## Architecture

```
User question
      │
      ▼
 embed_query()          ← OpenAI text-embedding-3-small
      │
      ▼
 Qdrant vector search   ← top-20 candidates (cosine similarity)
      │
      ▼
 CrossEncoder rerank    ← top-5 most relevant chunks
      │
      ▼
 Claude (claude-opus-4-6)  ← synthesize answer with context
      │
      ▼
 Streamlit UI           ← streamed response + source cards
```

---
## Setup

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key
- Anthropic API key

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Start Qdrant

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

---

## Running the pipeline

Run these scripts in order. Each one builds on the previous.

### Step 1 — Ingest race results (Ergast API)

Fetches results, qualifying, fastest laps, and pit stops for seasons 2010–2023.

```bash
python ingest_ergast.py
```

Output: `data/passages/<season>_<round>_<race>.txt`  
Time: ~15 minutes (rate-limited to respect the free API)

### Step 2 — Ingest telemetry (FastF1)

Fetches lap-by-lap telemetry and tyre strategy data for 2018–2023.

```bash
python ingest_fastf1.py
```

Output: `data/passages/<season>_<round>_<race>_telemetry.txt`  
Time: 30–60 minutes on first run (downloads and caches raw data)  
Cache: `data/fastf1_cache/` (~1–2 GB for all seasons)

### Step 3 — Embed and index

Chunks all passages, embeds them with OpenAI, and upserts into Qdrant.

```bash
python embed_and_index.py
```

Time: ~5 minutes  
Cost: ~$0.10–0.20 in OpenAI embedding credits for the full dataset

### Step 4 — Start the API

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Step 5 — Launch the chat UI

In a new terminal:

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

## Example queries

- "Who won the 2021 Abu Dhabi Grand Prix and how did the strategy unfold?"
- "Which driver had the most fastest laps in the 2020 season?"
- "Compare tyre strategies used at Monaco between 2018 and 2019."
- "What is the record fastest pit stop in the dataset?"
- "When was the last time a safety car directly affected the championship outcome?"
- "How many one-stop strategies were used at Silverstone in 2022?"

---

## Project structure

```
pitwall-ai/
├── ingest_ergast.py      # Script 1 — race results from Ergast API
├── ingest_fastf1.py      # Script 2 — telemetry from FastF1
├── embed_and_index.py    # Script 3 — chunk, embed, index to Qdrant
├── retriever.py          # Script 4 — semantic search + reranking
├── main.py               # Script 5 — FastAPI backend
├── app.py                # Script 6 — Streamlit chat UI
├── requirements.txt
├── .env.example
├── data/
│   ├── passages/         # generated text passages
│   └── fastf1_cache/     # FastF1 local cache
└── README.md
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data — race results | Ergast REST API |
| Data — telemetry | FastF1 Python library |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Vector DB | Qdrant |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Anthropic Claude (claude-opus-4-6) |
| API | FastAPI + uvicorn |
| UI | Streamlit |
