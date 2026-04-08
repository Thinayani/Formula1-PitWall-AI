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
