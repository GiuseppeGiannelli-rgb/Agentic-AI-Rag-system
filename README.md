# RAG Base

Ricerca del candidato più adatto tra 4 curriculum con una pipeline RAG:
chunking → embeddings → ChromaDB → ricerca per similarità → risposta dell'LLM.

Il provider AI è intercambiabile e si sceglie nel file `.env`:

| `PROVIDER` | LLM | Embeddings | Requisiti |
|---|---|---|---|
| `ollama` (default) | `llama3.2` | `bge-m3` | Ollama avviato, modelli scaricati |
| `openai` | `gpt-4o-mini` | `text-embedding-3-small` | `OPENAI_API_KEY` nel `.env` |

## Setup

```bash
cp .env.example .env              # se .env non esiste
ollama pull llama3.2              # solo per Ollama
ollama pull bge-m3                # solo per Ollama
poetry config virtualenvs.in-project true
poetry install
```

## Esecuzione

```bash
poetry run python rag_base.py                                 # modalità interattiva
poetry run python rag_base.py "mi serve qualcuno per promuovere il mio prodotto"
```

## Struttura

- `llm_provider.py` — sceglie il provider e fornisce `client`, `LLM_MODEL`, `embedding_fn`
- `rag_base.py` — la pipeline RAG, indipendente dal provider
- `resumes/` — i 4 curriculum di esempio
- `.env.example` — modello di configurazione (il `.env` reale non va su Git)
