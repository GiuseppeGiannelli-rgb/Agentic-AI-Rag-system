"""
Livello di astrazione sul provider AI.

Il resto del progetto usa solo:
    client          -> client OpenAI-compatibile per la chat
    LLM_MODEL       -> nome del modello di chat
    embedding_fn    -> funzione di embedding per ChromaDB

Il provider si sceglie nel file .env:
    PROVIDER=ollama   -> tutto in locale con Ollama (default se manca la API key)
    PROVIDER=openai   -> OpenAI (richiede OPENAI_API_KEY)
Se PROVIDER non è impostato, si usa OpenAI quando è presente OPENAI_API_KEY, altrimenti Ollama.
"""

import os

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

CONFIG = {
    "openai": {
        "llm_model": os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    },
    "ollama": {
        "llm_model": os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
    },
}


def _scegli_provider() -> str:
    provider = os.getenv("PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    if provider not in CONFIG:
        raise ValueError(f"PROVIDER non valido: '{provider}'. Usa 'ollama' oppure 'openai'.")
    return provider


def _verifica_ollama(modelli: list[str]) -> None:
    import ollama

    try:
        installati = [m.model for m in ollama.Client(host=OLLAMA_URL).list().models]
    except Exception as e:
        raise RuntimeError(f"Ollama non raggiungibile su {OLLAMA_URL}: avvia l'app Ollama. ({e})")
    for modello in modelli:
        if not any(nome.split(":")[0] == modello.split(":")[0] for nome in installati):
            raise RuntimeError(f"Modello mancante: esegui  ollama pull {modello}")


PROVIDER = _scegli_provider()
LLM_MODEL = CONFIG[PROVIDER]["llm_model"]
EMBED_MODEL = CONFIG[PROVIDER]["embed_model"]

if PROVIDER == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("PROVIDER=openai ma OPENAI_API_KEY è vuota nel file .env")
    client = OpenAI(api_key=api_key)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )
else:
    _verifica_ollama([LLM_MODEL, EMBED_MODEL])
    # Ollama espone un'API compatibile con OpenAI: stesso client, cambia solo l'indirizzo
    client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )


def info() -> str:
    return f"Provider: {PROVIDER} | LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL}"
