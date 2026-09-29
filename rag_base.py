"""
RAG Base — ricerca del candidato più adatto tra i curriculum in resumes/.

Flusso: lettura CV -> chunking -> embeddings -> ChromaDB -> ricerca per similarità
        -> contesto nel prompt -> risposta dell'LLM.

Il provider (Ollama locale o OpenAI) si sceglie nel file .env: vedi llm_provider.py.

Esecuzione:
    poetry run python rag_base.py                          # modalità interattiva
    poetry run python rag_base.py "mi serve un esperto di SEO"   # singola domanda
"""

import os
import sys

import chromadb

from llm_provider import LLM_MODEL, client, embedding_fn, info

DOCUMENTS_DIR = "resumes"
COLLECTION_NAME = "CVs"
SYSTEM_PROMPT = "Sei un assistente HR, specializzato nella ricerca di profili professionali"


def estrai_info(testo: str) -> str:
    """Estrae nome, email e telefono dall'intestazione del CV, in formato JSON."""
    prompt = f"""Estrai le seguenti informazioni dal testo:
- nome completo
- email
- numero di telefono

Restituisci solo un dizionario JSON nel formato:
{{"nome": "...", "email": "...", "phone": "..."}}

Testo: {testo}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def leggi_documenti(documents_dir: str = DOCUMENTS_DIR):
    """Legge i CV e li divide in chunk usando le intestazioni '### ' come separatore."""
    documents, metadatas, ids = [], [], []
    id = 0

    for filename in sorted(os.listdir(documents_dir)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(documents_dir, filename), "r", encoding="utf-8") as file:
            chunks = file.read().replace("\n", ".").split("### ")

        info_candidato = estrai_info(chunks[1])  # chunks[1] = intestazione con i dati personali
        print(f"  {filename}: {info_candidato}")

        for chunk in chunks:
            if not chunk.isspace() and not chunk == "":
                documents.append(chunk)
                metadatas.append({"source": filename, "info": info_candidato})
                ids.append(str(id))
                id += 1

    return documents, metadatas, ids


def crea_collezione(documents, metadatas, ids):
    """Crea la collezione ChromaDB in memoria e inserisce i chunk (gli embeddings li calcola Chroma)."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # distanza coseno, sintassi valida sia in ChromaDB 0.6 sia 1.x
    )
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def crea_prompt(results, user_question: str) -> str:
    """Costruisce il prompt con il contesto recuperato dal vector DB."""
    meta = results["metadatas"][0][0]
    context = (
        f"CONTESTO: nome file {meta['source']} ecco il paragrafo piu' significativo: "
        f"{results['documents'][0][0]} ricorda sempre di menzionare il nome del candidato "
        f"all'inizio e i dati personali alla fine per il contatto, ti lascio tutto qui: {meta['info']}"
    )
    return f"""Dato il seguente contesto {context} rispondi alla domanda dell'utente {user_question}
spiegando che nel file individuato c'e' il profilo piu' adatto.
Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto.
Rispondi in italiano.
"""


def chiedi(collection, user_question: str) -> None:
    """Pipeline RAG completa per una domanda: retrieval + generazione in streaming."""
    results = collection.query(query_texts=[user_question], n_results=1)
    meta = results["metadatas"][0][0]
    print(f"\n[file trovato: {meta['source']} - distanza {results['distances'][0][0]:.3f}]\n")

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": crea_prompt(results, user_question)},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def main():
    print(info())

    print("\nLettura e chunking dei documenti...")
    documents, metadatas, ids = leggi_documenti()
    print(f"Totale chunk: {len(documents)}")

    print("Calcolo embeddings e inserimento in ChromaDB...")
    collection = crea_collezione(documents, metadatas, ids)
    print(f"Documenti nel database: {collection.count()}")

    # Domanda passata da riga di comando
    if len(sys.argv) > 1:
        chiedi(collection, " ".join(sys.argv[1:]))
        return

    # Modalità interattiva
    print("\nScrivi cosa cerchi (oppure 'q' per uscire).")
    while True:
        user_question = input("\nDomanda: ").strip()
        if user_question.lower() in ("q", "quit", "exit"):
            print("Ciao!")
            break
        if user_question:
            chiedi(collection, user_question)


if __name__ == "__main__":
    main()
