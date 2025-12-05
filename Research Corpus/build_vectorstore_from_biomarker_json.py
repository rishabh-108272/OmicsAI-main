import json
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
INPUT_JSON = "data/papers_biomarker.json"
PERSIST_DIR = "chroma_biomarker_lit"

# Small, fast embedding model (good on CPU / integrated GPU)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"{INPUT_JSON} not found. Make sure the file exists.")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {INPUT_JSON}")

    texts = []
    metadatas = []

    for rec in records:
        # Main content for retrieval
        text = rec.get("text", "")
        if not text.strip():
            continue

        texts.append(text)

        meta = {
            "id": rec.get("id"),
            "file_name": rec.get("file_name"),
            "title": rec.get("title"),
            "cancer_type": rec.get("cancer_type"),
            "genes": ", ".join(rec.get("genes", [])),
            "drugs": ", ".join(rec.get("drugs", [])),
        }
        metadatas.append(meta)

    print(f"Building embeddings for {len(texts)} documents...")

    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # Create / overwrite Chroma DB
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing vector store at {PERSIST_DIR}...")
        import shutil
        shutil.rmtree(PERSIST_DIR)

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()

    print(f"✅ Vector store built and saved to '{PERSIST_DIR}'")
    print("Some stored docs (title, genes, drugs):")
    for i in range(min(5, len(metadatas))):
        m = metadatas[i]
        print(f"  - {m['id']}: {m['title']} | genes={m['genes']} | drugs={m['drugs']}")


if __name__ == "__main__":
    main()
