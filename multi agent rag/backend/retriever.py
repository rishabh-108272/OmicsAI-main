from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_biomarker_lit"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": 4})
