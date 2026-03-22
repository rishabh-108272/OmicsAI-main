import os
from typing import List
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

PERSIST_DIR = "chroma_biomarker_lit"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_retriever():
    """Load Chroma vector store and return a retriever-like object."""
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    # In new LangChain, this is a Runnable retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_hf_client() -> InferenceClient:
    """Create a direct Hugging Face Inference client (no LangChain wrapper)."""
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set.\n"
            "Set it in PowerShell:\n"
            "  $env:HUGGINGFACEHUB_API_TOKEN = 'hf_...'\n"
        )

    # google/flan-t5-base is a small text2text-generation model
    return InferenceClient("google/flan-t5-base", token=token)


def pretty_print_docs(docs: List):
    print("\n[Retrieved documents]")
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        print(f"\n--- Doc {i} ---")
        print(f"Title      : {meta.get('title')}")
        print(f"Cancer type: {meta.get('cancer_type')}")
        print(f"Genes      : {meta.get('genes')}")
        print(f"Drugs      : {meta.get('drugs')}")
        snippet = d.page_content[:400].replace("\n", " ")
        print(f"Snippet    : {snippet}...")


def build_prompt(question: str, docs: List) -> str:
    """Build a single prompt string for the LLM using retrieved docs as context."""
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        title = meta.get("title")
        cancer_type = meta.get("cancer_type")
        genes = meta.get("genes")
        drugs = meta.get("drugs")
        text = d.page_content.replace("\n", " ")

        block = (
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Cancer type: {cancer_type}\n"
            f"Genes: {genes}\n"
            f"Drugs: {drugs}\n"
            f"Text: {text[:1000]}...\n"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    prompt = f"""
You are an expert biomedical literature assistant.

You are given several abstracts about NSCLC and colorectal cancer biomarkers, drug targets,
and therapies. Use ONLY the information from these documents to answer the question.

If the answer is uncertain or not clearly supported, say so explicitly.

---------------- CONTEXT ----------------
{context_str}
-----------------------------------------

QUESTION:
{question}

Answer in 2–4 paragraphs, clearly referencing which documents you are drawing from
(e.g., "According to Document 1, ...", "Document 3 suggests ...").
"""
    return prompt.strip()




def call_llm(prompt: str) -> str:
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1  # CPU
    )
    out = generator(prompt, max_new_tokens=512, temperature=0.1)
    return out[0]["generated_text"]



def test_retrieval_only():
    retriever = get_retriever()

    queries = [
        "Evidence for FASN inhibition with TVB-2640 in NSCLC and colorectal cancer",
        "Role of EGFR and KRAS mutations in colorectal cancer treated with cetuximab or panitumumab",
        "Multi-omics biomarkers for colorectal cancer prognosis and therapy",
    ]

    for q in queries:
        print("\n==============================")
        print("QUERY:", q)
        docs = retriever.invoke(q)  # Runnable retriever
        pretty_print_docs(docs)


def test_rag_with_llm():
    retriever = get_retriever()

    question = (
        "Summarize the evidence that FASN is a therapeutic target in NSCLC and colorectal "
        "cancer, and describe how TVB-2640 or other FASN inhibitors have been evaluated."
    )

    print("\n==============================")
    print("RAG QUESTION:", question)

    docs = retriever.invoke(question)
    pretty_print_docs(docs)

    prompt = build_prompt(question, docs)
    print("\n[LLM PROMPT PREVIEW]\n")
    print(prompt[:800], "...\n")

    print("[ANSWER]")
    answer = call_llm(prompt)
    print(answer)


if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIR):
        raise RuntimeError(
            f"Vector store directory '{PERSIST_DIR}' not found. "
            "Run build_vectorstore_from_biomarker_json.py first."
        )

    # 1) Retrieval sanity check
    test_retrieval_only()

    # 2) Full RAG with Hugging Face Inference
    print("\n\nNow testing full RAG with LLM...\n")
    test_rag_with_llm()
