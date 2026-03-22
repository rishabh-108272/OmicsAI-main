# classification/ai_agent.py

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Locate "multi agent rag" folder and load its .env
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../server/classification
RAG_DIR = BASE_DIR / "multi agent rag"              # .../server/classification/multi agent rag

# Make sure Python can import graph.py from that folder
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# Load the local .env for Gemini / HF keys used by the agents
env_path = RAG_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()   # fallback to project-level .env if any

# We will lazily import the graph app, so Django can start even if
# langgraph is not installed. The first call will raise a clear error.
_app_cache = None


def _get_graph_app():
    """
    Lazy import of the LangGraph `app` from graph.py.
    This avoids hard crashes on Django startup if dependencies are missing.
    """
    global _app_cache
    if _app_cache is not None:
        return _app_cache

    try:
        from graph import app  # type: ignore
        logger.info("[AI Agent] Successfully imported multi-agent RAG graph app.")
        _app_cache = app
        return _app_cache
    except ModuleNotFoundError as e:
        # Usually means `langgraph` is not installed
        logger.exception("[AI Agent] LangGraph or graph.py dependency missing: %s", e)
        raise RuntimeError(
            "Multi-agent RAG graph cannot be loaded. "
            "Make sure `langgraph` is installed in this venv:\n"
            "    pip install 'langgraph[all]'"
        ) from e
    except Exception as e:
        logger.exception("[AI Agent] Failed to import graph.app: %s", e)
        raise RuntimeError(f"Failed to import multi-agent graph app: {e}") from e


# -----------------------------------------------------------------------------
# 2. Helper: serialise LangChain / LangGraph docs to JSON-safe dicts
# -----------------------------------------------------------------------------
def _serialize_docs(raw_docs: Any) -> List[Dict[str, Any]]:
    """
    Convert LangChain Document objects (or any other type) into JSON-safe dicts
    that your React frontend can display.
    """
    docs_out: List[Dict[str, Any]] = []

    if not raw_docs:
        return docs_out

    for idx, d in enumerate(raw_docs):
        try:
            page_content = getattr(d, "page_content", None)
            metadata = getattr(d, "metadata", {}) or {}

            doc_dict = {
                "id": getattr(d, "id", None)
                or metadata.get("id")
                or f"doc-{idx}",
                "title": getattr(d, "title", None)
                or metadata.get("title")
                or metadata.get("paper_title")
                or f"Document {idx + 1}",
                "cancer_type": getattr(d, "cancer_type", None)
                or metadata.get("cancer_type")
                or metadata.get("cancer"),
                "year": getattr(d, "year", None) or metadata.get("year"),
                "page_content": page_content if page_content is not None else str(d),
                "metadata": metadata,
            }

            docs_out.append(doc_dict)
        except Exception as inner_e:
            logger.warning("[AI Agent] Failed to serialise doc %s: %s", idx, inner_e)
            docs_out.append(
                {"id": f"doc-{idx}", "page_content": str(d), "metadata": {}}
            )

    return docs_out


# -----------------------------------------------------------------------------
# 3. Public function called by Django view
# -----------------------------------------------------------------------------
def run_multi_agent_rag(question: str) -> Dict[str, Any]:
    """
    Run the multi-agent RAG graph for the given question and return a
    JSON-serialisable dict for the frontend.

    This is what your Django `views.py` will call.
    """
    app = _get_graph_app()  # may raise RuntimeError if graph/langgraph not available

    logger.info("[AI Agent] Running multi-agent graph for question: %s", question)

    # Shape must match what graph.py expects. Most examples use {"question": ...}
    state = {"question": question}

    # Invoke the LangGraph app
    result = app.invoke(state)

    # Extract fields according to your graph state in graph.py
    final_answer = (
        result.get("final_answer")
        or result.get("answer")
        or "No final_answer field produced by the multi-agent graph."
    )

    route = result.get("route")
    gene_answer = result.get("gene_answer")
    drug_answer = result.get("drug_answer")

    raw_docs = (
        result.get("docs")
        or result.get("retrieved_docs")
        or result.get("documents")
        or []
    )
    docs = _serialize_docs(raw_docs)

    payload: Dict[str, Any] = {
        "final_answer": final_answer,
        "route": route,
        "gene_answer": gene_answer,
        "drug_answer": drug_answer,
        "docs": docs,
    }

    logger.info(
        "[AI Agent] Graph finished. Route=%s, gene_answer_len=%s, drug_answer_len=%s, docs=%d",
        route,
        len(gene_answer or ""),
        len(drug_answer or ""),
        len(docs),
    )

    return payload
