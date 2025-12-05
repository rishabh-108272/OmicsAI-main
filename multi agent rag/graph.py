# graph.py
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from backend.retriever import get_retriever
from agents.router_agent import route_question
from agents.gene_agent import run_gene_agent
from agents.drug_agent import run_drug_agent
from agents.summary_agent import run_summary_agent


# 1) Define the state schema for the graph
class GraphState(TypedDict, total=False):
    question: str
    docs: List[Any]
    route: str
    gene_answer: str
    drug_answer: str
    final_answer: str


def build_graph():
    retriever = get_retriever()

    # --- Node functions -------------------------------------------------

    def retrieve(state: GraphState) -> GraphState:
        """First node: retrieve relevant documents for the question."""
        q = state["question"]
        docs = retriever.invoke(q)
        # Only update docs; question stays in state
        return {"docs": docs}

    def route(state: GraphState) -> GraphState:
        """Router node: decide whether the query is gene-focused, drug-focused, or both."""
        q = state["question"]
        r = route_question(q)
        return {"route": r}

    def gene(state: GraphState) -> GraphState:
        """Gene Agent node."""
        q = state["question"]
        docs = state.get("docs", [])
        gene_answer = run_gene_agent(q, docs)
        return {"gene_answer": gene_answer}

    def drug(state: GraphState) -> GraphState:
        """Drug Agent node."""
        q = state["question"]
        docs = state.get("docs", [])
        drug_answer = run_drug_agent(q, docs)
        return {"drug_answer": drug_answer}

    def summarize(state: GraphState) -> GraphState:
        """Summary Agent node: merge gene + drug answers into a literature review."""
        q = state["question"]
        docs = state.get("docs", [])
        gene_answer = state.get("gene_answer", "")
        drug_answer = state.get("drug_answer", "")

        final_answer = run_summary_agent(
            question=q,
            gene_answer=gene_answer,
            drug_answer=drug_answer,
            docs=docs,
        )
        return {"final_answer": final_answer}

    # --- Build the LangGraph graph -------------------------------------

    graph = StateGraph(GraphState)  # IMPORTANT: pass the state schema here

    # Register nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("route", route)
    graph.add_node("gene", gene)
    graph.add_node("drug", drug)
    graph.add_node("summarize", summarize)

    # Entry point
    graph.set_entry_point("retrieve")

    # retrieve → route
    graph.add_edge("retrieve", "route")

    # route → (gene / drug / both)
    def route_decider(state: GraphState) -> str:
        return state["route"]

    graph.add_conditional_edges(
        "route",
        route_decider,
        {
            "gene": "gene",
            "drug": "drug",
            "both": "drug",
        },
    )

    # gene → summarize
    graph.add_edge("gene", "summarize")

    # drug → summarize
    graph.add_edge("drug", "summarize")

    # summarize → END
    graph.add_edge("summarize", END)

    # Compile into an executable graph
    return graph.compile()
