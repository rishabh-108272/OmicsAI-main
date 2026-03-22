from rag_backend.llm_local import run_llm

ROUTER_PROMPT = """
Classify the user query into one or more of the following categories:
- gene
- drug
- both

Return ONLY one word: "gene", "drug", or "both".

Query: {question}
"""

def route_question(question: str) -> str:
    response = run_llm(ROUTER_PROMPT.format(question=question))
    response = response.lower().strip()

    if "gene" in response and "drug" in response:
        return "both"
    if "drug" in response:
        return "drug"
    return "gene"
