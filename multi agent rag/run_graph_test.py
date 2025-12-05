from graph import build_graph

graph = build_graph()

question = (
    "What evidence supports FASN as a therapeutic biomarker and how do "
    "FASN inhibitors like TVB-2640 perform in NSCLC and colorectal cancer?"
)

result = graph.invoke({"question": question})

print("\n==== FINAL MULTI-AGENT LITERATURE REVIEW ====\n")
print(result["final_answer"])
