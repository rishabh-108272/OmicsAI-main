import React, { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:8000/api/multi-agent-rag/"; // ⬅️ change if needed

const AIChat = () => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      id: "sys-1",
      role: "assistant",
      type: "system",
      content:
        "Hi! I’m your Multi-Agent RAG assistant for NSCLC & CRC biomarker and drug literature. Ask me about genes, drugs, mechanisms, or evidence.",
      agents: null,
      retrievedDocs: [],
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [topK, setTopK] = useState(4);
  const [cancerType, setCancerType] = useState("ANY");
  const messagesEndRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const trimmed = question.trim();
    if (!trimmed) return;

    // Add user message to chat
    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      type: "question",
      content: trimmed,
      agents: null,
      retrievedDocs: [],
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmed,
          top_k: topK,
          cancer_type: cancerType === "ANY" ? null : cancerType,
        }),
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(
          `Backend error (${resp.status}): ${
            txt?.slice(0, 200) || "No response body"
          }`
        );
      }

      const data = await resp.json();

      // Normalize response in case some fields are missing
      const finalAnswer =
        data.final_answer ||
        data.answer ||
        "Backend responded, but no final answer field was found.";
      const agents = data.agents || {};
      const retrievedDocs = data.retrieved_docs || data.docs || [];

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        type: "answer",
        content: finalAnswer,
        agents,
        retrievedDocs,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Multi-agent RAG error:", err);
      setError(err.message || "Something went wrong while calling the backend.");
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          type: "error",
          content:
            "❗ I couldn’t get a response from the backend multi-agent pipeline. Please check the server logs and try again.",
          agents: null,
          retrievedDocs: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderAgentBlocks = (agents) => {
    if (!agents || Object.keys(agents).length === 0) return null;

    return (
      <div className="mt-4 space-y-3 border-t border-slate-700/60 pt-3">
        <p className="text-xs font-medium uppercase tracking-wide text-sky-300">
          Multi-Agent Reasoning
        </p>
        <div className="grid gap-3 md:grid-cols-3">
          {Object.entries(agents).map(([name, info]) => (
            <div
              key={name}
              className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-3 shadow-sm"
            >
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-200">
                  {name.replace(/_/g, " ")}
                </span>
              </div>
              <p className="text-xs text-slate-200 whitespace-pre-wrap">
                {info?.answer || info?.summary || "No agent output provided."}
              </p>
              {info?.notes && (
                <p className="mt-1 text-[11px] text-slate-400 whitespace-pre-wrap">
                  <span className="font-semibold text-slate-300">
                    Notes:&nbsp;
                  </span>
                  {info.notes}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderDocs = (docs) => {
    if (!docs || docs.length === 0) return null;

    return (
      <div className="mt-3 rounded-xl bg-slate-900/70 border border-slate-700/60 p-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-violet-300 mb-2">
          Key Papers Retrieved
        </p>
        <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
          {docs.map((doc, idx) => (
            <div
              key={doc.id || idx}
              className="rounded-lg bg-slate-950/60 px-2 py-1.5 text-xs"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-slate-100">
                  {doc.title || `Paper ${idx + 1}`}
                </span>
                <span className="text-[10px] text-slate-400">
                  {doc.cancer_type || "Unknown type"}
                  {doc.year ? ` • ${doc.year}` : ""}
                </span>
              </div>
              {doc.genes && doc.genes.length > 0 && (
                <p className="mt-1 text-[11px] text-slate-300">
                  <span className="font-semibold">Genes:</span>{" "}
                  {doc.genes.join(", ")}
                </p>
              )}
              {doc.drugs && doc.drugs.length > 0 && (
                <p className="mt-0.5 text-[11px] text-slate-300">
                  <span className="font-semibold">Drugs:</span>{" "}
                  {doc.drugs.join(", ")}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-[calc(100vh-80px)] w-full flex-col bg-slate-950/95 text-slate-50">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              Multi-Agentic RAG Assistant
            </h1>
            <p className="text-xs text-slate-400">
              NSCLC / CRC biomarkers • drug repurposing • literature synthesis
            </p>
          </div>
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-[11px] font-medium text-emerald-300 border border-emerald-600/40">
            Agentic · Local LLM · RAG
          </span>
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-4 gap-4">
        {/* Chat area */}
        <div className="flex-1 overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/60 shadow-lg">
          <div className="flex h-full flex-col">
            {/* Messages */}
            <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex ${
                    m.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[90%] rounded-2xl px-4 py-2.5 text-sm shadow-sm ${
                      m.role === "user"
                        ? "bg-sky-600 text-white rounded-br-none"
                        : m.type === "error"
                        ? "bg-rose-900/70 text-rose-50 border border-rose-700/70"
                        : "bg-slate-900/80 text-slate-100 border border-slate-700/70 rounded-bl-none"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{m.content}</p>
                    {m.role === "assistant" && m.type === "answer" && (
                      <>
                        {renderAgentBlocks(m.agents)}
                        {renderDocs(m.retrievedDocs)}
                      </>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs text-slate-300 border border-slate-700/70">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                    Running multi-agent graph (retriever → gene → drug →
                    summarizer)…
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Error banner */}
            {error && (
              <div className="border-t border-rose-800 bg-rose-950/70 px-4 py-2 text-xs text-rose-100">
                <span className="font-semibold">Error:</span> {error}
              </div>
            )}

            {/* Input area */}
            <form
              onSubmit={handleSubmit}
              className="border-t border-slate-800 bg-slate-950/90 px-4 py-3"
            >
              {/* Advanced controls */}
              <div className="mb-2 flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen((o) => !o)}
                  className="text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
                >
                  {advancedOpen ? "▾ Hide" : "▸ Show"} advanced controls
                </button>
              </div>

              {advancedOpen && (
                <div className="mb-2 grid gap-3 rounded-xl border border-slate-800 bg-slate-950/80 p-3 md:grid-cols-3">
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-medium uppercase tracking-wide text-slate-300">
                      Top-K Documents
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={topK}
                      onChange={(e) =>
                        setTopK(Math.min(10, Math.max(1, Number(e.target.value))))
                      }
                      className="w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    />
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-medium uppercase tracking-wide text-slate-300">
                      Cancer Type Filter
                    </label>
                    <select
                      value={cancerType}
                      onChange={(e) => setCancerType(e.target.value)}
                      className="w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    >
                      <option value="ANY">Any (NSCLC + CRC + mixed)</option>
                      <option value="NSCLC">NSCLC only</option>
                      <option value="CRC">CRC only</option>
                    </select>
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-medium uppercase tracking-wide text-slate-300">
                      Example questions
                    </label>
                    <div className="flex flex-wrap gap-1">
                      {[
                        "Explain the role of FASN in NSCLC and CRC.",
                        "Which drugs target EGFR in NSCLC and CRC and what evidence supports them?",
                        "Summarize biomarkers relevant for TVB-2640 in colorectal cancer.",
                      ].map((ex) => (
                        <button
                          key={ex}
                          type="button"
                          onClick={() => setQuestion(ex)}
                          className="rounded-full border border-slate-700 bg-slate-900/80 px-2 py-1 text-[11px] text-slate-200 hover:border-sky-500 hover:text-sky-200"
                        >
                          {ex.length > 55 ? ex.slice(0, 52) + "…" : ex}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Textarea + button */}
              <div className="flex items-end gap-2">
                <textarea
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask about a gene, drug, or biomarker context. Example: “What is the evidence for targeting FASN with TVB-2640 in NSCLC and CRC?”"
                  className="flex-1 resize-none rounded-2xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
                />
                <button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  className={`rounded-2xl px-4 py-2 text-sm font-semibold shadow-sm transition-colors ${
                    isLoading || !question.trim()
                      ? "bg-slate-700 text-slate-300 cursor-not-allowed"
                      : "bg-sky-500 text-slate-950 hover:bg-sky-400"
                  }`}
                >
                  {isLoading ? "Thinking…" : "Ask"}
                </button>
              </div>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AIChat;
