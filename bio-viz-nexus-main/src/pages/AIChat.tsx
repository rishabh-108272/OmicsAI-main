import React, { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:8000/api/classification/ai-agent/";

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
  const messagesEndRef = useRef(null);

  // Auto-scroll messages
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
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

      const finalAnswer = data.final_answer || "No final answer returned.";
      const route = data.route || null;
      const geneAnswer = data.gene_answer || "";
      const drugAnswer = data.drug_answer || "";

      const docs =
        data.docs || data.retrieved_docs || data.documents || [];

      const agents = {
        ...(route && {
          Router: { answer: `Query routed as: ${route.toUpperCase()}` },
        }),
        ...(geneAnswer && { "Gene Agent": { answer: geneAnswer } }),
        ...(drugAnswer && { "Drug Agent": { answer: drugAnswer } }),
      };

      const assistantMsg = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        type: "answer",
        content: finalAnswer,
        agents: Object.keys(agents).length ? agents : null,
        retrievedDocs: docs,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      setError(err.message);

      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          type: "error",
          content:
            "❗ I couldn’t get a response from the backend multi-agent pipeline. Please check server logs.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Render Agent Blocks (light mode)
  const renderAgentBlocks = (agents) => {
    if (!agents) return null;

    return (
      <div className="mt-4 space-y-3 border-t border-slate-300 pt-3">
        <p className="text-xs font-medium uppercase tracking-wide text-blue-700">
          Multi-Agent Reasoning
        </p>

        <div className="grid gap-3 md:grid-cols-3">
          {Object.entries(agents).map(([name, info]) => (
            <div
              key={name}
              className="rounded-xl border border-slate-300 bg-white p-3 shadow-sm"
            >
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-700">
                {name}
              </span>
              <p className="text-xs text-slate-700 whitespace-pre-wrap mt-1">
                {info.answer || "No agent output provided."}
              </p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render Retrieved Documents
  const renderDocs = (docs) => {
    if (!docs || docs.length === 0) return null;

    return (
      <div className="mt-3 rounded-xl bg-white border border-slate-300 p-3 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-purple-700 mb-2">
          Key Literature Context
        </p>

        <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
          {docs.map((doc, idx) => {
            const meta = doc.metadata || {};
            const title =
              doc.title ||
              meta.title ||
              meta.paper_title ||
              `Document ${idx + 1}`;
            const text =
              doc.page_content || meta.abstract || "(no preview available)";

            const snippet =
              text.length > 260 ? text.slice(0, 257) + "…" : text;

            return (
              <div
                key={idx}
                className="rounded-lg bg-slate-50 px-2 py-2 text-xs border border-slate-200"
              >
                <strong className="text-slate-700">{title}</strong>
                <p className="text-[11px] text-slate-600 mt-1">{snippet}</p>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-[calc(100vh-80px)] w-full flex-col bg-slate-100 text-slate-800">
      {/* Header */}
      <header className="border-b border-slate-300 bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-slate-800">
              Multi-Agentic RAG Assistant
            </h1>
            <p className="text-xs text-slate-500">
              NSCLC / CRC biomarkers • drug repurposing • literature synthesis
            </p>
          </div>

          <span className="rounded-full bg-green-100 text-green-700 px-3 py-1 text-[11px] font-medium border border-green-300">
            Agentic • LLM • RAG
          </span>
        </div>
      </header>

      {/* Main Section */}
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-4 gap-4">
        <div className="flex-1 overflow-hidden rounded-2xl border border-slate-300 bg-white shadow-md">
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
                    className={`max-w-[90%] rounded-2xl px-4 py-2.5 text-sm shadow ${
                      m.role === "user"
                        ? "bg-blue-500 text-white rounded-br-none"
                        : m.type === "error"
                        ? "bg-red-100 border border-red-300 text-red-800 rounded-bl-none"
                        : "bg-slate-100 border border-slate-300 text-slate-800 rounded-bl-none"
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
                  <div className="flex items-center gap-2 rounded-2xl bg-yellow-100 border border-yellow-300 px-3 py-2 text-xs text-yellow-700">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-yellow-600" />
                    Running multi-agent graph…
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Error */}
            {error && (
              <div className="border-t border-red-300 bg-red-50 px-4 py-2 text-xs text-red-700">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Input */}
            <form
              onSubmit={handleSubmit}
              className="border-t border-slate-300 bg-white px-4 py-3"
            >
              {/* Example questions toggle */}
              <div className="mb-2">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen(!advancedOpen)}
                  className="text-[11px] text-slate-600 hover:text-slate-800"
                >
                  {advancedOpen ? "▾ Hide" : "▸ Show"} example questions
                </button>
              </div>

              {advancedOpen && (
                <div className="mb-3 rounded-xl border border-slate-300 bg-slate-50 p-3">
                  <label className="text-[11px] font-semibold uppercase text-slate-700 mb-1 block">
                    Example questions
                  </label>

                  <div className="flex flex-wrap gap-2">
                    {[
                      "Explain the role of FASN in NSCLC and CRC.",
                      "Which drugs target EGFR in NSCLC and CRC and what evidence supports them?",
                      "Summarize biomarkers relevant for TVB-2640 in colorectal cancer.",
                    ].map((ex) => (
                      <button
                        key={ex}
                        type="button"
                        onClick={() => setQuestion(ex)}
                        className="rounded-full bg-white border border-slate-300 px-2 py-1 text-[11px] text-slate-700 hover:bg-slate-100"
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-end gap-2">
                <textarea
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything about biomarkers, drugs, or literature…"
                  className="flex-1 resize-none rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-500 focus:ring-1 focus:ring-blue-400 outline-none"
                />

                <button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  className={`rounded-xl px-4 py-2 text-sm font-semibold shadow transition-colors ${
                    isLoading || !question.trim()
                      ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                      : "bg-blue-500 text-white hover:bg-blue-400"
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
