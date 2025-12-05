import React, { useState } from "react";

const DrugRepurposing = () => {
  const [cancerType, setCancerType] = useState("LUAD_LUSC");
  const [ppiFile, setPpiFile] = useState(null);
  const [biomarkersInput, setBiomarkersInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    setPpiFile(file || null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults(null);

    if (!ppiFile) {
      setError("Please upload a PPI CSV file.");
      return;
    }

    const biomarkers = biomarkersInput
      .split(/[,;\n]/)
      .map((b) => b.trim())
      .filter(Boolean);

    if (biomarkers.length === 0) {
      setError("Please enter at least one biomarker gene/protein ID.");
      return;
    }

    try {
      setIsLoading(true);

      const formData = new FormData();
      formData.append("cancer_type", cancerType);
      formData.append("ppi_file", ppiFile);
      formData.append("biomarkers", JSON.stringify(biomarkers));

      const response = await fetch("http://localhost:8000/api/classification/drug-repurposing/", {
    method: "POST",
    body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to fetch drug candidates.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderResultsTable = () => {
    if (!results || !results.candidates || results.candidates.length === 0) {
      return (
        <p className="text-sm text-gray-500 mt-4">
          No drug candidates found for the given biomarkers and PPI network.
        </p>
      );
    }

    return (
      <div className="mt-6 overflow-x-auto">
        <table className="min-w-full border border-gray-200 text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-3 py-2 border-b text-left">Rank</th>
              <th className="px-3 py-2 border-b text-left">Drug</th>
              <th className="px-3 py-2 border-b text-left">Target</th>
              <th className="px-3 py-2 border-b text-left">Distance / BFS Hop</th>
              <th className="px-3 py-2 border-b text-left">Score</th>
              <th className="px-3 py-2 border-b text-left">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {results.candidates.map((drug, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-3 py-2 border-b">{idx + 1}</td>
                <td className="px-3 py-2 border-b font-medium">
                  {drug.drug_name || "-"}
                </td>
                <td className="px-3 py-2 border-b">
                  {drug.target || drug.target_gene || "-"}
                </td>
                <td className="px-3 py-2 border-b">
                  {drug.hops_from_biomarker ?? "-"}
                </td>
                <td className="px-3 py-2 border-b">
                  {drug.score !== undefined ? drug.score.toFixed(3) : "-"}
                </td>
                <td className="px-3 py-2 border-b max-w-xs">
                  {drug.evidence || "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-semibold text-center mb-2">
        Drug Repurposing Engine
      </h1>
      <p className="text-center text-gray-600 mb-8 text-sm md:text-base">
        Upload a protein-protein interaction (PPI) network and provide biomarker
        genes to identify candidate drugs via BFS traversal on the PPI graph.
      </p>

      <div className="bg-white shadow-md rounded-xl p-5 md:p-6">
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Cancer Type */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Cancer Type
            </label>
            <select
              value={cancerType}
              onChange={(e) => setCancerType(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="LUAD_LUSC">LUAD / LUSC (NSCLC)</option>
              <option value="CRC">Colorectal Cancer</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Select the cancer-specific PPI network you are using.
            </p>
          </div>

          {/* Biomarkers */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Biomarker Genes / Proteins
            </label>
            <textarea
              value={biomarkersInput}
              onChange={(e) => setBiomarkersInput(e.target.value)}
              placeholder="Example: EGFR, ALK, KRAS&#10;Use commas, semicolons, or new lines to separate entries."
              className="w-full border rounded-lg px-3 py-2 text-sm h-24 resize-y focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              These will be the starting nodes for BFS on the PPI graph.
            </p>
          </div>

          {/* PPI CSV Upload */}
          <div>
            <label className="block text-sm font-medium mb-1">
              PPI Network CSV
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-700
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-full file:border-0
                         file:text-sm file:font-semibold
                         file:bg-blue-50 file:text-blue-700
                         hover:file:bg-blue-100"
            />
            <p className="text-xs text-gray-500 mt-1">
              CSV format could be: source,target,weight / confidence, etc.
            </p>
            {ppiFile && (
              <p className="text-xs text-green-600 mt-1">
                Selected file: {ppiFile.name}
              </p>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-center">
            <button
              type="submit"
              disabled={isLoading}
              className="inline-flex items-center justify-center px-5 py-2.5 rounded-full text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed transition"
            >
              {isLoading ? "Running BFS & Ranking Drugs..." : "Run Drug Repurposing"}
            </button>
          </div>
        </form>
      </div>

      {/* Results */}
      {results && (
        <div className="mt-8 bg-white shadow-md rounded-xl p-5 md:p-6">
          <h2 className="text-xl font-semibold mb-2">
            Candidate Drugs
          </h2>
          {results.biomarkers && (
            <p className="text-xs text-gray-500 mb-2">
              Seed biomarkers: {results.biomarkers.join(", ")}
            </p>
          )}
          {renderResultsTable()}
        </div>
      )}
    </div>
  );
};

export default DrugRepurposing;
