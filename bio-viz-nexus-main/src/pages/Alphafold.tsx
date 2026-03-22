// src/pages/Alphafold.jsx (or wherever you put it)
import React, { useState, useEffect, useRef } from "react";
import * as NGL from "ngl"; // import the viewer

const BACKEND_BASE = "http://localhost:8000"; // change if different

const Alphafold = () => {
  const [accession, setAccession] = useState("Q5VSL9"); // default example
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Define a simple type structure for bestModel for type safety (optional but good)
  const [bestModel, setBestModel] = useState<{
    pdbUrl: string;
    globalMetricValue: number | null;
    paeImageUrl?: string;
    uniprotId?: string;
    uniprotAccession?: string;
    uniprotDescription?: string;
    organismScientificName?: string;
  } | null>(null);

  const viewerRef = useRef<HTMLDivElement>(null); // div for NGL, specify type
  const stageRef = useRef<NGL.Stage | null>(null); // keep track of NGL Stage, specify type

  // 🧠 Fetch AlphaFold prediction from your Django API
  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    setBestModel(null);

    try {
      const res = await fetch(
        `${BACKEND_BASE}/api/alphafold/prediction/${accession}/`
      );

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to fetch AlphaFold data");
      }

      const data = await res.json(); // this is an array
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error("No models found for this UniProt accession.");
      }

      // 🔍 Pick the best model (highest globalMetricValue)
      const best = data.reduce((bestSoFar, item) => {
        if (!bestSoFar) return item;
        if ((item.globalMetricValue || 0) > (bestSoFar.globalMetricValue || 0)) {
          return item;
        }
        return bestSoFar;
      }, null);

      setBestModel(best);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  // 🎨 NGL Viewer Initialization and Rendering Effect
  useEffect(() => {
    // 1. Initial Checks
    if (!bestModel || !bestModel.pdbUrl || !viewerRef.current) {
      return;
    }

    // 2. Cleanup Previous Stage
    if (stageRef.current) {
      stageRef.current.removeAllComponents(); 
      stageRef.current.dispose();
      stageRef.current = null;
    }

    // 3. Initialize NGL Stage
    const stage = new NGL.Stage(viewerRef.current, {
      backgroundColor: "white",
    });
    stageRef.current = stage;

    // 4. Handle Resize
    const handleResize = () => stage.handleResize();
    window.addEventListener("resize", handleResize);

    // 5. Load File and Add Representation
    stage
      .loadFile(bestModel.pdbUrl, { ext: "pdb" })
      .then((component) => {
        // 🛑 FIX: Ensure component exists (for TypeScript and runtime safety)
        if (!component) {
          console.warn("NGL loadFile resolved without a component.");
          return;
        }

        // 💡 COLOR FIX: Register the Gradient color scheme
        // Note: The key part is ensuring NGL.ColormakerRegistry.addScheme is correctly called
        // and the scheme name matches the colorScheme parameter below.
        NGL.ColormakerRegistry.addScheme(function (params) {
          this.atomColor = function (atom) {
            // Safely calculate the max residue number (floor at 1)
            const maxResno = atom.structure.atomCount > 0 
              ? atom.structure.getAtomProxy(atom.structure.atomCount - 1).resno 
              : 1;
            
            // Normalize residue index (resno) to 0-1 range
            const t = atom.resno / maxResno;

            // Simple Blue-to-Red gradient based on sequence position (Blue start, Red end)
            const r = Math.floor(255 * t);      
            const g = Math.floor(70 * t);       
            const b = Math.floor(255 * (1 - t)); 

            return (r << 16) | (g << 8) | b; // Return color as a packed integer
          };
        }, "proteinGradientColor"); // <- Scheme is registered under this key

        // ✅ Apply the registered scheme
        component.addRepresentation("cartoon", {
          colorScheme: "proteinGradientColor", // Must match the name above!
          smoothSheet: true,
          radiusScale: 1.2,
        });

        stage.autoView();
      })
      .catch((err) => console.error("NGL load error:", err));

    // 6. Cleanup Function
    return () => {
      window.removeEventListener("resize", handleResize);
      if (stageRef.current) {
        stageRef.current.removeAllComponents();
        stageRef.current.dispose();
        stageRef.current = null;
      }
    };
  }, [bestModel]); // Depend on bestModel to re-run on new data

  // --- JSX Rendering (Unchanged) ---
  return (
    <div className="flex flex-col items-center px-4 py-6">
      <h1 className="text-2xl font-semibold mb-4">
        AlphaFold for 3D Protein Structure Prediction
      </h1>
      
      <div className="flex flex-wrap gap-2 items-center mb-6">
        <input
          type="text"
          value={accession}
          onChange={(e) => setAccession(e.target.value)}
          placeholder="Enter UniProt accession, e.g. Q5VSL9"
          className="border border-gray-300 rounded px-3 py-2 min-w-[260px]"
        />
        <button
          onClick={fetchPrediction}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? "Fetching..." : "Fetch Structure"}
        </button>
      </div>

      {loading && <p className="text-gray-600 mb-4">Loading structure data...</p>}
      {error && (
        <p className="text-red-600 mb-4 max-w-xl text-center">Error: {error}</p>
      )}

      {bestModel && (
        <div className="w-full max-w-6xl">
          {/* Basic info */}
          <div className="mb-4 text-left">
            <h2 className="text-xl font-semibold mb-1">
              {bestModel.uniprotId} ({bestModel.uniprotAccession})
            </h2>
            <p className="text-gray-700">
              {bestModel.uniprotDescription} —{" "}
              <span className="italic">{bestModel.organismScientificName}</span>
            </p>
            <p className="text-gray-700 mt-1">
              Global pLDDT:{" "}
              <span className="font-semibold">
                {bestModel.globalMetricValue?.toFixed(2)}
              </span>
            </p>
          </div>

          {/* Layout: 3D viewer + PAE image */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* NGL 3D viewer */}
            <div>
              <h3 className="font-semibold mb-2 text-left">
                3D Structure (Residue Index Gradient)
              </h3>
              <div
                ref={viewerRef}
                className="border border-gray-300 rounded"
                style={{ width: "100%", height: "400px" }}
              />
            </div>

            {/* PAE image if available */}
            <div>
              <h3 className="font-semibold mb-2 text-left">
                Predicted Aligned Error (PAE)
              </h3>
              {bestModel.paeImageUrl ? (
                <img
                  src={bestModel.paeImageUrl}
                  alt="Predicted Aligned Error"
                  className="border border-gray-300 rounded max-w-full"
                  style={{ aspectRatio: "1/1", objectFit: "contain" }}
                />
              ) : (
                <p className="text-gray-600">
                  No PAE image available for this model.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Alphafold;