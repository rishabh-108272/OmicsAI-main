import React, { useState } from 'react';

const Biomarker = () => {
  // Default model – you can change this to 'lung_cancer' or 'liver_cancer' if you like
  const [selectedModel, setSelectedModel] = useState('colorectal_cancer');
  const [file, setFile] = useState<File | null>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError('');
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
        setError('Please upload a CSV file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setImageData(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please upload a CSV file');
      return;
    }

    setLoading(true);
    setError('');
    setImageData(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', selectedModel);

    try {
      const response = await fetch(
        'http://localhost:8000/api/classification/xai/graph/',
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate explanation');
      }

      setImageData(data.image);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to generate explanation');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Biomarker Prediction Using XAI</h1>

      {/* Model selector */}
      <div className="mb-4">
        <label className="block mb-2 font-medium">Target Model</label>
        <select
          value={selectedModel}
          onChange={(e) => {
            setSelectedModel(e.target.value);
            setImageData(null);
            setError('');
          }}
          className="border rounded px-3 py-2"
        >
          <option value="lung_cancer">Lung Cancer (LUAD vs LUSC)</option>
          <option value="colorectal_cancer">Colorectal Cancer</option>
          <option value="liver_cancer">Liver Cancer</option>
        </select>
      </div>

      {/* File upload */}
      <div className="mb-4">
        <label className="block mb-2 font-medium">Upload Patient CSV</label>
        <input type="file" accept=".csv" onChange={handleFileChange} />
      </div>

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
      >
        {loading ? 'Generating...' : 'Generate Explanation'}
      </button>

      {error && <p className="text-red-600 mt-2">{error}</p>}

      {imageData && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-3">
            Biomarker Importance (XAI)
          </h2>
          <img
            src={`data:image/png;base64,${imageData}`}
            alt="Biomarker XAI Plot"
            className="max-w-full border rounded shadow"
          />
        </div>
      )}
    </div>
  );
};

export default Biomarker;
