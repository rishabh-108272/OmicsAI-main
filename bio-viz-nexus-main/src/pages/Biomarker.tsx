import React, { useState } from 'react';

const Biomarker = () => {
  const [selectedModel, setSelectedModel] = useState('lung_cancer');
  const [file, setFile] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setError('');
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError('Please upload a CSV file');
        setFile(null);
        return;
      }
      setFile(file);
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
      const response = await fetch('http://localhost:8000/api/classification/xai/graph/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to generate explanation');
      }

      const data = await response.json();
      setImageData(data.image_base64);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Biomarker Prediction Using XAI</h1>
      <div className="mb-4">
        <label className="mr-2 font-medium">Select Model:</label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="p-2 border rounded"
        >
          <option value="lung_cancer">Lung Cancer</option>
          <option value="colorectal_cancer">Colorectal Cancer</option>
          <option value="liver_cancer">Liver Cancer</option>
        </select>
      </div>
      <div className="mb-4">
        <input type="file" accept=".csv" onChange={handleFileChange} />
      </div>
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
          <img src={`data:image/png;base64,${imageData}`} alt="SHAP Explanation Graph" className="max-w-full rounded shadow" />
        </div>
      )}
    </div>
  );
};

export default Biomarker;
