import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, Brain, Play, FileX, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type PatientMetadata = {
  age: string;
  gender: string;
  patientId: string;
};

type ClassificationResult = {
  label: string;
  confidence: number;
  probability: number;
};

type ModelPerformanceMetric = {
  metric: string;
  value: string;
  description: string;
};

export default function Classification() {
  const [selectedModel, setSelectedModel] = useState("lung_cancer");
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformanceMetric[]>([]);
  const [patientMetadata, setPatientMetadata] = useState<PatientMetadata>({ age: "", gender: "", patientId: "" });
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files ? e.target.files[0] : null;
    if (file) {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError("Please upload a CSV file only");
        setUploadedFile(null);
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError("File size should be less than 10MB");
        setUploadedFile(null);
        return;
      }
      setUploadedFile(file);
      setError("");
      setClassificationResult(null);
      setModelPerformance([]);
      setSessionId(null);
    }
  }, []);

  const handleRunAnalysis = async () => {
    if (!uploadedFile) {
      setError("Please upload a CSV file before running analysis");
      return;
    }
    if (!patientMetadata.age || !patientMetadata.gender) {
      setError("Please fill in patient metadata (age and gender)");
      return;
    }
    setIsProcessing(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('model_type', selectedModel);
      formData.append('age', patientMetadata.age);
      formData.append('gender', patientMetadata.gender);
      formData.append('patient_id', patientMetadata.patientId || `AUTO-${Date.now()}`);

      const response = await fetch('http://localhost:8000/api/classification/analyze/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      setClassificationResult(data.classification_results);
      setModelPerformance(data.model_performance);
      setSessionId(data.session_id);
    } catch (err: unknown) {
      let message = "Analysis failed";
      if (err instanceof Error) {
        message = `Analysis failed: ${err.message}`;
      }
      setError(message);
      setClassificationResult(null);
      setModelPerformance([]);
      setSessionId(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleMetadataChange = (field: keyof PatientMetadata, value: string) => {
    setPatientMetadata(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="p-6 space-y-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-primary" />
            AI Classification
          </h1>
          <p className="text-muted-foreground mt-2">Upload single patient RNA-seq CSV for classification</p>
        </div>

        {/*
          Model Type Selector Dropdown now including Liver Cancer option
        */}
        <select
          className="mr-4 p-2 border rounded bg-background"
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
          disabled={isProcessing}
        >
          <option value="lung_cancer">Lung Cancer</option>
          <option value="colorectal_cancer">Colorectal Cancer</option>
          <option value="liver_cancer">Liver Cancer</option>
        </select>

        <Button variant="hero" onClick={handleRunAnalysis} disabled={isProcessing || !uploadedFile}>
          {isProcessing ? "Processing..." : (<><Play className="h-4 w-4 mr-2" />Run Analysis</>)}
        </Button>
      </motion.div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid lg:grid-cols-3 gap-8">

        {/* Upload and Metadata Column */}
        <div className="lg:col-span-1">
          <Card className="border-none shadow-medium">
            <CardHeader><CardTitle>Data Upload</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center relative">
                {!uploadedFile ? (
                  <>
                    <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="font-medium mb-2">Drop your single patient CSV file here</p>
                    <p className="text-sm text-muted-foreground mb-4">Expect gene names as row index, and one patient expression column</p>
                    <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="file-upload" />
                    <Button variant="outline" onClick={() => document.getElementById('file-upload')!.click()}>Choose CSV File</Button>
                  </>
                ) : (
                  <div className="space-y-2">
                    <p className="font-medium text-green-600">File Uploaded</p>
                    <p className="text-sm">{uploadedFile.name}</p>
                    <p className="text-xs text-muted-foreground">{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    <Button variant="outline" size="sm" onClick={() => setUploadedFile(null)} className="mt-2">
                      <FileX className="h-4 w-4 mr-1" /> Remove
                    </Button>
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Patient Metadata</label>
                  <div className="space-y-3">
                    <input
                      placeholder="Patient ID (optional)"
                      className="w-full p-2 border rounded bg-background"
                      value={patientMetadata.patientId}
                      onChange={e => handleMetadataChange('patientId', e.target.value)}
                    />
                    <div className="grid grid-cols-2 gap-3">
                      <input
                        placeholder="Age *"
                        type="number"
                        min={0}
                        max={150}
                        className="p-2 border rounded bg-background"
                        value={patientMetadata.age}
                        onChange={e => handleMetadataChange('age', e.target.value)}
                        required
                      />
                      <select
                        className="p-2 border rounded bg-background"
                        value={patientMetadata.gender}
                        onChange={e => handleMetadataChange('gender', e.target.value)}
                        required
                      >
                        <option value="">Gender *</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Results and Model Performance Column */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="results" className="space-y-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="results">Classification Result</TabsTrigger>
              <TabsTrigger value="performance">Model Performance</TabsTrigger>
            </TabsList>

            <TabsContent value="results">
              <Card className="border-none shadow-medium p-6">
                <CardTitle>Prediction Result for Patient ID: {patientMetadata.patientId || 'N/A'}</CardTitle>
                {classificationResult ? (
                  <>
                    <p>Predicted Class: <b>{classificationResult.label}</b></p>
                    <p>Confidence: <b>{classificationResult.confidence}%</b></p>
                    <p>Probability: <b>{classificationResult.probability}%</b></p>
                  </>
                ) : (
                  <p className="text-muted-foreground">Run analysis to see prediction results.</p>
                )}
              </Card>
            </TabsContent>

            <TabsContent value="performance">
              <Card className="border-none shadow-medium p-6">
                <CardTitle>Model Performance Metrics</CardTitle>
                {modelPerformance.length > 0 ? (
                  <ul>
                    {modelPerformance.map((metric, idx) => (
                      <li key={idx}>
                        <b>{metric.metric}</b>: {metric.value} — <em>{metric.description}</em>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-muted-foreground">Run analysis to see model performance metrics.</p>
                )}
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
