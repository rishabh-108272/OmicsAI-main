import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Brain, BarChart3, Download, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const classificationResults = [
  { subtype: "Microsatellite Stable", probability: 87.3, color: "bg-primary" },
  { subtype: "Microsatellite Instable", probability: 8.9, color: "bg-secondary" },
  { subtype: "CpG Island Methylator", probability: 3.8, color: "bg-accent" }
];

const modelPerformance = [
  { metric: "Accuracy", value: "94.7%", description: "Overall classification accuracy" },
  { metric: "Sensitivity", value: "92.1%", description: "True positive rate" },
  { metric: "Specificity", value: "96.8%", description: "True negative rate" },
  { metric: "F1 Score", value: "93.4%", description: "Harmonic mean of precision and recall" }
];

export default function Classification() {
  const [selectedModel, setSelectedModel] = useState("colorectal");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleRunAnalysis = () => {
    setIsProcessing(true);
    setTimeout(() => setIsProcessing(false), 3000);
  };

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-primary" />
            AI Classification
          </h1>
          <p className="text-muted-foreground mt-2">
            Upload patient gene expression data for instant subtype prediction
          </p>
        </div>
        <Button variant="hero" onClick={handleRunAnalysis} disabled={isProcessing}>
          {isProcessing ? (
            <>Processing...</>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Run Analysis
            </>
          )}
        </Button>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-1"
        >
          <Card className="border-none shadow-medium">
            <CardHeader>
              <CardTitle>Data Upload</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center">
                <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="font-medium mb-2">Drop your files here</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Supports CSV, TSV, JSON formats
                </p>
                <Button variant="outline">Choose Files</Button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Model Type</label>
                  <select 
                    className="w-full p-3 border rounded-lg bg-background"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="colorectal">Colorectal Cancer</option>
                    <option value="lung">Lung Cancer</option>
                    <option value="proteomics">Proteomics Analysis</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Patient Metadata</label>
                  <div className="grid grid-cols-2 gap-3">
                    <input placeholder="Age" className="p-2 border rounded bg-background" />
                    <select className="p-2 border rounded bg-background">
                      <option>Gender</option>
                      <option>Male</option>
                      <option>Female</option>
                    </select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Results Section */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <Tabs defaultValue="results" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="results">Classification Results</TabsTrigger>
              <TabsTrigger value="performance">Model Performance</TabsTrigger>
              <TabsTrigger value="visualization">Visualizations</TabsTrigger>
            </TabsList>

            <TabsContent value="results" className="space-y-6">
              <Card className="border-none shadow-medium">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Subtype Predictions</CardTitle>
                    <Badge variant="default">Patient #CRC-001</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {classificationResults.map((result, index) => (
                      <motion.div
                        key={result.subtype}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="space-y-2"
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{result.subtype}</span>
                          <span className="text-lg font-bold">{result.probability}%</span>
                        </div>
                        <div className="relative">
                          <Progress value={result.probability} className="h-3" />
                          <div 
                            className={`absolute top-0 left-0 h-3 rounded-full ${result.color} transition-all duration-1000`}
                            style={{ width: `${result.probability}%` }}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                  <div className="mt-6 pt-6 border-t">
                    <div className="flex gap-3">
                      <Button variant="outline" className="gap-2">
                        <Download className="h-4 w-4" />
                        Export Report
                      </Button>
                      <Button variant="secondary" className="gap-2">
                        <BarChart3 className="h-4 w-4" />
                        Detailed Analysis
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="performance" className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                {modelPerformance.map((metric, index) => (
                  <motion.div
                    key={metric.metric}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card className="border-none shadow-soft">
                      <CardContent className="p-6 text-center">
                        <div className="text-3xl font-bold text-primary mb-2">
                          {metric.value}
                        </div>
                        <div className="font-medium mb-1">{metric.metric}</div>
                        <div className="text-sm text-muted-foreground">
                          {metric.description}
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="visualization">
              <Card className="border-none shadow-medium">
                <CardHeader>
                  <CardTitle>Feature Importance Heatmap</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-96 bg-muted/30 rounded-xl flex items-center justify-center">
                    <div className="text-center">
                      <BarChart3 className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        Interactive heatmap visualization will appear here
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </div>
  );
}