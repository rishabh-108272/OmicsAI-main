import { useState } from "react";
import { motion } from "framer-motion";
import { 
  Upload as UploadIcon, 
  FileText, 
  CheckCircle, 
  AlertCircle, 
  X,
  Eye,
  Download
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface UploadedFile {
  id: string;
  name: string;
  size: string;
  type: string;
  status: "uploading" | "completed" | "error";
  progress: number;
}

export default function Upload() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([
    {
      id: "1",
      name: "patient_gene_expression.csv",
      size: "2.4 MB",
      type: "Gene Expression",
      status: "completed",
      progress: 100
    },
    {
      id: "2", 
      name: "clinical_metadata.json",
      size: "156 KB",
      type: "Clinical Data",
      status: "completed",
      progress: 100
    },
    {
      id: "3",
      name: "proteomics_data.tsv",
      size: "5.7 MB", 
      type: "Proteomics",
      status: "uploading",
      progress: 67
    }
  ]);

  const removeFile = (id: string) => {
    setUploadedFiles(files => files.filter(f => f.id !== id));
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
            <UploadIcon className="h-8 w-8 text-primary" />
            Data Upload
          </h1>
          <p className="text-muted-foreground mt-2">
            Upload patient data and metadata for multi-omics analysis
          </p>
        </div>
        <Button variant="hero">
          Start Analysis
        </Button>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Zone */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="border-none shadow-medium">
            <CardHeader>
              <CardTitle>Upload Files</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-primary/25 rounded-xl p-12 text-center bg-gradient-to-br from-primary/5 to-secondary/5 hover:from-primary/10 hover:to-secondary/10 transition-colors">
                <UploadIcon className="h-16 w-16 text-primary mx-auto mb-6" />
                <h3 className="text-xl font-semibold mb-3">Drop files here or click to browse</h3>
                <p className="text-muted-foreground mb-6">
                  Supports CSV, TSV, JSON, Excel files up to 100MB
                </p>
                <div className="space-y-3">
                  <Button variant="hero" size="lg" className="w-full">
                    Choose Files
                  </Button>
                  <Button variant="outline" size="lg" className="w-full">
                    Upload from URL
                  </Button>
                </div>
              </div>

              <div className="mt-8 space-y-4">
                <h4 className="font-semibold">Supported File Types:</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">CSV</Badge>
                    <span>Gene expression data</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">TSV</Badge>
                    <span>Proteomics data</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">JSON</Badge>
                    <span>Clinical metadata</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">XLSX</Badge>
                    <span>Mixed datasets</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* File List */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="border-none shadow-medium">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Uploaded Files ({uploadedFiles.length})</CardTitle>
                <Button variant="outline" size="sm">
                  Clear All
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {uploadedFiles.map((file, index) => (
                  <motion.div
                    key={file.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="border rounded-xl p-4 space-y-3"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <FileText className="h-5 w-5 text-primary mt-1" />
                        <div className="space-y-1">
                          <div className="font-medium">{file.name}</div>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <span>{file.size}</span>
                            <span>•</span>
                            <Badge variant="outline" className="text-xs">
                              {file.type}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {file.status === "completed" && (
                          <CheckCircle className="h-5 w-5 text-accent" />
                        )}
                        {file.status === "error" && (
                          <AlertCircle className="h-5 w-5 text-destructive" />
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(file.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {file.status === "uploading" && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Uploading...</span>
                          <span>{file.progress}%</span>
                        </div>
                        <Progress value={file.progress} />
                      </div>
                    )}

                    {file.status === "completed" && (
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="gap-2">
                          <Eye className="h-4 w-4" />
                          Preview
                        </Button>
                        <Button variant="outline" size="sm" className="gap-2">
                          <Download className="h-4 w-4" />
                          Download
                        </Button>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Data Validation */}
          <Card className="border-none shadow-medium mt-6">
            <CardHeader>
              <CardTitle>Data Validation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-accent/10 rounded-lg">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-accent" />
                    <span className="font-medium">Format Validation</span>
                  </div>
                  <Badge variant="default">Passed</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-accent/10 rounded-lg">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-accent" />
                    <span className="font-medium">Data Integrity</span>
                  </div>
                  <Badge variant="default">Passed</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-muted-foreground" />
                    <span className="font-medium">Sample Size Check</span>
                  </div>
                  <Badge variant="secondary">Pending</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}