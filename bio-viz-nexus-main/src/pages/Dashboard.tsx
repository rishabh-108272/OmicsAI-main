import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Brain,
  Microscope,
  Pill,
  MessageCircle,
  Atom,
  Upload,
  BarChart3,
  TrendingUp,
  Users,
  Activity,
  ChevronRight,
  Play,
  Download,
  Eye
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const featureCards = [
  {
    title: "AI Classification",
    description: "Upload patient gene expression data and get instant subtype predictions",
    icon: Brain,
    href: "/classification",
    status: "Ready",
    gradient: "from-primary to-primary-light",
    actions: ["Upload Data", "Run Model"]
  },
  {
    title: "XAI Biomarker Discovery",
    description: "Discover significant biomarkers with explainable AI and interactive visualizations",
    icon: Microscope,
    href: "/biomarkers",
    status: "Processing",
    gradient: "from-secondary to-secondary-light",
    actions: ["View Heatmaps", "Export Results"]
  },
  {
    title: "Drug Repurposing",
    description: "Explore drug-biomarker interactions through knowledge graph analysis",
    icon: Pill,
    href: "/drug-discovery",
    status: "New Updates",
    gradient: "from-accent to-accent-light",
    actions: ["Explore Graph", "Generate Report"]
  },
  {
    title: "AI Research Agent",
    description: "Query scientific literature with RAG-powered AI for instant insights",
    icon: MessageCircle,
    href: "/ai-agent",
    status: "Active",
    gradient: "from-primary to-secondary",
    actions: ["Start Chat", "Browse Topics"]
  },
  {
    title: "3D Protein Viewer",
    description: "Interactive AlphaFold3 structures with detailed molecular annotations",
    icon: Atom,
    href: "/protein-viewer",
    status: "Updated",
    gradient: "from-secondary to-accent",
    actions: ["View Structures", "Compare Models"]
  },
  {
    title: "Analytics Dashboard",
    description: "Comprehensive data visualization and statistical analysis tools",
    icon: BarChart3,
    href: "/analytics",
    status: "Ready",
    gradient: "from-accent to-primary",
    actions: ["View Charts", "Export Data"]
  }
];

const recentAnalyses = [
  {
    id: "CRC-001",
    type: "Colorectal Classification",
    patient: "Patient #12847",
    status: "Completed",
    accuracy: 94.7,
    time: "2 minutes ago"
  },
  {
    id: "LUNG-045",
    type: "Lung Cancer Subtyping",
    patient: "Patient #12846",
    status: "Processing",
    accuracy: null,
    time: "5 minutes ago"
  },
  {
    id: "BIO-337",
    type: "Biomarker Discovery",
    patient: "Cohort Study #03",
    status: "Completed",
    accuracy: 98.1,
    time: "1 hour ago"
  }
];

const quickStats = [
  { label: "Active Projects", value: "24", icon: Activity, change: "+12%" },
  { label: "Patients Analyzed", value: "1,247", icon: Users, change: "+5.7%" },
  { label: "Success Rate", value: "97.3%", icon: TrendingUp, change: "+2.1%" },
  { label: "Models Running", value: "8", icon: Brain, change: "0%" }
];

export default function Dashboard() {
  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex flex-col md:flex-row md:items-center justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold">Welcome back, Dr. Saurav</h1>
          <p className="text-muted-foreground">
            Here's an overview of your multi-omics analysis platform
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="hero" className="gap-2">
            <Play className="h-4 w-4" />
            Quick Start
          </Button>
        </div>
      </motion.div>

      {/* Quick Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {quickStats.map((stat, index) => (
          <Card key={stat.label} className="border-none shadow-soft hover:shadow-medium transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className="text-sm text-accent font-medium">{stat.change}</p>
                </div>
                <div className="p-3 bg-gradient-primary rounded-xl">
                  <stat.icon className="h-6 w-6 text-white" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>

      {/* Feature Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold mb-6">Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {featureCards.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 * index }}
              whileHover={{ y: -4 }}
              className="group"
            >
              <Card className="h-full border-none shadow-medium hover:shadow-strong transition-all duration-300 overflow-hidden">
                <div className={`h-2 bg-gradient-to-r ${feature.gradient}`} />
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className={`p-3 bg-gradient-to-r ${feature.gradient} rounded-xl shadow-soft`}>
                      <feature.icon className="h-6 w-6 text-white" />
                    </div>
                    <Badge variant="secondary" className="text-xs">
                      {feature.status}
                    </Badge>
                  </div>
                  <CardTitle className="text-xl group-hover:text-primary transition-colors">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-muted-foreground mb-6 leading-relaxed">
                    {feature.description}
                  </p>
                  <div className="space-y-3">
                    {feature.actions.map((action) => (
                      <Button
                        key={action}
                        asChild
                        variant="ghost" 
                        size="sm"
                        className="w-full justify-between group-hover:bg-white color-black"
                      >
                        <Link to={feature.href}>
                          {action}
                          <ChevronRight className="h-4 w-4" />
                        </Link>
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Recent Analyses */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="grid lg:grid-cols-3 gap-6"
      >
        <Card className="lg:col-span-2 border-none shadow-soft">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Recent Analyses</CardTitle>
              <Button variant="ghost" size="sm">
                View All <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentAnalyses.map((analysis) => (
                <div key={analysis.id} className="flex items-center justify-between p-4 bg-muted/30 rounded-xl">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{analysis.type}</span>
                      <Badge variant={analysis.status === "Completed" ? "default" : "secondary"}>
                        {analysis.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{analysis.patient}</p>
                    {analysis.accuracy && (
                      <div className="flex items-center gap-2 text-sm">
                        <span>Accuracy:</span>
                        <span className="font-medium text-accent">{analysis.accuracy}%</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">{analysis.time}</span>
                    <Button size="sm" variant="ghost">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-none shadow-soft">
          <CardHeader>
            <CardTitle>System Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Processing Power</span>
                <span>87%</span>
              </div>
              <Progress value={87} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Storage Used</span>
                <span>62%</span>
              </div>
              <Progress value={62} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Active Models</span>
                <span>8/12</span>
              </div>
              <Progress value={67} className="h-2" />
            </div>
            <div className="pt-4 border-t">
              <h4 className="font-medium mb-3">Quick Actions</h4>
              <div className="space-y-2">
                <Button variant="outline" size="sm" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Export Report
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Dataset
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}