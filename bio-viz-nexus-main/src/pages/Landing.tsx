import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { 
  Brain, 
  Microscope, 
  Pill, 
  MessageCircle, 
  Atom, 
  BarChart3,
  ChevronRight,
  CheckCircle,
  Zap,
  Shield,
  Globe
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: Brain,
    title: "AI Classification",
    description: "Instant patient subtype prediction using advanced machine learning models for colorectal, lung, and proteomics analysis.",
    color: "text-primary"
  },
  {
    icon: Microscope,
    title: "XAI Biomarker Discovery", 
    description: "Explainable AI reveals critical biomarkers with interactive heatmaps and feature importance analysis.",
    color: "text-secondary"
  },
  {
    icon: Pill,
    title: "Drug Repurposing",
    description: "Knowledge graph-powered drug discovery connecting biomarkers to therapeutic compounds.",
    color: "text-accent"
  },
  {
    icon: MessageCircle,
    title: "AI Research Agent",
    description: "RAG-powered scientific literature analysis with real-time insights and citation tracking.",
    color: "text-primary"
  },
  {
    icon: Atom,
    title: "3D Protein Visualization",
    description: "AlphaFold3 integration for interactive molecular structure exploration and analysis.",
    color: "text-secondary"
  },
  {
    icon: BarChart3,
    title: "Advanced Analytics",
    description: "Comprehensive data visualization and statistical analysis tools for precision medicine.",
    color: "text-accent"
  }
];

const stats = [
  { value: "50,000+", label: "Patients Analyzed" },
  { value: "12,000+", label: "Biomarkers Discovered" },
  { value: "98.7%", label: "Classification Accuracy" },
  { value: "500+", label: "Drug Compounds" }
];

const benefits = [
  "HIPAA-compliant secure cloud infrastructure",
  "Real-time collaborative research environment", 
  "Automated regulatory reporting and documentation",
  "Integration with major EHR systems"
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-hero py-24 px-6">
        <div className="absolute inset-0 bg-black/20" />
        <div className="relative mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Precision Medicine
              <br />
              <span className="bg-gradient-to-r from-white to-accent-light bg-clip-text text-transparent">
                Powered by AI
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-white/90 mb-8 max-w-3xl mx-auto leading-relaxed">
              Transform genomic data into actionable insights with our comprehensive multi-omics platform.
              Accelerate discovery, improve outcomes, personalize treatment.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" variant="secondary" className="text-lg px-8 py-6">
                <Link to="/dashboard">
                  Start Analysis <ChevronRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="text-lg px-8 py-6 border-white text-black hover:bg-white hover:text-primary">
                <Link to="/demo">
                  View Demo
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-muted/30">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-2">
                  {stat.value}
                </div>
                <div className="text-muted-foreground font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6">
        <div className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Complete Multi-Omics Platform
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              From data upload to drug discovery, our integrated platform handles every step 
              of your precision medicine workflow with cutting-edge AI and visualization tools.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="h-full border-none shadow-medium hover:shadow-strong transition-shadow duration-300 group">
                  <CardContent className="p-8">
                    <feature.icon className={`h-12 w-12 ${feature.color} mb-6 group-hover:scale-110 transition-transform duration-300`} />
                    <h3 className="text-xl font-semibold mb-4">{feature.title}</h3>
                    <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-24 bg-gradient-secondary">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                Enterprise-Ready Platform
              </h2>
              <p className="text-xl text-white/90 mb-8 leading-relaxed">
                Built for healthcare organizations and research institutions requiring 
                the highest standards of security, compliance, and performance.
              </p>
              <ul className="space-y-4">
                {benefits.map((benefit, index) => (
                  <motion.li
                    key={benefit}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-center text-white/90"
                  >
                    <CheckCircle className="h-6 w-6 text-accent-light mr-3 flex-shrink-0" />
                    {benefit}
                  </motion.li>
                ))}
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="grid grid-cols-2 gap-6"
            >
              <Card className="border-none shadow-strong bg-white/10 backdrop-blur">
                <CardContent className="p-6 text-center">
                  <Zap className="h-12 w-12 text-accent-light mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Lightning Fast</h3>
                  <p className="text-white/80 text-sm">Process thousands of samples in minutes</p>
                </CardContent>
              </Card>
              <Card className="border-none shadow-strong bg-white/10 backdrop-blur">
                <CardContent className="p-6 text-center">
                  <Shield className="h-12 w-12 text-accent-light mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Secure & Compliant</h3>
                  <p className="text-white/80 text-sm">HIPAA, GDPR, and SOC 2 certified</p>
                </CardContent>
              </Card>
              <Card className="border-none shadow-strong bg-white/10 backdrop-blur col-span-2">
                <CardContent className="p-6 text-center">
                  <Globe className="h-12 w-12 text-accent-light mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">Global Collaboration</h3>
                  <p className="text-white/80 text-sm">Connect with researchers worldwide</p>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6 bg-background">
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to Transform Your Research?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join leading research institutions and healthcare organizations already using 
              OmicsAI to accelerate discovery and improve patient outcomes.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" variant="hero" className="text-lg px-8 py-6">
                <Link to="/dashboard">
                  Get Started Free <ChevronRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="text-lg px-8 py-6">
                <Link to="/contact">
                  Schedule Demo
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}