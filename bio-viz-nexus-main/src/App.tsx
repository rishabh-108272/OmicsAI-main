import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import Classification from "./pages/Classification";
import Biomarker from "./pages/Biomarker";
import DrugRepurposing from "./pages/DrugRepurposing";
import AIChat from './pages/AIChat';
import Alphafold from './pages/Alphafold'
import { MainLayout } from "./components/layout/MainLayout";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Landing page */}
          <Route path="/" element={<Landing />} />
          
          {/* Main app routes with layout */}
          <Route path="/" element={<MainLayout />}>
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="classification" element={<Classification />} />
            <Route path="biomarkers" element={<Biomarker />} />
            <Route path="drug-discovery" element={<DrugRepurposing />} />
            <Route path="ai-agent" element={<AIChat />} />
            <Route path="protein-viewer" element={<Alphafold />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
