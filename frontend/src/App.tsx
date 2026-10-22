import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from './components/common/ProtectedRoute';
import Navbar from './components/common/Navbar';
import Login from './pages/Login';
import CommandCenter from './pages/CommandCenter';
import LiveMonitoring from './pages/LiveMonitoring';
import ForensicSearch from './pages/ForensicSearch';
import CameraHealthPage from './pages/CameraHealth';
import CameraManagement from './pages/CameraManagement';
import EventHistory from './pages/EventHistory';
import AnalyticsReport from './pages/AnalyticsReport';
import EmailSettings from './pages/EmailSettings';
import ModelTesting from './pages/ModelTesting';
import './App.css';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-zinc-950 text-white flex flex-col font-sans">
        <Routes>
          {/* Public Login */}
          <Route path="/login" element={<Login />} />

          {/* Protected Command Center Routes */}
          <Route
            path="*"
            element={
              <ProtectedRoute>
                <div className="flex flex-col min-h-screen">
                  <Navbar />
                  <div className="flex-1">
                    <Routes>
                      <Route path="/" element={<CommandCenter />} />
                      <Route path="/monitoring" element={<LiveMonitoring />} />
                      <Route path="/forensics" element={<ForensicSearch />} />
                      <Route path="/health" element={<CameraHealthPage />} />
                      <Route path="/cameras" element={<CameraManagement />} />
                      <Route path="/events" element={<EventHistory />} />
                      <Route path="/reports" element={<AnalyticsReport />} />
                      <Route path="/settings/email" element={<EmailSettings />} />
                      <Route path="/email" element={<EmailSettings />} />
                      <Route path="/model-test" element={<ModelTesting />} />
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </div>
                </div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
};

export default App;
