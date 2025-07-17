import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import CrashesExplorer from './components/CrashesExplorer';
import PopulationViewer from './components/PopulationViewer';
import ShellcodeAnalyzer from './components/ShellcodeAnalyzer';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    // Verificar conexión con el backend
    checkConnection();
    
    // Actualizar cada 5 segundos
    const interval = setInterval(() => {
      checkConnection();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch('/api/stats');
      if (response.ok) {
        setIsConnected(true);
        setLastUpdate(new Date());
      } else {
        setIsConnected(false);
      }
    } catch (error) {
      setIsConnected(false);
    }
  };

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>KernelHunter Dashboard</h1>
          <div className="status-indicator">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            <span className="status-text">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
            {lastUpdate && (
              <span className="last-update">
                Last update: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </div>
        </header>

        <Navigation />

        <main className="App-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/crashes" element={<CrashesExplorer />} />
            <Route path="/population" element={<PopulationViewer />} />
            <Route path="/shellcode/:hash" element={<ShellcodeAnalyzer />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 