import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const CrashesExplorer = () => {
  const [crashes, setCrashes] = useState([]);
  const [criticalCrashes, setCriticalCrashes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCrash, setSelectedCrash] = useState(null);

  useEffect(() => {
    fetchCrashes();
    const interval = setInterval(fetchCrashes, 15000); // Actualizar cada 15 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchCrashes = async () => {
    try {
      setLoading(true);
      const [crashesResponse, criticalResponse] = await Promise.all([
        fetch('/api/crashes'),
        fetch('/api/critical')
      ]);

      if (crashesResponse.ok && criticalResponse.ok) {
        const crashesData = await crashesResponse.json();
        const criticalData = await criticalResponse.json();
        
        setCrashes(crashesData.crashes || []);
        setCriticalCrashes(criticalData.critical_crashes || []);
      }
    } catch (error) {
      console.error('Error fetching crashes:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCrashSeverity = (crashType) => {
    if (crashType.includes('SIGILL') || crashType.includes('SIGTRAP')) {
      return 'critical';
    } else if (crashType.includes('SIGSEGV') || crashType.includes('SIGBUS')) {
      return 'high';
    } else if (crashType.includes('SIGFPE')) {
      return 'medium';
    } else {
      return 'low';
    }
  };

  const getCrashIcon = (crashType) => {
    if (crashType.includes('SIGILL')) return '🚨';
    if (crashType.includes('SIGSEGV')) return '💥';
    if (crashType.includes('SIGFPE')) return '⚠️';
    if (crashType.includes('SIGTRAP')) return '🔍';
    return '❓';
  };

  const filteredCrashes = () => {
    let allCrashes = [];
    
    if (filter === 'all' || filter === 'normal') {
      allCrashes = [...crashes];
    }
    if (filter === 'all' || filter === 'critical') {
      allCrashes = [...allCrashes, ...criticalCrashes];
    }

    if (searchTerm) {
      allCrashes = allCrashes.filter(crash => 
        crash.crash_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        crash.shellcode_hex?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        crash.generation?.toString().includes(searchTerm)
      );
    }

    return allCrashes.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
  };

  const formatShellcode = (shellcode) => {
    if (!shellcode) return 'N/A';
    return shellcode.length > 32 ? shellcode.substring(0, 32) + '...' : shellcode;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleString();
  };

  if (loading) {
    return <div className="loading">Loading crashes...</div>;
  }

  return (
    <div className="crashes-explorer">
      <div className="explorer-header">
        <h2>Crashes Explorer</h2>
        <div className="controls">
          <div className="filter-controls">
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Crashes</option>
              <option value="critical">Critical Only</option>
              <option value="normal">Normal Only</option>
            </select>
            
            <input
              type="text"
              placeholder="Search crashes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
          
          <button onClick={fetchCrashes} className="refresh-btn">
            Refresh
          </button>
        </div>
      </div>

      <div className="stats-summary">
        <div className="stat-item">
          <span className="stat-label">Total Crashes:</span>
          <span className="stat-value">{crashes.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Critical Crashes:</span>
          <span className="stat-value critical">{criticalCrashes.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Showing:</span>
          <span className="stat-value">{filteredCrashes().length}</span>
        </div>
      </div>

      <div className="crashes-container">
        <div className="crashes-list">
          <h3>Crashes Found</h3>
          {filteredCrashes().length === 0 ? (
            <div className="no-crashes">
              <p>No crashes found matching the current filters.</p>
            </div>
          ) : (
            <div className="crashes-grid">
              {filteredCrashes().map((crash, index) => (
                <div 
                  key={index}
                  className={`crash-card ${getCrashSeverity(crash.crash_type)} ${crash.system_impact ? 'system-impact' : ''}`}
                  onClick={() => setSelectedCrash(crash)}
                >
                  <div className="crash-header">
                    <span className="crash-icon">{getCrashIcon(crash.crash_type)}</span>
                    <span className="crash-type">{crash.crash_type}</span>
                    {crash.system_impact && (
                      <span className="system-impact-badge">SYSTEM IMPACT</span>
                    )}
                  </div>
                  
                  <div className="crash-details">
                    <div className="detail-row">
                      <span className="label">Generation:</span>
                      <span className="value">{crash.generation}</span>
                    </div>
                    <div className="detail-row">
                      <span className="label">Program ID:</span>
                      <span className="value">{crash.program_id}</span>
                    </div>
                    <div className="detail-row">
                      <span className="label">Length:</span>
                      <span className="value">{crash.shellcode_length} bytes</span>
                    </div>
                    <div className="detail-row">
                      <span className="label">Shellcode:</span>
                      <span className="value shellcode-preview">
                        {formatShellcode(crash.shellcode_hex)}
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="label">Time:</span>
                      <span className="value">{formatTimestamp(crash.timestamp)}</span>
                    </div>
                  </div>
                  
                  <div className="crash-actions">
                    <Link 
                      to={`/shellcode/${crash.shellcode_hex?.substring(0, 16)}`}
                      className="action-btn"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Analyze
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {selectedCrash && (
          <div className="crash-detail-panel">
            <div className="panel-header">
              <h3>Crash Details</h3>
              <button 
                onClick={() => setSelectedCrash(null)}
                className="close-btn"
              >
                ×
              </button>
            </div>
            
            <div className="crash-detail-content">
              <div className="detail-section">
                <h4>Basic Information</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <label>Crash Type:</label>
                    <span className={`crash-type ${getCrashSeverity(selectedCrash.crash_type)}`}>
                      {selectedCrash.crash_type}
                    </span>
                  </div>
                  <div className="detail-item">
                    <label>Generation:</label>
                    <span>{selectedCrash.generation}</span>
                  </div>
                  <div className="detail-item">
                    <label>Program ID:</label>
                    <span>{selectedCrash.program_id}</span>
                  </div>
                  <div className="detail-item">
                    <label>Return Code:</label>
                    <span>{selectedCrash.return_code}</span>
                  </div>
                  <div className="detail-item">
                    <label>System Impact:</label>
                    <span className={selectedCrash.system_impact ? 'critical' : 'normal'}>
                      {selectedCrash.system_impact ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <label>Timestamp:</label>
                    <span>{formatTimestamp(selectedCrash.timestamp)}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h4>Shellcode Analysis</h4>
                <div className="shellcode-display">
                  <div className="shellcode-hex">
                    <label>Hex:</label>
                    <pre>{selectedCrash.shellcode_hex}</pre>
                  </div>
                  <div className="shellcode-info">
                    <div className="info-item">
                      <label>Length:</label>
                      <span>{selectedCrash.shellcode_length} bytes</span>
                    </div>
                    <div className="info-item">
                      <label>Entropy:</label>
                      <span>{calculateEntropy(selectedCrash.shellcode_hex).toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {selectedCrash.stderr && (
                <div className="detail-section">
                  <h4>Error Output</h4>
                  <pre className="error-output">{selectedCrash.stderr}</pre>
                </div>
              )}

              {selectedCrash.stdout && (
                <div className="detail-section">
                  <h4>Standard Output</h4>
                  <pre className="stdout-output">{selectedCrash.stdout}</pre>
                </div>
              )}

              <div className="detail-actions">
                <Link 
                  to={`/shellcode/${selectedCrash.shellcode_hex?.substring(0, 16)}`}
                  className="action-btn primary"
                >
                  Full Analysis
                </Link>
                <button className="action-btn secondary">
                  Export JSON
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Función auxiliar para calcular entropía
const calculateEntropy = (hexString) => {
  if (!hexString) return 0;
  
  const bytes = hexString.match(/.{1,2}/g) || [];
  const freq = {};
  let total = bytes.length;
  
  bytes.forEach(byte => {
    freq[byte] = (freq[byte] || 0) + 1;
  });
  
  let entropy = 0;
  Object.values(freq).forEach(count => {
    const p = count / total;
    entropy -= p * Math.log2(p);
  });
  
  return entropy;
};

export default CrashesExplorer; 