import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/population', label: 'Población', icon: '🧬' },
    { path: '/crashes', label: 'Crashes', icon: '💥' },
    { path: '/analyzer', label: 'Analizador', icon: '🔍' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-header">
        <h1>KernelHunter Advanced</h1>
      </div>
      
      <ul className="nav-menu">
        {navItems.map((item) => (
          <li key={item.path} className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}>
            <Link to={item.path} className="nav-link">
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </Link>
          </li>
        ))}
      </ul>
      
      <div className="nav-footer">
        <div className="status-indicator">
          <span className="status-dot online"></span>
          <span>Conectado</span>
        </div>
      </div>
    </nav>
  );
};

export default Navigation; 