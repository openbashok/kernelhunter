import React, { useState, useEffect } from 'react';
import { Line, Bar, Pie, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Actualizar cada 10 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [metricsResponse, statsResponse] = await Promise.all([
        fetch('/api/metrics'),
        fetch('/api/stats')
      ]);

      if (metricsResponse.ok && statsResponse.ok) {
        const metricsData = await metricsResponse.json();
        const statsData = await statsResponse.json();
        
        setMetrics(metricsData.metrics);
        setStats(statsData);
        setError(null);
      } else {
        setError('Failed to fetch data');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!metrics || !stats) {
    return <div className="no-data">No data available</div>;
  }

  // Preparar datos para gráficos
  const crashRateData = {
    labels: metrics.generations?.map(gen => `Gen ${gen}`) || [],
    datasets: [
      {
        label: 'Crash Rate (%)',
        data: metrics.crash_rates?.map(rate => rate * 100) || [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        tension: 0.1,
      },
    ],
  };

  const systemImpactsData = {
    labels: metrics.generations?.map(gen => `Gen ${gen}`) || [],
    datasets: [
      {
        label: 'System Impacts',
        data: metrics.system_impacts || [],
        backgroundColor: 'rgba(255, 159, 64, 0.8)',
      },
    ],
  };

  const shellcodeLengthData = {
    labels: metrics.generations?.map(gen => `Gen ${gen}`) || [],
    datasets: [
      {
        label: 'Avg Shellcode Length',
        data: metrics.shellcode_lengths || [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        tension: 0.1,
      },
    ],
  };

  const crashTypeData = {
    labels: Object.keys(stats.crash_type_distribution || {}),
    datasets: [
      {
        data: Object.values(stats.crash_type_distribution || {}),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
        ],
      },
    ],
  };

  const attackDistributionData = {
    labels: Object.keys(stats.attack_distribution || {}).slice(0, 10),
    datasets: [
      {
        label: 'Attack Usage',
        data: Object.values(stats.attack_distribution || {}).slice(0, 10),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
      },
    ],
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>KernelHunter Evolution Dashboard</h2>
        <div className="stats-overview">
          <div className="stat-card">
            <h3>Total Generations</h3>
            <p>{stats.total_generations}</p>
          </div>
          <div className="stat-card">
            <h3>Total Crashes</h3>
            <p>{stats.total_crashes}</p>
          </div>
          <div className="stat-card">
            <h3>Critical Crashes</h3>
            <p>{stats.critical_crashes}</p>
          </div>
          <div className="stat-card">
            <h3>Avg Crash Rate</h3>
            <p>{(stats.avg_crash_rate * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3>Crash Rate Evolution</h3>
          <Line 
            data={crashRateData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Crash Rate Over Generations' }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  ticks: {
                    callback: function(value) {
                      return value + '%';
                    }
                  }
                }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>System Impacts</h3>
          <Bar 
            data={systemImpactsData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'System-Level Impacts' }
              },
              scales: {
                y: {
                  beginAtZero: true
                }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Shellcode Length Evolution</h3>
          <Line 
            data={shellcodeLengthData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Average Shellcode Length' }
              },
              scales: {
                y: {
                  beginAtZero: true
                }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Crash Type Distribution</h3>
          <Doughnut 
            data={crashTypeData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'right' },
                title: { display: true, text: 'Types of Crashes Found' }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Top Attack Types</h3>
          <Bar 
            data={attackDistributionData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Most Used Attack Types' }
              },
              scales: {
                y: {
                  beginAtZero: true
                }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Latest Generation Details</h3>
          <div className="generation-details">
            {metrics.generations && metrics.generations.length > 0 && (
              <div>
                <p><strong>Current Generation:</strong> {Math.max(...metrics.generations)}</p>
                <p><strong>Latest Crash Rate:</strong> 
                  {metrics.crash_rates && metrics.crash_rates.length > 0 
                    ? (metrics.crash_rates[metrics.crash_rates.length - 1] * 100).toFixed(1) + '%'
                    : 'N/A'
                  }
                </p>
                <p><strong>Latest System Impacts:</strong> 
                  {metrics.system_impacts && metrics.system_impacts.length > 0 
                    ? metrics.system_impacts[metrics.system_impacts.length - 1]
                    : 'N/A'
                  }
                </p>
                <p><strong>Most Common Crash:</strong> {stats.most_common_crash || 'N/A'}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="real-time-updates">
        <h3>Real-Time Updates</h3>
        <p>Dashboard updates every 10 seconds</p>
        <button onClick={fetchData} className="refresh-btn">
          Refresh Now
        </button>
      </div>
    </div>
  );
};

export default Dashboard; 