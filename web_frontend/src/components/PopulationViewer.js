import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PopulationViewer = () => {
  const [population, setPopulation] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPopulation();
    const interval = setInterval(fetchPopulation, 5000); // Actualizar cada 5 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchPopulation = async () => {
    try {
      const response = await axios.get('/api/population');
      setPopulation(response.data.population || []);
      setLoading(false);
    } catch (err) {
      setError('Error cargando población');
      setLoading(false);
    }
  };

  if (loading) return <div>Cargando población...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="population-viewer">
      <h2>Población Actual ({population.length} individuos)</h2>
      <div className="population-grid">
        {population.map((individual, index) => (
          <div key={index} className="individual-card">
            <h3>Individuo {individual.id}</h3>
            <p><strong>Longitud:</strong> {individual.length} bytes</p>
            <p><strong>Shellcode:</strong></p>
            <code className="shellcode-display">
              {individual.shellcode}
            </code>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PopulationViewer; 