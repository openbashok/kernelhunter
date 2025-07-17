import React, { useState } from 'react';

const ShellcodeAnalyzer = () => {
  const [shellcode, setShellcode] = useState('');
  const [analysis, setAnalysis] = useState(null);

  const analyzeShellcode = () => {
    if (!shellcode.trim()) {
      setAnalysis(null);
      return;
    }

    // Análisis básico del shellcode
    const hexBytes = shellcode.replace(/\s/g, '');
    const bytes = [];
    
    for (let i = 0; i < hexBytes.length; i += 2) {
      bytes.push(parseInt(hexBytes.substr(i, 2), 16));
    }

    const analysis = {
      length: bytes.length,
      hexString: hexBytes,
      bytes: bytes,
      containsSyscall: hexBytes.includes('0f05'),
      containsNullBytes: hexBytes.includes('00'),
      printableChars: bytes.filter(b => b >= 32 && b <= 126).length,
      entropy: calculateEntropy(bytes)
    };

    setAnalysis(analysis);
  };

  const calculateEntropy = (bytes) => {
    const freq = {};
    bytes.forEach(b => freq[b] = (freq[b] || 0) + 1);
    
    let entropy = 0;
    const len = bytes.length;
    
    Object.values(freq).forEach(count => {
      const p = count / len;
      entropy -= p * Math.log2(p);
    });
    
    return entropy.toFixed(2);
  };

  return (
    <div className="shellcode-analyzer">
      <h2>Analizador de Shellcode</h2>
      
      <div className="input-section">
        <textarea
          value={shellcode}
          onChange={(e) => setShellcode(e.target.value)}
          placeholder="Ingresa el shellcode en hexadecimal (ej: 48 c7 c0 3c 00 00 00)"
          rows={5}
          cols={50}
        />
        <button onClick={analyzeShellcode}>Analizar</button>
      </div>

      {analysis && (
        <div className="analysis-results">
          <h3>Resultados del Análisis</h3>
          <div className="analysis-grid">
            <div className="analysis-item">
              <strong>Longitud:</strong> {analysis.length} bytes
            </div>
            <div className="analysis-item">
              <strong>Entropía:</strong> {analysis.entropy}
            </div>
            <div className="analysis-item">
              <strong>Contiene Syscall:</strong> {analysis.containsSyscall ? 'Sí' : 'No'}
            </div>
            <div className="analysis-item">
              <strong>Contiene Null Bytes:</strong> {analysis.containsNullBytes ? 'Sí' : 'No'}
            </div>
            <div className="analysis-item">
              <strong>Caracteres imprimibles:</strong> {analysis.printableChars}/{analysis.length}
            </div>
          </div>
          
          <div className="hex-display">
            <h4>Bytes en Hexadecimal:</h4>
            <code>{analysis.hexString}</code>
          </div>
        </div>
      )}
    </div>
  );
};

export default ShellcodeAnalyzer; 