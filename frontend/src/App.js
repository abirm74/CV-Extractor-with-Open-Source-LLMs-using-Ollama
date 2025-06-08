import React, { useState } from 'react';
import { Upload, FileText, Download, AlertCircle, CheckCircle } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null); // For PDF preview
  const [jsonOutput, setJsonOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onDrop = (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      // Create URL for PDF preview
      const url = URL.createObjectURL(selectedFile);
      setPdfUrl(url);
      setError('');
      // Clear previous results
      setJsonOutput(null);
    } else {
      setError('Please select a PDF file');
    }
  };

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPdfUrl(url);
      setError('');
      setJsonOutput(null);
    } else {
      setError('Please select a PDF file');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();

      // Filter out any text file references from the response
      const filteredData = filterTextFiles(data);
      setJsonOutput(filteredData);
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  // Function to filter out text file references from JSON response
  const filterTextFiles = (data) => {
    if (!data) return data;

    // Create a deep copy to avoid mutating original data
    const filtered = JSON.parse(JSON.stringify(data));

    // Remove any properties that might contain text file references
    const removeTextFileRefs = (obj) => {
      if (Array.isArray(obj)) {
        return obj.map(item => removeTextFileRefs(item));
      } else if (obj && typeof obj === 'object') {
        const cleaned = {};
        for (const [key, value] of Object.entries(obj)) {
          // Skip properties that might reference text files
          if (key.toLowerCase().includes('txt') ||
              key.toLowerCase().includes('text_file') ||
              (typeof value === 'string' && value.toLowerCase().includes('.txt'))) {
            continue;
          }
          cleaned[key] = removeTextFileRefs(value);
        }
        return cleaned;
      }
      return obj;
    };

    return removeTextFileRefs(filtered);
  };

  const downloadJson = () => {
    if (!jsonOutput) return;

    const blob = new Blob([JSON.stringify(jsonOutput, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name?.replace('.pdf', '')}_extracted.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Clean up URL when component unmounts or file changes
  React.useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  return (
    <div style={{
      minHeight: '100vh',
      padding: '20px',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.95)',
          borderRadius: '20px',
          padding: '30px',
          marginBottom: '30px',
          textAlign: 'center',
          boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
        }}>
          <FileText size={48} style={{ color: '#667eea', marginBottom: '15px' }} />
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: 'bold',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '10px'
          }}>
            Resume Extractor
          </h1>
          <p style={{ color: '#666', fontSize: '1.1rem' }}>
            Upload your PDF resume and get structured JSON data instantly
          </p>
        </div>

        {/* Upload Section */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.95)',
          borderRadius: '20px',
          padding: '30px',
          marginBottom: '30px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
        }}>
          <div
            onClick={() => document.getElementById('file-input').click()}
            style={{
              border: `3px dashed #ccc`,
              borderRadius: '15px',
              padding: '50px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.style.borderColor = '#667eea';
              e.currentTarget.style.background = 'rgba(102, 126, 234, 0.1)';
            }}
            onDragLeave={(e) => {
              e.currentTarget.style.borderColor = '#ccc';
              e.currentTarget.style.background = 'transparent';
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.currentTarget.style.borderColor = '#ccc';
              e.currentTarget.style.background = 'transparent';
              const files = Array.from(e.dataTransfer.files);
              if (files.length > 0) {
                const file = files[0];
                if (file.type === 'application/pdf') {
                  setFile(file);
                  const url = URL.createObjectURL(file);
                  setPdfUrl(url);
                  setError('');
                  setJsonOutput(null);
                } else {
                  setError('Please select a PDF file');
                }
              }
            }}
          >
            <input
              id="file-input"
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <Upload size={48} style={{
              color: '#999',
              marginBottom: '15px'
            }} />
            <p style={{
              fontSize: '1.2rem',
              color: '#666',
              marginBottom: '10px'
            }}>
              Drag & drop your PDF resume here
            </p>
            <p style={{ color: '#999', fontSize: '0.9rem' }}>
              or click to browse files
            </p>
          </div>

          {file && (
            <div style={{
              marginTop: '20px',
              padding: '15px',
              background: 'rgba(102, 126, 234, 0.1)',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <FileText size={20} style={{ color: '#667eea', marginRight: '10px' }} />
                <span style={{ color: '#667eea', fontWeight: 'bold' }}>{file.name}</span>
              </div>
              <CheckCircle size={20} style={{ color: '#22c55e' }} />
            </div>
          )}

          {error && (
            <div style={{
              marginTop: '15px',
              padding: '12px',
              background: 'rgba(239, 68, 68, 0.1)',
              color: '#dc2626',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center'
            }}>
              <AlertCircle size={18} style={{ marginRight: '8px' }} />
              {error}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            style={{
              width: '100%',
              marginTop: '20px',
              padding: '15px',
              background: !file || loading ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '12px',
              fontSize: '1.1rem',
              fontWeight: 'bold',
              cursor: !file || loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            {loading ? 'Processing...' : 'Extract Resume Data'}
          </button>
        </div>

        {/* Results Section - Show when file is selected OR when results are ready */}
        {(file || jsonOutput) && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '30px',
            marginBottom: '30px'
          }}>

            {/* PDF Preview */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '20px',
              overflow: 'hidden',
              boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
            }}>
              <div style={{
                padding: '20px',
                background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
                color: 'white'
              }}>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 'bold' }}>Resume Preview</h3>
              </div>
              <div style={{
                padding: '25px',
                height: '500px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {pdfUrl ? (
                  <iframe
                    src={pdfUrl}
                    style={{
                      width: '100%',
                      height: '100%',
                      border: 'none',
                      borderRadius: '8px'
                    }}
                    title="PDF Preview"
                  />
                ) : (
                  <div style={{
                    textAlign: 'center',
                    color: '#666'
                  }}>
                    <FileText size={48} style={{ color: '#ccc', marginBottom: '15px' }} />
                    <p>PDF preview will appear here after uploading</p>
                  </div>
                )}
              </div>
            </div>

            {/* JSON Output */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '20px',
              overflow: 'hidden',
              boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
            }}>
              <div style={{
                padding: '20px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 'bold' }}>Extracted JSON</h3>
                {jsonOutput && (
                  <button
                    onClick={downloadJson}
                    style={{
                      background: 'rgba(255, 255, 255, 0.2)',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '8px 12px',
                      color: 'white',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px'
                    }}
                  >
                    <Download size={16} />
                    Download
                  </button>
                )}
              </div>
              <div style={{
                padding: '25px',
                height: '500px',
                overflowY: 'auto',
                display: 'flex',
                alignItems: jsonOutput ? 'flex-start' : 'center',
                justifyContent: jsonOutput ? 'flex-start' : 'center'
              }}>
                {jsonOutput ? (
                  <pre style={{
                    background: '#f8f9fa',
                    padding: '15px',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    lineHeight: '1.4',
                    color: '#333',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    width: '100%'
                  }}>
                    {JSON.stringify(jsonOutput, null, 2)}
                  </pre>
                ) : (
                  <div style={{
                    textAlign: 'center',
                    color: '#666'
                  }}>
                    <FileText size={48} style={{ color: '#ccc', marginBottom: '15px' }} />
                    <p>JSON output will appear here after processing</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;