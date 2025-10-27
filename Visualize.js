import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import Nav from './components/Nav';
export default function VisualizePage() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const sortedFiles = Array.isArray(files)
  ? [...files].sort((a, b) => a.label.localeCompare(b.label))
  : [];

const sortedColumns = Array.isArray(columns)
  ? [...columns].sort((a, b) => a.label.localeCompare(b.label))
  : [];
  const [taskId, setTaskId] = useState(null);
  const [result, setResult] = useState(null);
  
  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(() => {
      fetch(`/api/task-status/${taskId}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.status === 'done') {
            console.log('Task result:', data.result);
            setResult(data.result); // <-- You must set this, because your JSX uses `result.visualizations`
            setVisualizations(data.result.visualizations);
            setLoading(false);
            clearInterval(interval);
          } else if (data.status === 'failed') {
            setError('Task failed: ' + data.error);
            clearInterval(interval);
          }
        })
        .catch((err) => {
          setError('Error polling task: ' + err.message);
          clearInterval(interval);
        });
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId]);

  // Fetch list of files
  useEffect(() => {
    fetch(`${process.env.REACT_APP_API_BASE_URL}/files`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) =>
        setFiles(data.files.map((file) => ({ value: file.trim(), label: file.trim() })))
      )
      .catch((err) => setError(`Error fetching files: ${err.message}`));
  }, []);

  // Fetch columns for the selected file
  useEffect(() => {
    if (selectedFile) {
      fetch(
        `${process.env.REACT_APP_API_BASE_URL}/columns/${encodeURIComponent(selectedFile.value)}`
      )
        .then((res) => res.json())
        .then((data) => {
          setColumns(data.columns.map((column) => ({ value: column, label: column })));
        })
        .catch((err) => setError(`Error fetching columns: ${err.message}`));
    }
  }, [selectedFile]);

  const handleVisualization = () => {
    if (!selectedFile) {
      alert('Please select a file!');
      return;
    }

    setLoading(true);
    setError('');
    setVisualizations(null);

    const filepath = `${process.env.REACT_APP_UPLOAD_PATH}/${selectedFile.value}`;
    const column = selectedColumn ? selectedColumn.value : null;

    fetch('/api/start-task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filepath: filepath,
        task_type: 'visualize',
        column: column,
      }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setTaskId(data.task_id); // triggers useEffect to poll for results
      })
      .catch((err) => {
        setError(`Failed to start task: ${err.message}`);
        setLoading(false);
      });
  };

  return (
    <>
    <Nav/>
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-gray-800 text-white p-4 text-center text-xl font-bold">
        Features - Visualize Your Data
      </header>

      <main className="flex-grow p-8">
        <div className="max-w-4xl mx-auto bg-white p-6 shadow-lg rounded">
          <h2 className="text-2xl font-semibold mb-4">Select a File to Visualize</h2>

          {error && <div className="text-red-500 mb-4">{error}</div>}

          <Select
            options={sortedFiles}
            onChange={(file) => {
              setSelectedFile(file);
              setSelectedColumn(null); // Reset column selection
              setVisualizations(null);
            }}
            placeholder="Choose a file..."
          />

          {selectedFile && columns && (
            <Select
              options={sortedColumns}
              onChange={setSelectedColumn}
              placeholder="Select a column for the histogram"
            />
          )}

          {selectedFile && (
            <button
              className={`mt-4 p-2 rounded ${
                loading ? 'bg-gray-500' : 'bg-green-500 hover:bg-green-600'
              } text-white`}
              onClick={handleVisualization}
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Generate Visualizations'}
            </button>
          )}
          {/* How it works / inline help */}
          <div className="mb-4 text-sm text-gray-700 bg-gray-50 border rounded p-3">
            <details>
              <summary className="cursor-pointer font-medium">
                What do these charts mean?
              </summary>
              <div className="mt-2 space-y-2">
                <p><strong>Correlation heatmap</strong>: shows linear relationships among numeric columns (−1 to +1). Values near +1 = strong positive, near −1 = strong negative.</p>
                <p><strong>Feature importance</strong>: trains a Random Forest. If no <code>target</code> column is present, the backend auto-creates a binary target from the first valid numeric column ( its mean). Bars show relative importance (sum ≈ 1).</p>
                <p><strong>Histogram</strong>: numeric/bool → 30-bin histogram; non-numeric → JSON-stringified counts, top 10 categories.</p>
              </div>
            </details>

            {/* Download guide */}
            <div className="mt-3">
              <a
                href={`${process.env.REACT_APP_API_BASE_URL}/visualization-guide`}
                className="inline-block px-3 py-1 rounded bg-gray-800 text-white hover:bg-gray-700"
                download
              >
                Download full visualization guide
              </a>
            </div>
          </div>

          {result &&
            result.visualizations &&
            Object.entries(result.visualizations)
              .filter(([key]) => key !== 'recommendations')
              .map(([key, value]) => (
                <div key={key} className="mb-4">
                  <h4 className="font-semibold flex items-center gap-2">
                    {key.replace('_', ' ')}
                    {key === 'correlation_heatmap' && (
                      <span
                        title="Numeric columns only. Pearson correlation (−1 to +1)."
                        className="text-xs px-2 py-0.5 border rounded cursor-help"
                      >?</span>
                    )}
                    {key === 'feature_importance' && (
                      <span
                        title="Random Forest feature importances; larger = more influence on predictions. Target auto-created if missing."
                        className="text-xs px-2 py-0.5 border rounded cursor-help"
                      >?</span>
                    )}
                    {key === 'histogram' && (
                      <span
                        title="Numeric/bool: 30 bins. Non-numeric: top 10 categories."
                        className="text-xs px-2 py-0.5 border rounded cursor-help"
                      >?</span>
                    )}
                  </h4>


                  {typeof value === 'string' && value.endsWith('.png') ? (
                    <img
                      src={`${process.env.REACT_APP_API_BASE_URL}/${value}`}
                      alt={key}
                      className="w-full rounded"
                    />
                  ) : Array.isArray(value) ? (
                    <table className="table-auto border-collapse border border-gray-400 w-full">
                      <thead>
                        <tr>
                          {Object.keys(value[0] || {}).map((col) => (
                            <th key={col} className="border border-gray-300 px-2 py-1">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {value.map((row, idx) => (
                          <tr key={idx}>
                            {Object.values(row).map((cell, i) => (
                              <td key={i} className="border border-gray-300 px-2 py-1">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              ))}
        </div>
      </main>
    </div>
    </>
  );
}
