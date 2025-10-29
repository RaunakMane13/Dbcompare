import React, { useState, useEffect } from 'react';
import Select from 'react-select';

function FeaturesPage() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
    fetch('https://dbcompare.webdev.gccis.rit.edu/api/files')
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
        `https://dbcompare.webdev.gccis.rit.edu/columns/${encodeURIComponent(
          selectedFile.value
        )}`
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

    const filepath = `/home/student/my-app/src/uploads/${selectedFile.value}`;
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
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-gray-800 text-white p-4 text-center text-xl font-bold">
        Features - Visualize Your Data
      </header>

      <main className="flex-grow p-8">
        <div className="max-w-4xl mx-auto bg-white p-6 shadow-lg rounded">
          <h2 className="text-2xl font-semibold mb-4">Select a File to Visualize</h2>

          {error && <div className="text-red-500 mb-4">{error}</div>}

          <Select
            options={files}
            onChange={(file) => {
              setSelectedFile(file);
              setSelectedColumn(null); // Reset column selection
              setVisualizations(null);
            }}
            placeholder="Choose a file..."
          />

          {selectedFile && columns && (
            <Select
              options={columns}
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

          {result &&
            result.visualizations &&
            Object.entries(result.visualizations)
              .filter(([key]) => key !== 'recommendations')
              .map(([key, value]) => (
                <div key={key} className="mb-4">
                  <h4 className="font-semibold">{key.replace('_', ' ')}</h4>

                  {typeof value === 'string' && value.endsWith('.png') ? (
                    <img
                      src={`https://dbcompare.webdev.gccis.rit.edu/api/${value}`}
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
  );
}

export default FeaturesPage;
