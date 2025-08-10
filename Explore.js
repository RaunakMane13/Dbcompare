import React, { useState, useEffect } from 'react';

export default function ExplorePage() {
  const [files, setFiles] = useState([]);
  const [preview, setPreview] = useState({ columns: [], rows: [] });
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/files')
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(data => setFiles(data.files || []))
      .catch(e => setError(`Could not load file list: ${e}`));
  }, []);

  const fetchFilePreview = async (filename) => {
    setError(null);
    setPreview({ columns: [], rows: [] });
    try {
      const url = `/file-preview/${encodeURIComponent(filename)}`;
      const res = await fetch(url);
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }
      const { preview: rows } = await res.json();
      console.log('▶️ preview rows for', filename, rows);
      if (!Array.isArray(rows) || rows.length === 0) {
        setError('No preview data available');
        return;
      }
      setPreview({
        columns: Object.keys(rows[0]),
        rows
      });
    } catch (err) {
      setError(`Preview failed: ${err.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-blue-800 text-white p-4">
        <h1 className="text-xl font-bold">DBCompare — File Explorer</h1>
      </header>

      <main className="flex-grow p-6">
        {error && (
          <div className="mb-4 p-2 bg-red-200 text-red-800 rounded">
            {error}
          </div>
        )}

        <section className="mb-6">
          <h2 className="font-semibold mb-2">Uploaded files:</h2>
          <div className="flex flex-wrap gap-2">
            {files.map(f => (
              <button
                key={f}
                onClick={() => fetchFilePreview(f)}
                className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                {f}
              </button>
            ))}
            {!files.length && <span>No files found.</span>}
          </div>
        </section>

        {preview.rows.length > 0 && (
          <section>
            <h2 className="font-semibold mb-2">Preview:</h2>
            <div className="overflow-x-auto border rounded">
              <table className="min-w-full bg-white">
                <thead className="bg-gray-200">
                  <tr>
                    {preview.columns.map(col => (
                      <th key={col} className="px-4 py-2 text-left">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.map((row, i) => (
                    <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : ''}>
                      {preview.columns.map(col => (
                        <td key={col} className="px-4 py-2">
                          {row[col]?.toString() ?? ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      <footer className="bg-gray-200 p-4 text-center text-sm">
        &copy; {new Date().getFullYear()} DBCompare
      </footer>
    </div>
  );
}
