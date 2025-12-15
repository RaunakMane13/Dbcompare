// src/Update.js
import React, { useState } from 'react';
import './ProjectOverview.css';
import Nav from './components/Nav';

export default function DatabaseEditor() {
  const [selectedDatabaseType, setSelectedDatabaseType] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState(null);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(0);          // ← current page index (0-based)
  const [dirtyRows, setDirtyRows] = useState(new Set());
  const [selectedRowIndex, setSelectedRowIndex] = useState(null); // which row is open in the details panel
  const [saveError, setSaveError] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const [columnTypes, setColumnTypes] = useState({});

  // Fetch exactly 10 records for given page & table
  const fetchRecords = async (pageNum = 0, tableName = selectedTable) => {
    setLoading(true);
    const limit = 10;
    const offset = pageNum * limit;
    try {
      const url = `/api/${selectedDatabaseType}/records`
        + `?database=${selectedDatabase}`
        + `&table=${tableName}`
        + `&limit=${limit}`
        + `&offset=${offset}`;
      const res = await fetch(url);
      const data = await res.json();
      setRecords(data.records || []);
      setPage(pageNum);
    } catch (err) {
      console.error('Error fetching page', pageNum, err);
    } finally {
      setLoading(false);
    }
  };

  const handleDatabaseTypeSelect = async (dbType) => {
    setSelectedDatabaseType(dbType);
    setSelectedDatabase(null);
    setTables([]);
    setSelectedTable(null);
    setRecords([]);
    setPage(0);
    setLoading(true);
    try {
      const res = await fetch(`/api/${dbType}/databases`);
      const data = await res.json();
      setDatabases(data.databases || []);
    } catch (err) {
      console.error('Error fetching databases:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDatabaseSelect = async (db) => {
    setSelectedDatabase(db);
    setTables([]);
    setSelectedTable(null);
    setRecords([]);
    setPage(0);
    setLoading(true);
    try {
      const res = await fetch(`/api/${selectedDatabaseType}/tables?database=${db}`);
      const data = await res.json();
      setTables(data.tables || []);
    } catch (err) {
      console.error('Error fetching tables:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTableSelect = async (table) => {
    setSelectedTable(table);
    setRecords([]);
    setPage(0);
  
    // Fetch expected datatypes (MySQL only)
    if (selectedDatabaseType === "mysql") {
      try {
        const res = await fetch(`/api/mysql/column-types?database=${selectedDatabase}&table=${table}`);
        const data = await res.json();
        setColumnTypes(data.types || {});
      } catch (e) {
        console.error("Failed to fetch column types:", e);
        setColumnTypes({});
      }
    } else {
      // MongoDB: no fixed types unless you enforce schema
      setColumnTypes({});
    }
  
    // Load first page of 10
    await fetchRecords(0, table);
  };
  

  const handleRecordChange = (idx, field, val) => {
    const updated = [...records];
    updated[idx] = { ...updated[idx], [field]: val };
    setRecords(updated);
    setDirtyRows(prev => {
      const nxt = new Set(prev);
      nxt.add(idx);
      return nxt;
    });
  };

  const handleSaveChanges = async () => {
    setSaveError(null);
    setFieldErrors({});
  
    try {
      const res = await fetch(
        `/api/${selectedDatabaseType}/update?database=${selectedDatabase}&table=${selectedTable}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            records: records.filter((_, i) => dirtyRows.has(i))
          })
        }
      );
  
      const data = await res.json().catch(() => ({}));
  
      if (!res.ok) {
        // MySQL: show which column failed
        if (data.fieldError?.field) {
          setSaveError(data.message || "Update failed.");
          setFieldErrors({
            [data.fieldError.field]: data.fieldError.reason || data.details || "Invalid value"
          });
        } else {
          setSaveError(data.message || data.details || data.error || "Failed to save changes.");
        }
        return;
      }
  
      alert('Records updated successfully!');
      setDirtyRows(new Set());
    } catch (err) {
      console.error('Save failed:', err);
      setSaveError(err.message || 'Failed to save changes.');
    }
  };
  

  return (
    <>
    <Nav />
    <div className="min-h-screen bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white">
      

      <main className="container mx-auto p-6">
        <h1 className="text-center text-3xl font-bold mb-8">Database Editor</h1>

        {!selectedDatabaseType && (
          <div className="flex justify-center space-x-4">
            <button
              className="px-6 py-2 bg-blue-600 rounded-lg hover:bg-blue-700"
              onClick={() => handleDatabaseTypeSelect('mysql')}
            >
              MySQL
            </button>
            <button
              className="px-6 py-2 bg-green-600 rounded-lg hover:bg-green-700"
              onClick={() => handleDatabaseTypeSelect('mongodb')}
            >
              MongoDB
            </button>
          </div>
        )}

        {loading && <p className="text-center text-xl mt-6">Loading...</p>}

        {selectedDatabaseType && !selectedDatabase && databases.length > 0 && (
          <section className="mt-8">
            <h2 className="text-xl font-semibold mb-4">
              {selectedDatabaseType} Databases
            </h2>
            <div className="grid grid-cols-4 gap-4">
              {databases.map(db => (
                <button
                  key={db}
                  className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-800"
                  onClick={() => handleDatabaseSelect(db)}
                >
                  {db}
                </button>
              ))}
            </div>
          </section>
        )}

        {selectedDatabase && !selectedTable && tables.length > 0 && (
          <section className="mt-8">
            <h2 className="text-xl font-semibold mb-4">
              Tables in {selectedDatabase}
            </h2>
            <div className="grid grid-cols-4 gap-4">
              {tables.map(tbl => (
                <button
                  key={tbl}
                  className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-800"
                  onClick={() => handleTableSelect(tbl)}
                >
                  {tbl}
                </button>
              ))}
            </div>
          </section>
        )}

        {selectedTable && (
          <section className="mt-8">
            <h2 className="text-xl font-semibold mb-4">
              Editing Records in {selectedTable}
            </h2>

            {/* ← scrollable container for large tables */}
            {/* ← scrollable container for large tables (read-only view) */}
<div className="overflow-x-auto border border-gray-700 rounded">
  <table className="table-auto min-w-[1200px] w-full border-collapse">
    <thead>
      <tr className="bg-gray-800">
        {records[0] && Object.keys(records[0]).map(field => (
          <th
            key={field}
            className="px-4 py-2 text-left text-sm whitespace-nowrap"
          >
            {field}
          </th>
        ))}
      </tr>
    </thead>

    <tbody>
      {records.map((rec, i) => {
        const isSelected = i === selectedRowIndex;
        return (
          <tr
            key={i}
            onClick={() => setSelectedRowIndex(i)}
            className={
              (isSelected ? 'bg-blue-900' : (i % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'))
              + ' cursor-pointer'
            }
          >
            {Object.entries(rec).map(([field, val]) => (
              <td key={field} className="px-4 py-2 text-sm whitespace-nowrap">
                {val !== null && val !== undefined ? String(val) : ''}
              </td>
            ))}
          </tr>
        );
      })}
    </tbody>
  </table>
</div>

{/* Detailed editor for the selected row */}
{selectedRowIndex !== null && records[selectedRowIndex] && (
  <div className="mt-6 p-4 border border-gray-700 rounded-lg bg-gray-900">
    <h3 className="text-lg font-semibold mb-4">
      Row details (row #{page * 10 + selectedRowIndex + 1})
    </h3>

    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Object.entries(records[selectedRowIndex]).map(([field, val]) => (
        <div key={field} className="flex flex-col gap-1">
          <label
            className="text-xs font-semibold text-gray-400 uppercase"
            title={columnTypes[field] ? `Expected: ${columnTypes[field]}` : ""}
          >
            {field}
            {columnTypes[field] && (
              <span className="ml-2 text-[10px] text-gray-500 normal-case">(i)</span>
            )}
          </label>
          <input
            className={`w-full bg-gray-700 text-white px-2 py-1 rounded ${
              fieldErrors[field] ? 'border border-red-500' : ''
            }`}
            value={val ?? ''}
            onChange={(e) => handleRecordChange(selectedRowIndex, field, e.target.value)}
          />

          {fieldErrors[field] && (
            <div className="text-xs text-red-400">
              {fieldErrors[field]}
            </div>
          )}
        </div>
      ))}
    </div>
  </div>
)}


            {/* ← pagination controls */}
            <div className="mt-4 flex justify-end space-x-2">
              <button
                disabled={page === 0}
                onClick={() => fetchRecords(page - 1)}
                className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
              >
                Prev
              </button>
              <button
                disabled={records.length < 10}
                onClick={() => fetchRecords(page + 1)}
                className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
              >
                Next
              </button>
            </div>
            {saveError && (
              <div className="mt-4 mb-2 p-3 rounded bg-red-900/40 border border-red-700 text-red-200">
                {saveError}
              </div>
            )}
            <div className="mt-6 text-center">
              <button
                className="px-6 py-2 bg-green-600 rounded-lg hover:bg-green-700"
                onClick={handleSaveChanges}
              >
                Save Changes
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
    </>
  );
}
