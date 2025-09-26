import React, { useState, useEffect } from 'react';
import './ProjectOverview.css'; // Ensure the CSS file is linked
import Nav from './components/Nav';

export default function ShardingPage() {
  const [selectedDatabaseType, setSelectedDatabaseType] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState(null);
  const [records, setRecords] = useState([]);
  const [shardedData, setShardedData] = useState({});
  const [loading, setLoading] = useState(false);
  const [shardingMethod, setShardingMethod] = useState('hash');
  const [isSharded, setIsSharded] = useState(false);
  const [columns, setColumns] = useState([]);
  const sortedColumns = Array.isArray(columns)
  ? [...columns].sort((a, b) => a.localeCompare(b))
  : [];
  const [selectedColumn, setSelectedColumn] = useState('');
  const [numPartitions, setNumPartitions] = useState(4);
  const [selectedPartition, setSelectedPartition] = useState('');
  const [shardingStatus, setShardingStatus] = useState(null);
  const [userQuery, setUserQuery] = useState('');
  const [customQueryResults, setCustomQueryResults] = useState({});
  const [availablePartitions, setAvailablePartitions] = useState([]);
  const [customShardingRules, setCustomShardingRules] = useState('');
  const [queryPerformance, setQueryPerformance] = useState(null);
  const [filter, setFilter] = useState('{"key": "value"}');

  useEffect(() => {
  }, [isSharded, shardedData]); 

  const handleDatabaseTypeSelect = async (dbType) => {
    setSelectedDatabaseType(dbType);
    setSelectedDatabase(null);
    setTables([]);
    setRecords([]);
    setShardedData({});
    setLoading(true);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL}/${dbType}/databases`);
      const data = await response.json();
      setDatabases(data.databases);
    } catch (error) {
      console.error('Error fetching databases:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDatabaseSelect = async (database) => {
    setSelectedDatabase(database);
    setTables([]);
    setLoading(true);

    try {
      const endpoint =
  selectedDatabaseType === 'mongodb'
    ? `/api/mongodb/tables?database=${database}`
    : `/api/mysql/tables?database=${database}`;

const response = await fetch(endpoint);
const data = await response.json();
setTables(Array.isArray(data.tables) ? data.tables.slice().sort((a,b)=>a.localeCompare(b)) : []);

    } catch (error) {
      console.error('Error fetching tables/collections:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTableSelect = async (table) => {
    setSelectedTable(table);
    setLoading(true);

    try {
      const endpoint =
        selectedDatabaseType === 'mongodb'
          ? `/api/mongodb/records?database=${selectedDatabase}&table=${table}`
          : `/api/mysql/records?database=${selectedDatabase}&table=${table}`;

      const response = await fetch(endpoint);
      const data = await response.json();
      setRecords(data.records.slice(0, 5));

      if (selectedDatabaseType === 'mongodb') {
        const columnsResponse = await fetch(
          `/api/mongodb/columns?database=${selectedDatabase}&collection=${table}`
        );
        const columnsData = await columnsResponse.json();
        setColumns(columnsData.columns);
        // Set default selected column if available
        if (columnsData.columns.length > 0) {
          setSelectedColumn(columnsData.columns[0]);
        }
      } else {
        const columnsResponse = await fetch(
          `/api/mysql/columns?database=${selectedDatabase}&table=${table}`
        );
        const columnsData = await columnsResponse.json();
        setColumns(columnsData.columns);
        // Set default selected column if available
        if (columnsData.columns.length > 0) {
          setSelectedColumn(columnsData.columns[0]);
        }
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleShard = async () => {
    setLoading(true);
    try {
      const endpoint =
        selectedDatabaseType === 'mongodb' ? '/api/mongodb/shard' : '/api/mysql/shard';

      // Client-side validation for selectedColumn
      if (!selectedColumn) {
        alert('Please select a sharding column.');
        setLoading(false);
        return;
      }

      const body =
        selectedDatabaseType === 'mongodb'
          ? {
              database: selectedDatabase,
              collection: selectedTable,
              shardKey: selectedColumn,
              method: shardingMethod,
            }
          : {
              database: selectedDatabase,
              table: selectedTable,
              column: selectedColumn,
              method: shardingMethod,
              numPartitions: parseInt(numPartitions, 10),
              customRules: shardingMethod === 'custom' ? JSON.parse(customShardingRules) : {},
            };

      console.log('Shard Request Body:', body); // Debug: Request body

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText}`);
      }

      const responseData = await response.json();

      // Always expect a task_id for asynchronous processing
      if (responseData.task_id) {
        const { task_id } = responseData;

        const poll = async () => {
          const res = await fetch(`${process.env.REACT_APP_API_BASE_URL}/task-result/${task_id}`);
          const data = await res.json();

          if (data.status === 'done') {
            if (data.result.error) {
              alert('❌ Sharding failed: ' + data.result.error);
              setIsSharded(false);
            } else {
              setIsSharded(true);
              const shardData = data.result.partitions || data.result.shardData || {};
              setShardedData(shardData);
              setShardingStatus({
                totalChunks: Object.keys(shardData).length,
                shardData,
              });
              setAvailablePartitions(Object.keys(shardData));
              setSelectedPartition(Object.keys(shardData)[0] || '');
              alert(data.result.message || 'Sharding completed');
            }
            setLoading(false);
          } else if (data.status === 'failed') {
            alert('❌ Sharding task failed: ' + data.error);
            setIsSharded(false);
            setLoading(false);
          } else {
            setTimeout(poll, 2000); // keep polling
          }
        };
        poll();
      } else {
        // Fallback for unexpected synchronous response (should not happen with Celery)
        alert(
          'Unexpected synchronous response from sharding API. Please check server logs.'
        );
        setLoading(false);
      }
    } catch (error) {
      alert('Sharding error: ' + error.message);
      setLoading(false);
    }
  };

  const fetchShardingStatus = async () => {
    if (!selectedDatabase || !selectedTable) return;

    try {
      const response = await fetch(
        `/api/mongodb/shard-status?database=${selectedDatabase}&collection=${selectedTable}`
      );
      const data = await response.json();

      if (!data || !data.shardData || Object.keys(data.shardData).length === 0) {
        console.warn('❌ Sharding data is missing:', data);
        setShardingStatus({ totalChunks: 0, shardData: {} });
        return;
      }

      setShardingStatus(data);
    } catch (error) {
      console.error('Error fetching sharding status:', error);
      setShardingStatus({ totalChunks: 0, shardData: {} });
    }
  };

  const handleQueryPerformanceTest = async () => {
    setLoading(true);

    // ------- sanity checks (MySQL only) -------
    if (selectedDatabaseType === 'mysql') {
      if (!selectedPartition) {
        alert('Select a partition (e.g. p0) before running the test.');
        setLoading(false);
        return;
      }
      if (!userQuery.trim()) {
        alert('Enter an SQL query first.');
        setLoading(false);
        return;
      }
    }
    try {
      const endpoint =
        selectedDatabaseType === 'mongodb'
          ? '/api/mongodb/query-performance'
          : '/api/mysql/query-performance';

      const body =
        selectedDatabaseType === 'mongodb'
          ? { database: selectedDatabase, collection: selectedTable, filter }
          : { database: selectedDatabase, table: selectedTable, userQuery, selectedPartition };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText}`);
      }

      const responseData = await response.json();

      // Check if the response contains a task_id for polling
      if (responseData.task_id) {
        const { task_id } = responseData;
        console.log('Received task_id for query performance:', task_id);

        const pollPerf = async () => {
          const res = await fetch(`/api/task-result/${task_id}`);
          const data = await res.json();
          console.log('Polling Query Performance Task Result:', data);

          if (data.status === 'done') {
            if (data.result.error) {
              alert('Performance test failed: ' + data.result.error);
            } else {
              setQueryPerformance(data.result);
            }
            setLoading(false);
          } else if (data.status === 'failed') {
            alert('Performance test task failed: ' + data.error);
            setLoading(false);
          } else {
            setTimeout(pollPerf, 2000);
          }
        };
        pollPerf();
      } else {
        // If no task_id, assume the results are directly in the response
        console.log('Query Performance Results received directly:', responseData);
        setQueryPerformance(responseData);
        setLoading(false);
      }
    } catch (error) {
      alert('Error running query performance test: ' + error.message);
      setLoading(false);
    }
  };

  return (
    <>
    <Nav/>
    <div className="min-h-screen bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white">
      

      <main className="container mx-auto p-6">
        <h1 className="text-center text-3xl font-bold mb-8">Database Sharding</h1>

        {!selectedDatabaseType && (
          <div className="flex justify-center space-x-4">
            <button
              className="px-6 py-2 bg-blue-600 rounded-lg"
              onClick={() => handleDatabaseTypeSelect('mysql')}
            >
              MySQL
            </button>
            <button
              className="px-6 py-2 bg-green-600 rounded-lg"
              onClick={() => handleDatabaseTypeSelect('mongodb')}
            >
              MongoDB
            </button>
          </div>
        )}

        {loading && <p className="text-center text-xl mt-6">Loading...</p>}

        {selectedDatabaseType && databases.length > 0 && !selectedDatabase && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">
              Available Databases in {selectedDatabaseType}
            </h2>
            <div className="grid grid-cols-4 gap-4">
              {databases.map((db) => (
                <button
                  key={db}
                  className="px-4 py-2 bg-gray-700 rounded-lg"
                  onClick={() => handleDatabaseSelect(db)}
                >
                  {db}
                </button>
              ))}
            </div>
          </div>
        )}

        {selectedDatabase && tables.length > 0 && !selectedTable && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Tables in {selectedDatabase}</h2>
            <div className="grid grid-cols-4 gap-4">
              {tables.map((table) => (
                <button
                  key={table}
                  className="px-4 py-2 bg-gray-700 rounded-lg"
                  onClick={() => handleTableSelect(table)}
                >
                  {table}
                </button>
              ))}
            </div>
          </div>
        )}

        {selectedTable && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Sharding Options for {selectedTable}</h2>

            {selectedDatabaseType === 'mongodb' ? (
              <>
                <label>Shard Key (Field to Distribute Data):</label>
                <select
                  className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                  onChange={(e) => setSelectedColumn(e.target.value)}
                  value={selectedColumn}
                >
                  <option value="">Select a Field</option>
                  {sortedColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>

                <label>Sharding Method:</label>
                <select
                  className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                  onChange={(e) => setShardingMethod(e.target.value)}
                  value={shardingMethod}
                >
                  <option value="hash">Hash-Based</option>
                  <option value="range">Range-Based</option>
                </select>

                <button
                  className="px-6 py-2 bg-yellow-600 rounded-lg mt-4"
                  onClick={handleShard}
                >
                  Apply MongoDB Sharding
                </button>

                <div>
                  <label>Enter MongoDB Query:</label>
                  <input
                    type="text"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                    placeholder='Enter filter as JSON, e.g., {"key": "value"}'
                  />

                  <button
                    className="px-6 py-2 bg-green-600 rounded-lg mt-4"
                    onClick={handleQueryPerformanceTest}
                  >
                    Run Query & Compare Time
                  </button>
                </div>
              </>
            ) : (
              <>
                <label>Sharding Column:</label>
                <select
                  className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                  onChange={(e) => setSelectedColumn(e.target.value)}
                  value={selectedColumn}
                >
                  {/* Add a default empty option or ensure selectedColumn is set to a valid value */}
                  <option value="">Select a Column</option>
                  {sortedColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>

                <label>Sharding Method:</label>
                <select
                  className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                  onChange={(e) => setShardingMethod(e.target.value)}
                  value={shardingMethod}
                >
                  <option value="hash">Hash-Based</option>
                  <option value="range">Range-Based</option>
                  <option value="custom">Custom (User Defined)</option>
                </select>

                {shardingMethod === 'custom' && (
                  <div className="mt-4">
                    <label>Define Custom Sharding Rules (JSON Format):</label>
                    <textarea
                      className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full"
                      rows="4"
                      placeholder='Example: {"p0": ["UK", "Canada"], "p1": ["India", "USA"]}'
                      value={customShardingRules}
                      onChange={(e) => setCustomShardingRules(e.target.value)}
                    />
                  </div>
                )}

                <label>Number of Partitions:</label>
                <select
                  className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                  onChange={(e) => setNumPartitions(e.target.value)}
                  value={numPartitions}
                >
                  {[2, 4, 6].map((num) => (
                    <option key={num} value={num}>
                      {num}
                    </option>
                  ))}
                </select>

                <button
                  className="px-6 py-2 bg-yellow-600 rounded-lg mt-4"
                  onClick={handleShard}
                >
                  Apply MySQL Sharding
                </button>

                <div className="mt-12 p-6 border border-gray-700 rounded-lg bg-gray-900">
                  <h2 className="text-xl font-semibold mb-4">
                    Run Custom Query & Compare Execution Time
                  </h2>

                  <label>Select Partition to Query:</label>
                  <select
                    onChange={(e) => setSelectedPartition(e.target.value)}
                    value={selectedPartition}
                  >
                    {availablePartitions.length > 0 ? (
                      availablePartitions.map((partition) => (
                        <option key={partition} value={partition}>
                          {partition}
                        </option>
                      ))
                    ) : (
                      <option value="">
                        No partitions available
                      </option> // ✅ If no shards exist
                    )}
                  </select>

                  <label>Enter SQL Query:</label>
                  <input
                    type="text"
                    value={userQuery}
                    onChange={(e) => setUserQuery(e.target.value)}
                    className="bg-gray-800 text-white border border-gray-700 rounded px-2 py-1 w-full mt-2"
                    placeholder="Example: SELECT * FROM table WHERE column LIKE '%value%'"
                  />

                  <button
                    className="px-6 py-2 bg-green-600 rounded-lg mt-4"
                    onClick={handleQueryPerformanceTest}
                  >
                    Run Query & Compare Time
                  </button>
                </div>
              </>
            )}

            {queryPerformance && (
              <div className="mt-8 p-6 border border-green-500 rounded-lg bg-gray-900">
                <h2 className="text-xl font-semibold mb-4">
                  Query Performance Comparison
                </h2>
                {selectedDatabaseType === 'mysql' ? (
                  <>
                    <p className="text-green-400">
                      Time Taken (Unsharded Table):{' '}
                      <strong>{queryPerformance.timeTakenUnsharded}</strong> ms
                    </p>
                    <p className="text-yellow-400">
                      Time Taken (Sharded Table):{' '}
                      <strong>{queryPerformance.timeTakenSharded}</strong> ms
                    </p>

                    <div className="mt-6">
                      {(() => {
                        const rows = queryPerformance.queryResult || [];
                        if (!rows.length) return <p>No rows matched the query.</p>;

                        return (
                          <>
                            <h3 className="text-lg font-semibold text-gray-300">
                              Query Results (first 5 rows – source:{' '}
                              {queryPerformance.resultSource})
                            </h3>
                            {/* scrollable container */}
                            <div className="overflow-x-auto border rounded mt-2">
                              <table className="min-w-full bg-gray-800 text-white">
                                <thead>
                                  <tr>
                                    {Object.keys(rows[0]).map((field) => (
                                      <th key={field} className="px-4 py-2 text-left">
                                        {field}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {rows.map((row, idx) => (
                                    <tr
                                      key={idx}
                                      className={idx % 2 === 0 ? 'bg-gray-700' : ''}
                                    >
                                      {Object.values(row).map((val, j) => (
                                        <td key={j} className="px-4 py-2">
                                          {val?.toString() ?? ''}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-green-400">
                      Time Taken (Unsharded Collection):{' '}
                      <strong>{queryPerformance.unsharded.timeTaken}</strong> ms
                    </p>
                    <p className="text-yellow-400">
                      Time Taken (Sharded Collection):{' '}
                      <strong>{queryPerformance.sharded.timeTaken}</strong> ms
                    </p>
                    <div className="mt-6">
                      <h3 className="text-lg font-semibold text-gray-300">
                        Results from Sharded Collection
                      </h3>
                      {queryPerformance.sharded.results &&
                      queryPerformance.sharded.results.length > 0 ? (
                        <div className="overflow-x-auto border rounded mt-2">
                          <table className="min-w-full bg-gray-800 text-white">
                            <thead>
                              <tr>
                                {Object.keys(queryPerformance.sharded.results[0]).map(
                                  (field) => (
                                    <th key={field} className="px-4 py-2 text-left">
                                      {field}
                                    </th>
                                  )
                                )}
                              </tr>
                            </thead>
                            <tbody>
                              {queryPerformance.sharded.results.map((row, index) => (
                                <tr
                                  key={index}
                                  className={index % 2 === 0 ? 'bg-gray-700' : ''}
                                >
                                  {Object.values(row).map((value, idx) => (
                                    <td key={idx} className="px-4 py-2">
                                      {value?.toString() ?? ''}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p>No results found for the query.</p>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            {Object.keys(customQueryResults).length > 0 && (
              <div className="mt-6">
                <h2 className="text-xl font-semibold">
                  Query Results from {selectedPartition}
                </h2>
                <table className="table-auto w-full border border-gray-700">
                  <thead>
                    <tr>
                      {Object.keys(customQueryResults[0]).map((field) => (
                        <th key={field} className="border border-gray-700 px-4 py-2">
                          {field}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {customQueryResults.map((row, index) => (
                      <tr key={index}>
                        {Object.values(row).map((value, idx) => (
                          <td key={idx} className="border border-gray-700 px-4 py-2">
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {isSharded && (
              <div className="mt-4">
                <h2 className="text-xl font-semibold">Sharded Data</h2>

                {/* ✅ Debugging: Log partitioned data */}
                {console.log('🔍 Sharded Data Debug (Render Check):', shardedData)}

                {/* ✅ Show MySQL Partitioned Data */}
                {selectedDatabaseType === 'mysql' && Object.keys(shardedData).length > 0 && (
                  <div className="mt-6 p-6 border border-blue-500 rounded-lg bg-gray-900">
                    <h2 className="text-xl font-semibold mb-4">
                      Partitioned Data (First 5 Rows per Partition)
                    </h2>

                    {Object.entries(shardedData).map(([partition, rows]) => (
                      <div key={partition} className="mt-4">
                        <h3 className="text-yellow-400 text-lg">Partition: {partition}</h3>

                        {/* scrollable container */}
                        <div className="overflow-x-auto border rounded mt-2">
                          <table className="min-w-full bg-gray-800 text-white">
                            <thead>
                              <tr>
                                {columns.map((col) => (
                                  <th key={col} className="px-4 py-2 text-left">
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {rows.slice(0, 5).map((row, idx) => (
                                <tr
                                  key={idx}
                                  className={idx % 2 === 0 ? 'bg-gray-700' : ''}
                                >
                                  {columns.map((col, j) => (
                                    <td key={j} className="px-4 py-2">
                                      {row[col] ?? ''}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                )}{' '}
                :{' '}
                {(
                  <p className="text-red-500">No partition data available.</p>
                )}

                {/* ✅ Show MongoDB Sharded Data */}
                {selectedDatabaseType === 'mongodb' && Object.keys(shardedData).length > 0 && (
                  <div className="mt-6 p-6 border border-green-500 rounded-lg bg-gray-900">
                    <h2 className="text-xl font-semibold mb-4">MongoDB Sharding Data</h2>
                    {isSharded && shardingStatus && Object.keys(shardingStatus.shardData).length > 0 ? (
                      <div className="mt-4 p-4 bg-gray-800 border border-gray-600 rounded-lg">
                        <h3 className="text-lg font-semibold text-yellow-400">
                          Sharding Status
                        </h3>
                        <p className="text-gray-300">
                          Total Chunks: {shardingStatus.totalChunks || 0}
                        </p>

                        {/* Display per-shard data */}
                        {Object.entries(shardedData).map(([shard, data]) => (
                          <div key={shard} className="mt-4">
                            <h3 className="text-yellow-400">{shard} (Shard)</h3>
                            <p className="text-gray-300">Chunks: {data.chunkCount}</p>
                            <div className="overflow-x-auto border rounded mt-2">
                              <table className="min-w-full bg-gray-800 text-white">
                                <thead>
                                  <tr>
                                    {data.records.length > 0 &&
                                      Object.keys(data.records[0]).map((field) => (
                                        <th key={field} className="px-4 py-2 text-left">
                                          {field}
                                        </th>
                                      ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {data.records.map((row, idx) => (
                                    <tr
                                      key={idx}
                                      className={idx % 2 === 0 ? 'bg-gray-700' : ''}
                                    >
                                      {Object.values(row).map((val, j) => (
                                        <td key={j} className="px-4 py-2">
                                          {val?.toString() ?? ''}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-red-500">No shard data available.</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
    </>
  );
}
