import './App.css';
import Select from 'react-select';
import React, { useState , useEffect} from 'react';
import axios from 'axios';
import { Link, useNavigate } from 'react-router-dom';
import Nav from './components/Nav';

export default function DemoPage() {
  const databaseOptions = [
    { value: 'mysql', label: 'MySQL' },
    { value: 'mongodb', label: 'MongoDB' }
    // { value: 'neo4j', label: 'Neo4j' }
  ];

  const [file, setFile] = useState(null);
  const [selectedDatabases, setSelectedDatabases] = useState([]);
  const [databaseName, setDatabaseName] = useState('');
  const [fileExtension, setFileExtension] = useState('csv');
  const [uploadDetails, setUploadDetails] = useState([]);
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const navigate = useNavigate(); // To redirect to login page
  // track auth status: null = checking, true = allowed, false = rejected
  const [isAuthenticated, setIsAuthenticated] = useState(null);
  const handleFileChange = (e) => setFile(e.target.files[0]);
  const handleDatabaseChange = (selectedOptions) => setSelectedDatabases(selectedOptions.map(option => option.value));
  const handleDatabaseNameChange = (e) => setDatabaseName(e.target.value);
  const handleFileExtensionChange = (e) => setFileExtension(e.target.value);
  
  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const response = await fetch('/api/home', { credentials: 'include' });
        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
          navigate('/login');
        }
      } catch {
        setIsAuthenticated(false);
        navigate('/login');
      }
    };
    checkLoginStatus();
  }, [navigate]);
  

  // Don't show the page content until we know the authentication status
  if (isAuthenticated === null) {
    return null; // Or a loading spinner
  }

  // If isAuthenticated is false, the useEffect would have already redirected.
  // So, if we reach here and isAuthenticated is true, the user is allowed.

  const handleUpload = async (e) => {
    axios.defaults.baseURL = window.location.origin;
    axios.defaults.withCredentials = true;
    e.preventDefault();
  
    if (!file) {
      alert('Please select a file to upload.');
      return;
    }
  
    if (selectedDatabases.length === 0) {
      alert('Please select at least one database.');
      return;
    }
  
    const formData = new FormData();
    formData.append('file', file);
    formData.append('databases', JSON.stringify(selectedDatabases));
    formData.append('databaseName', databaseName);
    formData.append('fileExtension', fileExtension);
  
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/upload`, formData, {
        withCredentials: true
      });
  
      const taskId = response.data.task_id;
  
      const pollStatus = async () => {
        const statusRes = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/upload-status/${taskId}`, { withCredentials: true });
        const data = statusRes.data;
  
        if (data.status === 'done') {
          setUploadDetails(prev => [
            ...prev.slice(-4),
            ...data.result
          ]);
          setShowSuccessPopup(true);
          setTimeout(() => setShowSuccessPopup(false), 3000);
        } else if (data.status === 'failed') {
          alert("Upload failed: " + data.error);
        } else {
          setTimeout(pollStatus, 2000); // keep polling
        }
      };
  
      pollStatus();
  
    } catch (error) {
      console.error('Error during upload:', error);
      alert('There was an error uploading the file.');
    }
  };
  

  return (
    <>
    <Nav />
    <div className="min-h-screen bg-white flex flex-col">
      <div className="flex-1 flex flex-col items-center justify-start p-8">
        <div className="w-full max-w-7xl bg-white rounded-lg shadow-lg p-6">
          {showSuccessPopup && (
            <div className="fixed top-16 right-16 bg-green-500 text-white p-4 rounded shadow-md">
              Upload Successful!
            </div>
          )}
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">Recent Upload History</h1>
          </div>
          <div className="grid grid-cols-3 gap-8">
            <div className="col-span-1 bg-gray-50 p-4 rounded-lg shadow-inner">
              <h2 className="text-lg font-semibold mb-2">Import File Upload</h2>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">File Full</label>
                <input type="file" onChange={handleFileChange} className="w-full p-2 border border-gray-300 rounded" data-rs="13"/>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Select Database</label>
                <Select
                  options={databaseOptions}
                  isMulti
                  className="basic-multi-select"
                  classNamePrefix="select"
                  onChange={handleDatabaseChange}
                />
              </div>
              <div className="mb-4" data-rs="20">
                <label className="block text-sm font-medium text-gray-700 mb-1">Database Name</label>
                <input type="text" onChange={handleDatabaseNameChange} value={databaseName} className="w-full p-2 border border-gray-300 rounded"/>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Select File Extension</label>
                <select onChange={handleFileExtensionChange} value={fileExtension} className="w-full p-2 border border-gray-300 rounded">
                  <option value="csv">CSV</option>
                  <option value="json">JSON</option>
                </select>
              </div>
              <button onClick={handleUpload} className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600">File Upload</button>
            </div>
            <div className="col-span-2">
              <img className="w-full h-full rounded-lg object-cover" src="/data.jpeg" alt="An image of a person" />
            </div>
          </div>
          <div className="mt-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-1 ml-2 flex flex-col items-center">
                <div className="flex items-center mb-4">
                  <img style={{ width: '100px', height: '100px' }} src="/mysql-logo.svg" alt="MySQL"/>
                </div>
                <div className="flex items-center mb-4">
                  <img style={{ width: '200px', height: '100px' }} src="/mongodb-logo.svg" alt="MongoDB"/>
                </div>
                <div className="flex items-center mb-4">
                  <img style={{ width: '100px', height: '100px' }} src="/neo4j-logo.svg" alt="Neo4j"/>
                </div>
              </div>
              <div className="col-span-2 bg-gray-50 p-4 rounded-lg shadow-inner">
                <div className="grid grid-cols-3 gap-4">
                  <div className="col-span-1 font-semibold">Databases</div>
                  <div className="col-span-1 font-semibold">Date</div>
                  <div className="col-span-1 font-semibold">Upload Time</div>
                </div>
                {uploadDetails.slice().reverse().map((detail, index) => (
                  <div key={index} className="grid grid-cols-3 gap-4 mt-2">
                    <div className="col-span-1">{detail.database}</div>
                    <div className="col-span-1">{detail.date}</div>
                    <div className="col-span-1">{detail.timeTaken}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
