import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css'; // Use the global styles

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const navigate = useNavigate(); // To redirect after successful login

  const handleLogin = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password }),
        credentials: 'include'
      });

      const data = await response.json();

      if (!response.ok) {
        setErrorMessage(data.message || 'Login failed');
      } else {
        setSuccessMessage(data.message || 'Login successful');
        // Redirect to the home page (which should be DemoPage or lead to it) after a successful login
        setTimeout(() => navigate('/'), 1000); 
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrorMessage('Network error: Could not log in.');
    }
  };

  return (
    <div className="main-content flex justify-center items-center h-screen bg-gray-200">
      <div className="card w-96 p-6 bg-white shadow-lg rounded-lg">
        <h2 className="text-center text-2xl font-bold mb-4">Login</h2>
        <form onSubmit={handleLogin}>
          <div className="form-group mb-4">
            <label className="block text-gray-700 font-semibold mb-2">Username:</label>
            <input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="form-input w-full p-2 border border-gray-300 rounded"
            />
          </div>
          <div className="form-group mb-4">
            <label className="block text-gray-700 font-semibold mb-2">Password:</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="form-input w-full p-2 border border-gray-300 rounded"
            />
          </div>
          <button type="submit" className="button w-full p-2 bg-blue-600 text-white font-bold rounded">
            Login
          </button>
        </form>

        {/* Messages */}
        {errorMessage && <p className="text-red-500 text-center mt-4">{errorMessage}</p>}
        {successMessage && <p className="text-green-600 text-center mt-4">{successMessage}</p>}
      </div>
    </div>
  );
};

export default LoginPage;
