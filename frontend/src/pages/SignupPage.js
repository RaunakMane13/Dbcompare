import React, { useState } from 'react';
import Nav from './components/Nav';
export default function SignupPage() {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    confirmPassword: ''
  });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    try {
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password
        })
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('User registered successfully');
      } else {
        setError(data.message || 'Error registering user');
      }
    } catch (err) {
      setError('Network error: Could not register user');
    }
  };

  return (
    <>
    <Nav/>
    <div className="min-h-screen bg-gray-400 flex flex-col">
      <main className="flex-grow flex items-center justify-center">
        <div className="bg-white p-8 shadow-lg w-full max-w-md">
          <h2 className="text-2xl font-semibold mb-6">Create an Account</h2>
          {message && <div className="p-4 mb-4 text-green-800 bg-green-200 rounded">{message}</div>}
          {error && <div className="p-4 mb-4 text-red-800 bg-red-200 rounded">{error}</div>}
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="block text-gray-700">Username</label>
              <input
                type="text"
                name="username"
                placeholder="Enter your username"
                value={formData.username}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded mt-1"
                required
              />
            </div>
            <div>
              <label className="block text-gray-700">Password</label>
              <input
                type="password"
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded mt-1"
                required
              />
            </div>
            <div>
              <label className="block text-gray-700">Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                placeholder="Confirm your password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded mt-1"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-500 text-white p-2 rounded mt-4"
            >
              Sign Up
            </button>
          </form>
        </div>
      </main>
      <footer className="bg-gray-800 text-white p-4 flex justify-between">
        <div className="space-x-4">
          <a href="#" className="hover:underline">Privacy Policy</a>
        </div>
        <a href="#" className="hover:underline">Contact Us</a>
      </footer>
    </div>
    </>
  );
}
