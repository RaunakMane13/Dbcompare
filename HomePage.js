import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './ProjectOverview.css';
import Nav from './components/Nav';

export default function HomePage() {
  // const [folderSize, setFolderSize] = useState('Loading...'); // State for folder size
  const [userCount, setUserCount] = useState('Loading...'); // State for user count
  const [folderSize, setFolderSize] = useState('');

  useEffect(() => {
    fetch(`${process.env.REACT_APP_API_BASE_URL}/folder-size`)
      .then((res) => res.json())
      .then((data) => {
        console.log("Folder Size Data:", data);
        setFolderSize(data.size);  // ✅ Store the result
      });
  }, []);
  

useEffect(() => {
  fetch(`${process.env.REACT_APP_API_BASE_URL}/user-count`) // Ensure this matches your server's API endpoint
    .then(response => response.json())
    .then(data => setUserCount(data.count))
    .catch(error => {
      console.error('Error fetching user count:', error);
      setUserCount('Error fetching count');
    });
}, []);

  return (
    <>
    <Nav />
    <div className="main-content">
      {/* Header */}
      
      {/* Main Section */}
      <main className="main">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold mb-4">Compare and Replicate Databases for Scalability</h1>
          <h1 className="mb-8">Securely compare and shard databases for scalable data management.</h1>
          <div className="button-group">
            <Link to="/explore" className="button">Explore</Link>
            <Link to="/demo" className="button">Try demo</Link>
          </div>
          <img
            className="aspect-[3/2] w-full rounded-2xl object-cover"
            src="/main_logo.png"
            alt="Main image"
            width={300}
            height={200}
          />
        </div>
      </main>

      {/* Stats Section */}
      <section className="stats">
        <div className="max-w-8xl mx-auto grid grid-cols-4 gap-4">
          <div className="card">
            <h1 className="text-xl">Engaged users</h1>
            <p className="text-2xl">{userCount}</p>
          </div>
          <div className="card">
            <h1 className="text-xl">Databases compared</h1>
            <p className="text-2xl">5TB+</p>
          </div>
          <div className="card">
            <h1 className="text-xl">Data uploaded</h1>
            <p className="text-2xl">{folderSize}</p> {/* Dynamic folder size */}
          </div>
          <div className="card">
            <h1 className="text-xl">Sign up now</h1>
            <p className="text-2xl">Preview</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="max-w-8xl mx-auto flex help-section">
          <div>
            <h1 className="text-xl font-bold">DBCompare</h1>
          </div>
          <ul>
            <li><a href="#" className="hover:underline">FAQ</a></li>
            <li><a href="#" className="hover:underline">Support Center</a></li>
            <li><a href="#" className="hover:underline">Guides</a></li>
            <li><a href="#" className="hover:underline">Contact</a></li>
          </ul>
        </div>
      </footer>
    </div>
    </>
  );
}
