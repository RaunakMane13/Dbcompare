import React from 'react';
import { Link } from 'react-router-dom';
import './ProjectOverview.css';

export default function ProjectOverview() {
  return (
    <div className="steam-about">
      {/* Navigation */}
      <header className="header">
        <div className="flex items-center space-x-2">
          <div className="bg-gray-700 p-2 rounded">
            <img src="/database.svg" alt="Logo" width={24} height={24} />
          </div>
          <span className="font-bold">DBCompare</span>
        </div>
        <nav className="nav">
          <a href="/" className="hover:underline">Home</a>
          <Link to="/project-overview" className="hover:underline">Project Overview</Link>
          <a href="#" className="hover:underline">Goals</a>
          <a href="#" className="hover:underline">Features</a>
        </nav>
        <div className="space-x-4">
          <Link to="/signup" className="button">Sign up</Link>
          <Link to="/login" className="button">Login</Link>
        </div>
      </header>

      {/* Hero Section */}
      <div className="hero">
        <div className="hero-content">
          <img src="/po2.svg" alt="Steam Logo" className="hero-logo" />
          <h1 className="hero-title">Why This Project Matters</h1>
          <h3>
            Databases are the backbone of modern software architectures, yet traditional education often overlooks the
            practical challenges and solutions like replication. Our project addresses this gap by providing a practical
            platform to compare performance, optimize queries, and implement replication for scalability and fault
            tolerance.
          </h3>
          <div className="learn-more">
            <button
              className="learn-more-button"
              onClick={() => document.getElementById('why-this-matters').scrollIntoView({ behavior: 'smooth' })}
            >
              LEARN MORE
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>
          </div>

          <div className="stats" id="why-this-matters">
            <div className="stat">
              <div className="stat-label">Comprehensive Comparison</div>
              <div className="stat-value">
                Dive deep into the operational strengths and weaknesses of MySQL, MongoDB, and Neo4j through comparative
                analysis.
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Hands-On Sharding Implementation:</div>
              <div className="stat-value">
                Experience the enhancement of database scalability firsthand by participating in the development of a
                sharding solution using Python.
              </div>
            </div>
          </div>
        </div>

        <div className="hero-image">
          <h1 className="hero-title">Enhancing Database Education through Real-World Applications</h1>
          <h3>
            Join us in exploring the intricate world of databases through our hands-on capstone project. This initiative
            integrates three major database systems: MySQL, MongoDB, and Neo4j, offering students a unique opportunity to
            understand diverse database architectures and their applications in handling large-scale datasets.
          </h3>
          <img src="/po5.svg" alt="Steam Logo" className="hero-logo2" />
          <Link to="/explore" className="install-steam-button">Dive In</Link>
        </div>
      </div>

      {/* Learn More Button */}
    </div>
  );
}
