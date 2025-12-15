import React, { useEffect, useState } from 'react';
import { NavLink, Link, useNavigate, useLocation } from 'react-router-dom';

export default function Nav() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    fetch('/api/home', { credentials: 'include' })
      .then(async (res) => {
        if (!res.ok) {
          setIsLoggedIn(false);
          setUsername('');
          return;
        }
  
        const data = await res.json();
  
        // ✅ If you did OPTIONAL backend improvement (username field)
        if (data.username) {
          setIsLoggedIn(true);
          setUsername(data.username);
          return;
        }
  
        // ✅ If you keep current backend (message: "Welcome user!")
        const msg = data.message || '';
        const name = msg.replace('Welcome ', '').replace('!', '').trim();
  
        setIsLoggedIn(true);
        setUsername(name);
      })
      .catch(() => {
        setIsLoggedIn(false);
        setUsername('');
      });
  }, []);
  

  const links = [
    // { to: '/', label: 'Home' },
    { to: '/project-overview', label: 'Project Overview' },
    { to: '/upload', label: 'Upload' },
    { to: '/editor', label: 'Editor' },
    { to: '/explore', label: 'Explore' },
    { to: '/sharding', label: 'Sharding' },
    { to: '/visualize', label: 'Visualize' }
  ];
  // .sort((a,b) => a.label.localeCompare(b.label))

  return (
    <nav className="w-full bg-gray-900 text-white px-4 py-3 shadow-md">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-2 shrink-0">
          <div className="bg-gray-700 p-2 rounded">
            <img src="/database.svg" alt="Logo" width={20} height={20} />
          </div>
          <span className="font-bold">DBCompare</span>
        </Link>

        {/* Center links */}
        <div className="flex items-center gap-4 flex-wrap justify-center">
        {links.map(l => (
          <NavLink
            key={l.to}
            to={l.to}
            onClick={(e) => {
              if (location.pathname === l.to) {
                e.preventDefault();           // stop React Router navigation
                window.location.reload();     // force refresh
              }
            }}
            className={({ isActive }) =>
              `hover:underline ${isActive ? 'font-semibold' : ''}`
            }
          >
            {l.label}
          </NavLink>
        ))}
        </div>

        {/* Auth actions */}
        <div className="flex items-center gap-3">
          {!isLoggedIn ? (
            <>
              <Link to="/signup" className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600">
                Sign up
              </Link>
              <Link to="/login" className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600">
                Login
              </Link>
            </>
          ) : (
            <>
              <span className="text-gray-200">
                Hi <span className="font-semibold">{username || 'User'}</span>!
              </span>
          
              <button
                onClick={async () => {
                  try {
                    await fetch('/api/logout', {
                      method: 'POST',
                      credentials: 'include'
                    });
                  } catch (e) {
                    // ignore network errors; still force client-side logout
                  }
                  setIsLoggedIn(false);
                  setUsername('');
                  window.location.href = '/login';
                }}                
                className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600"
              >
                Logout
              </button>
            </>
          )}          
        </div>
      </div>
    </nav>
  );
}
