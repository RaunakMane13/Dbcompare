import { NavLink, Link } from 'react-router-dom';

export default function Nav() {
  const links = [
    { to: '/', label: 'Home' },
    { to: '/demo', label: 'Demo' },
    { to: '/editor', label: 'Editor' },
    { to: '/explore', label: 'Explore' },
    { to: '/sharding', label: 'Sharding' },
    { to: '/visualize', label: 'Visualize' }
  ].sort((a,b) => a.label.localeCompare(b.label));

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
          <Link to="/signup" className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600">
            Sign up
          </Link>
          <Link to="/login" className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600">
            Login
          </Link>
        </div>
      </div>
    </nav>
  );
}
