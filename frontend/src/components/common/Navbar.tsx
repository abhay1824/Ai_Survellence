import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield, Radio, Video, Sliders, Database, BarChart3, LogOut, User, Search, Activity, Mail, Cpu } from 'lucide-react';

export const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState('Operator');
  const [role, setRole] = useState('Viewer');

  useEffect(() => {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        setUsername(user.username);
        setRole(user.role);
      } catch (e) {}
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const navItems = [
    { name: 'Command Center', path: '/', icon: Radio },
    { name: 'Live Monitoring', path: '/monitoring', icon: Video },
    { name: 'Forensics', path: '/forensics', icon: Search },
    { name: 'Camera Health', path: '/health', icon: Activity },
    { name: 'Cameras', path: '/cameras', icon: Sliders },
    { name: 'Event Log', path: '/events', icon: Database },
    { name: 'Analytics', path: '/reports', icon: BarChart3 },
    { name: 'Model Lab', path: '/model-test', icon: Cpu },
    { name: 'Email Settings', path: '/settings/email', icon: Mail },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Title / Logo */}
        <div className="flex items-center space-x-3">
          <Shield className="h-6 w-6 text-emerald-400 animate-pulse" />
          <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-lg font-black tracking-wider text-transparent uppercase">
            SOC INTELLIGENCE PLATFORM
          </span>
        </div>

        {/* Navigation Items */}
        <nav className="hidden md:flex space-x-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold tracking-wide uppercase transition-all duration-200 ${
                  isActive
                    ? 'bg-emerald-950 text-emerald-400 border border-emerald-800 shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-150 hover:bg-zinc-900'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* User profile & Actions */}
        <div className="flex items-center space-x-4">
          <div className="hidden lg:flex flex-col items-end">
            <span className="text-xs font-bold text-zinc-200 flex items-center gap-1.5 font-mono">
              <User className="h-3 w-3 text-emerald-400" />
              {username}
            </span>
            <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest font-mono">
              {role}
            </span>
          </div>

          <button
            onClick={handleLogout}
            className="flex items-center justify-center p-2 text-zinc-400 hover:text-red-400 rounded-lg hover:bg-zinc-900 transition-all duration-200"
            title="Log Out"
          >
            <LogOut className="h-5 w-5" />
          </button>
        </div>
      </div>
    </header>
  );
};
export default Navbar;
