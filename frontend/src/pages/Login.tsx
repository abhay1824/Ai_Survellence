import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Lock, User as UserIcon, AlertTriangle } from 'lucide-react';
import { authService } from '../services/auth';

export const Login: React.FC = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const data = await authService.login(username, password);
      // Save details to localStorage
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user', JSON.stringify({
        username: data.username,
        role: data.role
      }));
      
      // Redirect to main command center dashboard
      navigate('/');
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Authentication failed. Please verify credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#09090b] px-4 overflow-hidden">
      {/* Sci-Fi Background grid and blur overlays */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f29370a_1px,transparent_1px),linear-gradient(to_bottom,#1f29370a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)]" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[100px] pointer-events-none" />

      {/* Login Terminal panel */}
      <div className="relative w-full max-w-md glass-panel p-8 rounded-2xl border border-zinc-800 shadow-2xl z-10">
        <div className="flex flex-col items-center mb-8">
          <div className="p-3 bg-zinc-900 border border-zinc-800 rounded-xl mb-4 text-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.15)]">
            <Shield className="h-8 w-8 animate-pulse" />
          </div>
          <h1 className="text-2xl font-black tracking-wider text-zinc-100 uppercase">
            EAGLE EYE SURVEILLANCE
          </h1>
          <p className="text-xs text-zinc-500 uppercase tracking-widest mt-1">
            Secure Terminal Authentication
          </p>
        </div>

        {error && (
          <div className="flex items-center space-x-2.5 p-3 rounded-lg border border-red-900/50 bg-red-950/20 text-red-400 text-sm mb-6">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-widest text-zinc-400 mb-2">
              Operator Username
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-zinc-500">
                <UserIcon className="h-4 w-4" />
              </span>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. operator"
                className="w-full bg-zinc-900/60 border border-zinc-800 rounded-xl py-2.5 pl-10 pr-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition-all text-sm"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-widest text-zinc-400 mb-2">
              Security Access Key
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-zinc-500">
                <Lock className="h-4 w-4" />
              </span>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full bg-zinc-900/60 border border-zinc-800 rounded-xl py-2.5 pl-10 pr-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition-all text-sm"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-blue-900/30 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all text-sm tracking-widest uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {loading ? (
              <span className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <span>AUTHENTICATE</span>
            )}
          </button>
        </form>

        <div className="mt-8 pt-6 border-t border-zinc-900/80 text-center">
          <p className="text-xs text-zinc-600 font-semibold uppercase tracking-wider">
            Police Department Sec-Grid Access Node
          </p>
        </div>
      </div>
    </div>
  );
};
export default Login;
