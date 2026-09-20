import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, RefreshCw, Radio, Zap, ShieldAlert, VideoOff } from 'lucide-react';
import type { CameraHealth } from '../types/surveillance';
import { cameraService } from '../services/camera';

export const CameraHealthPage: React.FC = () => {
  const [healthList, setHealthList] = useState<CameraHealth[]>([]);

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadHealth = async () => {
    try {
      const data = await cameraService.getCameraHealth();
      setHealthList(data);
    } catch (err) {
      console.error('Camera health fetch error:', err);
    }
  };

  const getStatusBadge = (status: string, freeze: boolean, tamper: boolean) => {
    if (status === 'No Video Input') {
      return (
        <span className="bg-amber-950/80 text-amber-400 border border-amber-800 text-xs px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1">
          <VideoOff className="w-3.5 h-3.5" /> NO VIDEO INPUT
        </span>
      );
    }
    if (tamper) {
      return (
        <span className="bg-purple-950 text-purple-400 border border-purple-800 text-xs px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1 animate-pulse">
          <ShieldAlert className="w-3.5 h-3.5" /> TAMPERED / COVERED
        </span>
      );
    }
    if (freeze) {
      return (
        <span className="bg-yellow-950 text-yellow-400 border border-yellow-800 text-xs px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1">
          <AlertTriangle className="w-3.5 h-3.5" /> FRAME FREEZE
        </span>
      );
    }
    if (status === 'Offline') {
      return (
        <span className="bg-zinc-800 text-zinc-400 border border-zinc-700 text-xs px-2.5 py-1 rounded font-mono font-bold">
          OFFLINE
        </span>
      );
    }
    return (
      <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1">
        <CheckCircle className="w-3.5 h-3.5" /> ONLINE & STABLE
      </span>
    );
  };

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
        <div className="flex items-center gap-3">
          <Activity className="w-7 h-7 text-emerald-400" />
          <div>
            <h1 className="text-xl font-bold text-white tracking-wide uppercase">CAMERA HEALTH & TAMPERING MONITORING</h1>
            <p className="text-xs text-zinc-400">Real-time telemetry, stream FPS, latency, freeze detection, and cover obstruction alerts</p>
          </div>
        </div>

        <button
          onClick={loadHealth}
          className="bg-zinc-800 hover:bg-zinc-700 text-white text-xs px-3.5 py-2 rounded-lg font-semibold flex items-center gap-1.5 border border-zinc-700 transition"
        >
          <RefreshCw className="w-3.5 h-3.5 text-emerald-400" /> Refresh Metrics
        </button>
      </div>

      {/* Camera Grid Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {healthList.map((cam) => (
          <div
            key={cam.id || cam.camera_id}
            className="bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 rounded-xl p-5 shadow-xl flex flex-col justify-between gap-4"
          >
            {/* Header */}
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs font-mono text-emerald-400 font-bold">{cam.camera_id}</div>
                <h3 className="text-base font-bold text-white tracking-wide">{cam.camera_name}</h3>
              </div>
              {getStatusBadge(cam.status, cam.freeze_detected, cam.tamper_detected)}
            </div>

            {/* Health Metrics Gauges */}
            <div className="grid grid-cols-2 gap-3 bg-zinc-950/80 p-3 rounded-lg border border-zinc-800/80 text-xs font-mono">
              <div>
                <div className="text-zinc-500 flex items-center gap-1">
                  <Radio className="w-3.5 h-3.5 text-cyan-400" /> Stream FPS
                </div>
                <div className={`text-base font-bold ${cam.fps < 10 ? 'text-red-400 animate-pulse' : 'text-emerald-400'}`}>
                  {cam.fps.toFixed(1)} FPS
                </div>
              </div>

              <div>
                <div className="text-zinc-500 flex items-center gap-1">
                  <Zap className="w-3.5 h-3.5 text-yellow-400" /> Latency
                </div>
                <div className="text-base font-bold text-white">
                  {cam.latency_ms} ms
                </div>
              </div>
            </div>

            {/* Status Footer */}
            <div className="text-[11px] font-mono text-zinc-500 flex justify-between border-t border-zinc-800/80 pt-3">
              <span>Last Heartbeat:</span>
              <span className="text-zinc-300">
                {cam.last_ping ? new Date(cam.last_ping).toLocaleTimeString() : 'Active'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
export default CameraHealthPage;
