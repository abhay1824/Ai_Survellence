import React from 'react';
import { Shield, Users, Activity, Layers, Radio, Volume2, VolumeX } from 'lucide-react';
import type { ThreatLevel } from '../../types/surveillance';

interface LiveHUDOverlayProps {
  cameraName: string;
  cameraId: string;
  threatScore: number;
  threatLevel: ThreatLevel;
  personCount: number;
  correlatedEvents?: string[];
  streamSource: 'webcam' | 'stream';
  onSourceChange: (source: 'webcam' | 'stream') => void;
  onOpenPolygonEditor: () => void;
  sirenMuted: boolean;
  onToggleSiren: () => void;
}

export const LiveHUDOverlay: React.FC<LiveHUDOverlayProps> = ({
  cameraName,
  cameraId,
  threatScore,
  threatLevel,
  personCount,
  correlatedEvents = [],
  streamSource,
  onSourceChange,
  onOpenPolygonEditor,
  sirenMuted,
  onToggleSiren,
}) => {
  const getThreatBadgeColor = (level: ThreatLevel) => {
    switch (level) {
      case 'CRITICAL':
        return 'bg-red-600 text-white animate-pulse border-red-500';
      case 'HIGH':
        return 'bg-amber-600 text-white border-amber-500';
      case 'MEDIUM':
        return 'bg-yellow-500 text-black border-yellow-400';
      default:
        return 'bg-emerald-600 text-white border-emerald-500';
    }
  };

  return (
    <div className="absolute top-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-auto">
      {/* Left: Camera Info & Threat Score Badge */}
      <div className="flex items-center gap-2 bg-zinc-950/80 backdrop-blur-md border border-zinc-800/80 px-3 py-1.5 rounded-lg shadow-lg">
        <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-400">
          <Activity className="w-4 h-4 text-emerald-400 animate-spin-slow" />
          <span className="font-bold">{cameraId}</span>
          <span className="text-zinc-500">|</span>
          <span className="text-zinc-300 truncate max-w-[140px]">{cameraName}</span>
        </div>

        {/* Live Threat Score Badge */}
        <div className={`px-2.5 py-0.5 rounded text-xs font-mono font-extrabold border ${getThreatBadgeColor(threatLevel)}`}>
          SCORE: {threatScore} [{threatLevel}]
        </div>
      </div>

      {/* Center: Correlated Events Badge if present */}
      {correlatedEvents.length > 0 && (
        <div className="bg-red-950/90 border border-red-600/80 text-red-200 px-3 py-1 rounded-lg text-xs font-mono font-bold animate-bounce shadow-lg flex items-center gap-1.5">
          <Shield className="w-4 h-4 text-red-400" />
          <span>{correlatedEvents[0]}</span>
        </div>
      )}

      {/* Right: Controls & Counters */}
      <div className="flex items-center gap-2">
        {/* Person Counter */}
        <div className="flex items-center gap-1.5 bg-zinc-950/80 backdrop-blur-md border border-zinc-800/80 px-3 py-1.5 rounded-lg text-xs font-mono text-cyan-400 shadow-lg">
          <Users className="w-3.5 h-3.5 text-cyan-400" />
          <span>PEOPLE:</span>
          <span className="font-bold text-white bg-cyan-950 border border-cyan-800 px-1.5 py-0.5 rounded">{personCount}</span>
        </div>

        {/* Polygon Zone Editor Toggle */}
        <button
          onClick={onOpenPolygonEditor}
          className="bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-300 px-2.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-lg transition"
        >
          <Layers className="w-3.5 h-3.5" /> Polygon Zones
        </button>

        {/* Stream Source Toggle (Camera Stream vs Browser Webcam) */}
        <div className="flex items-center bg-zinc-950/80 backdrop-blur-md border border-zinc-800/80 p-0.5 rounded-lg text-xs font-mono shadow-lg">
          <button
            onClick={() => onSourceChange('stream')}
            className={`px-2 py-1 rounded text-[11px] font-medium transition ${
              streamSource === 'stream' ? 'bg-zinc-800 text-white font-bold' : 'text-zinc-400 hover:text-white'
            }`}
          >
            Camera Stream
          </button>
          <button
            onClick={() => onSourceChange('webcam')}
            className={`px-2 py-1 rounded text-[11px] font-medium transition flex items-center gap-1 ${
              streamSource === 'webcam' ? 'bg-emerald-600 text-white font-bold' : 'text-zinc-400 hover:text-white'
            }`}
          >
            <Radio className="w-3 h-3 text-emerald-400 animate-pulse" /> Webcam
          </button>
        </div>

        {/* Siren Mute Toggle */}
        <button
          onClick={onToggleSiren}
          className={`p-1.5 rounded-lg border text-xs shadow-lg transition ${
            sirenMuted
              ? 'bg-zinc-900 border-zinc-800 text-zinc-500'
              : 'bg-red-950 border-red-700 text-red-400 animate-pulse'
          }`}
          title={sirenMuted ? 'Unmute Critical Siren' : 'Mute Critical Siren'}
        >
          {sirenMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
};
