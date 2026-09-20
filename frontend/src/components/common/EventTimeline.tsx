import React, { useState } from 'react';
import { Shield, Search, Download, Image as ImageIcon, Video } from 'lucide-react';
import type { SurveillanceEvent } from '../../types/surveillance';

interface EventTimelineProps {
  events: SurveillanceEvent[];
  onAcknowledge?: (eventId: string) => void;
  onResolve?: (eventId: string) => void;
  onEscalate?: (eventId: string) => void;
  onViewEvidence?: (event: SurveillanceEvent) => void;
  onExportCSV?: () => void;
}

export const EventTimeline: React.FC<EventTimelineProps> = ({
  events,
  onAcknowledge,
  onResolve,
  onEscalate,
  onViewEvidence,
  onExportCSV,
}) => {
  const [search, setSearch] = useState('');
  const [threatFilter, setThreatFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');

  const filteredEvents = events.filter((e) => {
    if (threatFilter !== 'ALL' && e.threat_level !== threatFilter) return false;
    if (statusFilter !== 'ALL' && e.status !== statusFilter) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      const matchType = e.event_type.toLowerCase().includes(q);
      const matchCam = e.camera_id.toLowerCase().includes(q);
      const matchId = e.event_id.toLowerCase().includes(q);
      if (!matchType && !matchCam && !matchId) return false;
    }
    return true;
  });

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'CRITICAL':
        return 'text-red-400 bg-red-950/80 border-red-800';
      case 'HIGH':
        return 'text-amber-400 bg-amber-950/80 border-amber-800';
      case 'MEDIUM':
        return 'text-yellow-400 bg-yellow-950/80 border-yellow-800';
      default:
        return 'text-emerald-400 bg-emerald-950/80 border-emerald-800';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ACKNOWLEDGED':
        return <span className="text-[10px] bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded font-mono font-bold">ACKNOWLEDGED</span>;
      case 'RESOLVED':
        return <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded font-mono font-bold">RESOLVED</span>;
      case 'ESCALATED':
        return <span className="text-[10px] bg-purple-950 text-purple-400 border border-purple-800 px-2 py-0.5 rounded font-mono font-bold animate-pulse">ESCALATED</span>;
      default:
        return <span className="text-[10px] bg-red-950 text-red-400 border border-red-800 px-2 py-0.5 rounded font-mono font-bold animate-pulse">ACTIVE ALARM</span>;
    }
  };

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 flex flex-col gap-4 shadow-xl">
      {/* Top Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-emerald-400" />
          <h2 className="text-sm font-bold text-white tracking-wide uppercase">Real-Time Event Timeline</h2>
          <span className="bg-zinc-800 text-zinc-300 text-xs font-mono px-2 py-0.5 rounded-full font-bold">
            {filteredEvents.length}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Search Box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
            <input
              type="text"
              placeholder="Search timeline..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 text-white rounded-lg pl-8 pr-3 py-1 text-xs focus:outline-none focus:border-emerald-500 w-36 sm:w-48"
            />
          </div>

          {/* Threat Filter */}
          <select
            value={threatFilter}
            onChange={(e) => setThreatFilter(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 text-zinc-300 rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Threat Levels</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="HIGH">HIGH</option>
            <option value="MEDIUM">MEDIUM</option>
            <option value="LOW">LOW</option>
          </select>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 text-zinc-300 rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Statuses</option>
            <option value="ACTIVE">ACTIVE</option>
            <option value="ACKNOWLEDGED">ACKNOWLEDGED</option>
            <option value="RESOLVED">RESOLVED</option>
            <option value="ESCALATED">ESCALATED</option>
          </select>

          {/* Export CSV Button */}
          {onExportCSV && (
            <button
              onClick={onExportCSV}
              className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" /> Export CSV
            </button>
          )}
        </div>
      </div>

      {/* Timeline Stream Feed */}
      <div className="flex flex-col gap-2 max-h-96 overflow-y-auto pr-1">
        {filteredEvents.length === 0 ? (
          <div className="py-8 text-center text-xs text-zinc-500 font-mono">No surveillance events match current filters.</div>
        ) : (
          filteredEvents.map((evt) => (
            <div
              key={evt._id || evt.event_id}
              className="bg-zinc-900/60 border border-zinc-800/80 hover:border-zinc-700 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 transition"
            >
              {/* Event Info Left */}
              <div className="flex items-center gap-3">
                <div className={`px-2 py-1 rounded text-xs font-mono font-bold border ${getThreatColor(evt.threat_level)}`}>
                  {evt.threat_level}
                </div>

                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white tracking-wide">{evt.event_type}</span>
                    {getStatusBadge(evt.status)}
                    {evt.person_id && (
                      <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-1.5 py-0.5 rounded">
                        Person #{evt.person_id}
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-zinc-400 font-mono flex items-center gap-3 mt-0.5">
                    <span>Camera: <strong className="text-zinc-200">{evt.camera_id}</strong></span>
                    <span>Score: <strong className="text-emerald-400">+{evt.threat_score}</strong></span>
                    <span>Time: {new Date(evt.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons Right */}
              <div className="flex items-center gap-2">
                {/* Evidence media indicators */}
                {evt.image_path && (
                  <button
                    onClick={() => onViewEvidence && onViewEvidence(evt)}
                    className="p-1 rounded bg-zinc-800 text-emerald-400 hover:text-emerald-300 transition"
                    title="View Evidence Image"
                  >
                    <ImageIcon className="w-4 h-4" />
                  </button>
                )}
                {evt.video_path && (
                  <button
                    onClick={() => onViewEvidence && onViewEvidence(evt)}
                    className="p-1 rounded bg-zinc-800 text-cyan-400 hover:text-cyan-300 transition"
                    title="View Evidence Clip"
                  >
                    <Video className="w-4 h-4" />
                  </button>
                )}

                {/* Alarm Action Controls */}
                {evt.status === 'ACTIVE' && (
                  <>
                    {onAcknowledge && (
                      <button
                        onClick={() => onAcknowledge(evt.event_id)}
                        className="bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-xs px-2.5 py-1 rounded font-mono transition"
                      >
                        ACK
                      </button>
                    )}
                    {onResolve && (
                      <button
                        onClick={() => onResolve(evt.event_id)}
                        className="bg-emerald-950 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 text-xs px-2.5 py-1 rounded font-mono transition"
                      >
                        RESOLVE
                      </button>
                    )}
                    {onEscalate && (
                      <button
                        onClick={() => onEscalate(evt.event_id)}
                        className="bg-purple-950 hover:bg-purple-900 border border-purple-700 text-purple-300 text-xs px-2.5 py-1 rounded font-mono transition"
                      >
                        ESCALATE
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
