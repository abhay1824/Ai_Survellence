import React, { useState, useEffect } from 'react';
import { Search, Image as ImageIcon, Download, X, Trash2 } from 'lucide-react';
import type { SurveillanceEvent, Camera } from '../types/surveillance';
import { eventService } from '../services/event';
import { cameraService } from '../services/camera';
import { API_BASE_URL } from '../services/api';

export const ForensicSearch: React.FC = () => {
  const [events, setEvents] = useState<SurveillanceEvent[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);

  // Search Filters
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [selectedCamera, setSelectedCamera] = useState('');
  const [threatLevel, setThreatLevel] = useState('');
  const [eventType, setEventType] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Selected Detail Modal
  const [selectedEvent, setSelectedEvent] = useState<SurveillanceEvent | null>(null);

  useEffect(() => {
    loadCameras();
    executeSearch();
  }, []);

  const loadCameras = async () => {
    try {
      const data = await cameraService.getCameras();
      setCameras(data);
    } catch (e) {
      console.error('Failed to load cameras:', e);
    }
  };

  const executeSearch = async () => {
    try {
      const res = await eventService.getEvents({
        page: 1,
        limit: 100,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        camera_id: selectedCamera || undefined,
        threat_level: threatLevel || undefined,
        event_type: eventType || undefined,
        search: searchQuery || undefined,
      });
      setEvents(res.events);
    } catch (err) {
      console.error('Forensic search failed:', err);
    }
  };

  const handleDeleteEvent = async (eventId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!window.confirm('Delete this surveillance evidence image?')) return;
    try {
      await eventService.deleteEvent(eventId);
      setEvents(prev => prev.filter(ev => ev.event_id !== eventId && ev._id !== eventId));
      if (selectedEvent && (selectedEvent.event_id === eventId || selectedEvent._id === eventId)) {
        setSelectedEvent(null);
      }
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete evidence.');
    }
  };

  const resetFilters = () => {
    setStartDate('');
    setEndDate('');
    setSelectedCamera('');
    setThreatLevel('');
    setEventType('');
    setSearchQuery('');
    executeSearch();
  };

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
        <div className="flex items-center gap-3">
          <Search className="w-7 h-7 text-emerald-400" />
          <div>
            <h1 className="text-xl font-bold text-white tracking-wide uppercase">FORENSIC SEARCH ENGINE</h1>
            <p className="text-xs text-zinc-400">Search historical surveillance incidents, evidence captures, and metadata</p>
          </div>
        </div>

        <button
          onClick={async () => {
            try {
              await eventService.downloadExportCSV({ camera_id: selectedCamera || undefined });
            } catch (err) {
              console.error('CSV export failed:', err);
              alert('Failed to export CSV.');
            }
          }}
          className="bg-zinc-800 hover:bg-zinc-700 text-white text-xs px-4 py-2 rounded-lg font-semibold flex items-center gap-2 border border-zinc-700 transition"
        >
          <Download className="w-4 h-4 text-emerald-400" /> Export Results (CSV)
        </button>
      </div>

      {/* Multi-Filter Search Panel */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 shadow-xl grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 items-end">
        <div>
          <label className="text-xs text-zinc-400 font-mono block mb-1">Start Date</label>
          <input
            type="datetime-local"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="text-xs text-zinc-400 font-mono block mb-1">End Date</label>
          <input
            type="datetime-local"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="text-xs text-zinc-400 font-mono block mb-1">Camera Node</label>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Cameras</option>
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.camera_name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-zinc-400 font-mono block mb-1">Threat Level</label>
          <select
            value={threatLevel}
            onChange={(e) => setThreatLevel(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-emerald-500"
          >
            <option value="">All Threat Levels</option>
            <option value="CRITICAL">CRITICAL</option>
            <option value="HIGH">HIGH</option>
            <option value="MEDIUM">MEDIUM</option>
            <option value="LOW">LOW</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-zinc-400 font-mono block mb-1">Search Keywords</label>
          <input
            type="text"
            placeholder="Event, Track ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={executeSearch}
            className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold py-2 rounded-lg flex items-center justify-center gap-1.5 transition"
          >
            <Search className="w-3.5 h-3.5" /> Search
          </button>
          <button
            onClick={resetFilters}
            className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs px-3 py-2 rounded-lg font-semibold transition"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Results Grid */}
      <div className="flex flex-col gap-3">
        <div className="text-xs font-mono text-zinc-400 flex items-center justify-between">
          <span>Found <strong className="text-white">{events.length}</strong> matching forensic incidents</span>
          <span>Showing page 1</span>
        </div>

        {events.length === 0 ? (
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-12 text-center text-zinc-500 font-mono text-xs">
            No historical forensic events match the specified search parameters.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {events.map((evt) => (
              <div
                key={evt._id || evt.event_id}
                onClick={() => setSelectedEvent(evt)}
                className="bg-zinc-900/60 border border-zinc-800 hover:border-emerald-500/50 rounded-xl overflow-hidden cursor-pointer flex flex-col justify-between transition shadow-lg group"
              >
                {/* Media Thumbnail */}
                <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden border-b border-zinc-800">
                  {evt.image_path ? (
                    <img
                      src={`${API_BASE_URL.replace('/api', '')}${evt.image_path}`}
                      alt="Event Screenshot"
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                    />
                  ) : (
                    <div className="text-zinc-600 font-mono text-xs flex flex-col items-center gap-1">
                      <ImageIcon className="w-6 h-6" /> No Screenshot
                    </div>
                  )}

                  {/* Threat Tag */}
                  <div className="absolute top-2 left-2 bg-black/80 backdrop-blur px-2 py-0.5 rounded text-[10px] font-mono font-bold text-emerald-400 border border-zinc-700">
                    {evt.threat_level} (+{evt.threat_score})
                  </div>

                  {/* Delete button overlay on card */}
                  <button
                    onClick={(e) => handleDeleteEvent(evt.event_id || evt._id, e)}
                    className="absolute top-2 right-2 bg-red-950/90 hover:bg-red-800 text-red-300 p-1.5 rounded border border-red-700 opacity-0 group-hover:opacity-100 transition shadow"
                    title="Delete Image"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Card Content */}
                <div className="p-3 flex flex-col gap-1.5">
                  <div className="text-xs font-bold text-white group-hover:text-emerald-400 transition">
                    {evt.event_type}
                  </div>

                  <div className="text-[11px] font-mono text-zinc-400 flex items-center justify-between">
                    <span>Node: <strong className="text-zinc-200">{evt.camera_id}</strong></span>
                    <span>Status: <strong className="text-cyan-400">{evt.status}</strong></span>
                  </div>

                  <div className="text-[10px] font-mono text-zinc-500">
                    {new Date(evt.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Forensic Detail Lightbox Drawer */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-6 max-w-3xl w-full flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white font-mono">{selectedEvent.event_type}</h3>
                <p className="text-xs text-zinc-400 font-mono">Event ID: {selectedEvent.event_id}</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleDeleteEvent(selectedEvent.event_id || selectedEvent._id)}
                  className="bg-red-950/80 hover:bg-red-900 border border-red-800 text-red-300 text-xs px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition font-semibold"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Delete Image
                </button>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-zinc-400 hover:text-white p-1 rounded hover:bg-zinc-800"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Image Preview */}
            <div className="w-full aspect-video bg-black border border-zinc-800 rounded-lg overflow-hidden flex items-center justify-center relative">
              {selectedEvent.image_path ? (
                <img
                  src={`${API_BASE_URL.replace('/api', '')}${selectedEvent.image_path}`}
                  alt="Evidence"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-zinc-500 font-mono text-xs">No video or image file associated with this event log.</div>
              )}
            </div>

            {/* Metadata Detail Table */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-zinc-900/60 p-4 rounded-lg border border-zinc-800 text-xs font-mono">
              <div>
                <div className="text-zinc-500">Camera Node</div>
                <div className="text-white font-bold">{selectedEvent.camera_id}</div>
              </div>
              <div>
                <div className="text-zinc-500">Threat Level</div>
                <div className="text-red-400 font-bold">{selectedEvent.threat_level}</div>
              </div>
              <div>
                <div className="text-zinc-500">Threat Score</div>
                <div className="text-emerald-400 font-bold">+{selectedEvent.threat_score}</div>
              </div>
              <div>
                <div className="text-zinc-500">Tracked Person ID</div>
                <div className="text-cyan-400 font-bold">{selectedEvent.person_id ? `#${selectedEvent.person_id}` : 'N/A'}</div>
              </div>
              <div>
                <div className="text-zinc-500">Status</div>
                <div className="text-white font-bold">{selectedEvent.status}</div>
              </div>
              <div>
                <div className="text-zinc-500">Coordinates</div>
                <div className="text-zinc-300">{selectedEvent.latitude.toFixed(4)}, {selectedEvent.longitude.toFixed(4)}</div>
              </div>
              <div className="col-span-2">
                <div className="text-zinc-500">Recorded Timestamp</div>
                <div className="text-zinc-300">{new Date(selectedEvent.timestamp).toLocaleString()}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
export default ForensicSearch;
