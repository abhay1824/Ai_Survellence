import React, { useEffect, useState } from 'react';
import { Database, Filter, Download, Eye, Brain, Mail, Trash2 } from 'lucide-react';
import CommandCenterLayout from '../layouts/CommandCenterLayout';
import { eventService } from '../services/event';
import { cameraService } from '../services/camera';
import type { EventFilters } from '../services/event';
import type { SurveillanceEvent, Camera, XAIExplanation } from '../types';
import { XAIExplanationModal } from '../components/modals/XAIExplanationModal';
import api from '../services/api';

export const EventHistory: React.FC = () => {
  const [events, setEvents] = useState<SurveillanceEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Filter states
  const [page, setPage] = useState(1);
  const [limit] = useState(15);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [selectedThreatLevel, setSelectedThreatLevel] = useState('');
  const [selectedEventType, setSelectedEventType] = useState('');
  const [exporting, setExporting] = useState(false);

  // Evidence & XAI view states
  const [viewImageEvent, setViewImageEvent] = useState<SurveillanceEvent | null>(null);
  const [selectedXAIEvent, setSelectedXAIEvent] = useState<SurveillanceEvent | null>(null);
  const [xaiExplanation, setXAIExplanation] = useState<XAIExplanation | null>(null);
  const [isXAIModalOpen, setIsXAIModalOpen] = useState(false);

  const fetchFiltersAndEvents = async () => {
    try {
      const cams = await cameraService.getCameras();
      setCameras(cams);
      
      const filters: EventFilters = {
        page,
        limit,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        camera_id: selectedCameraId || undefined,
        threat_level: selectedThreatLevel || undefined,
        event_type: selectedEventType || undefined
      };
      
      const data = await eventService.getEvents(filters);
      setEvents(data.events);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to query security logs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiltersAndEvents();
  }, [page, startDate, endDate, selectedCameraId, selectedThreatLevel, selectedEventType]);

  const handleClearFilters = () => {
    setStartDate('');
    setEndDate('');
    setSelectedCameraId('');
    setSelectedThreatLevel('');
    setSelectedEventType('');
    setPage(1);
  };

  const handleViewXAI = async (evt: SurveillanceEvent) => {
    setSelectedXAIEvent(evt);
    if (evt.xai_explanation) {
      setXAIExplanation(evt.xai_explanation);
      setIsXAIModalOpen(true);
      return;
    }

    try {
      const exp = await eventService.getEventExplanation(evt.event_id);
      setXAIExplanation(exp);
      setIsXAIModalOpen(true);
    } catch (err) {
      console.error('Failed to fetch XAI explanation:', err);
      // Fallback display
      setXAIExplanation(null);
      setIsXAIModalOpen(true);
    }
  };
  const handleExportCSV = async () => {
    setExporting(true);
    try {
      const filters = {
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        camera_id: selectedCameraId || undefined
      };
      
      // Perform direct auth request for blob
      const response = await api.get('/events/export', {
        params: filters,
        responseType: 'blob'
      });
      
      // Create local file pointer and download
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `incident_audit_log_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (e) {
      console.error('Failed to compile CSV export:', e);
    } finally {
      setExporting(false);
    }
  };

  const totalPages = Math.ceil(total / limit);

  return (
    <CommandCenterLayout>
      <div className="space-y-6">
        
        {/* Header Block */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-900 pb-5">
          <div>
            <h1 className="text-3xl font-black tracking-wider text-zinc-100 uppercase flex items-center gap-3">
              <Database className="h-8 w-8 text-blue-500" />
              Surveillance Audit Log
            </h1>
            <p className="text-zinc-500 text-sm mt-1">
              Historical ledger of threat events, object detections, and system alarms.
            </p>
          </div>
          
          <button
            onClick={handleExportCSV}
            disabled={exporting}
            className="flex items-center space-x-2 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 disabled:opacity-50 text-zinc-300 font-bold py-2.5 px-4 rounded-lg text-sm tracking-wider uppercase transition shadow-lg self-start sm:self-auto"
          >
            <Download className="h-4 w-4" />
            <span>{exporting ? 'EXPORTING...' : 'EXPORT CSV'}</span>
          </button>
        </div>

        {/* Filter controls panel */}
        <div className="glass-panel p-5 rounded-xl border border-zinc-800 space-y-4">
          <div className="flex items-center space-x-2 text-xs font-bold text-zinc-400 uppercase tracking-widest">
            <Filter className="h-4 w-4 text-blue-450" />
            <span>Audit Query Filters</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
            {/* Camera */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-550 mb-1.5">Camera Source</label>
              <select
                value={selectedCameraId}
                onChange={(e) => { setSelectedCameraId(e.target.value); setPage(1); }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-550"
              >
                <option value="">All cameras</option>
                {cameras.map((c) => (
                  <option key={c._id} value={c.camera_id}>{c.camera_name}</option>
                ))}
              </select>
            </div>

            {/* Threat Level */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-550 mb-1.5">Threat Level</label>
              <select
                value={selectedThreatLevel}
                onChange={(e) => { setSelectedThreatLevel(e.target.value); setPage(1); }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-555"
              >
                <option value="">All levels</option>
                <option value="LOW">LOW</option>
                <option value="MEDIUM">MEDIUM</option>
                <option value="HIGH">HIGH</option>
                <option value="CRITICAL">CRITICAL</option>
              </select>
            </div>

            {/* Event Type */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-550 mb-1.5">Violation Type</label>
              <select
                value={selectedEventType}
                onChange={(e) => { setSelectedEventType(e.target.value); setPage(1); }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-555"
              >
                <option value="">All types</option>
                <option value="Person">Person</option>
                <option value="Loitering Person">Loitering</option>
                <option value="Intruder">Intrusion</option>
                <option value="Knife">Knife</option>
                <option value="Gun">Gun</option>
                <option value="Car">Vehicle (Car)</option>
                <option value="FIGHT BEHAVIOR DETECTED">Fight</option>
                <option value="ACCIDENT DETECTED">Accident</option>
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-550 mb-1.5">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => { setStartDate(e.target.value); setPage(1); }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-555"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-550 mb-1.5">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => { setEndDate(e.target.value); setPage(1); }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-555"
              />
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={handleClearFilters}
              className="text-xs font-bold text-zinc-500 hover:text-zinc-300 uppercase tracking-widest"
            >
              Clear query settings
            </button>
          </div>
        </div>

        {/* Audit Log Table */}
        <div className="glass-panel rounded-xl border border-zinc-800 overflow-hidden shadow-2xl">
          {loading ? (
            <div className="py-20 flex justify-center">
              <div className="h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/40 text-xs font-bold text-zinc-500 uppercase tracking-widest">
                    <th className="py-4 px-6">Timestamp</th>
                    <th className="py-4 px-6">Camera Node</th>
                    <th className="py-4 px-6">Event Type</th>
                    <th className="py-4 px-6">Threat Level</th>
                    <th className="py-4 px-6">Confidence</th>
                    <th className="py-4 px-4">Fight</th>
                    <th className="py-4 px-4">Fight Conf</th>
                    <th className="py-4 px-4">Accident</th>
                    <th className="py-4 px-4">Accident Conf</th>
                    <th className="py-4 px-4">Email Sent</th>
                    <th className="py-4 px-6 text-right">Evidence Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900 text-sm">
                  {events.length > 0 ? (
                    events.map((evt) => (
                      <tr 
                        key={evt._id} 
                        className={`hover:bg-zinc-900/20 transition duration-150 ${
                          evt.threat_level === 'CRITICAL' ? 'bg-red-500/5' :
                          evt.threat_level === 'HIGH' ? 'bg-amber-500/5' : ''
                        }`}
                      >
                        <td className="py-4 px-6 text-zinc-400">
                          {new Date(evt.timestamp).toLocaleString()}
                        </td>
                        <td className="py-4 px-6 font-bold text-zinc-300">{evt.camera_id}</td>
                        <td className="py-4 px-6 text-zinc-200 font-extrabold">{evt.event_type}</td>
                        <td className="py-4 px-6">
                          <span className={`text-[10px] font-bold px-2.5 py-0.5 rounded border uppercase tracking-wider ${
                            evt.threat_level === 'CRITICAL' ? 'bg-red-950/60 text-red-400 border-red-800/40' :
                            evt.threat_level === 'HIGH' ? 'bg-amber-950/60 text-amber-400 border-amber-800/40' :
                            evt.threat_level === 'MEDIUM' ? 'bg-yellow-950/60 text-yellow-400 border-yellow-800/40' :
                            'bg-emerald-950/60 text-emerald-400 border-emerald-800/40'
                          }`}>
                            {evt.threat_level}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-zinc-350 font-mono font-bold">{Math.round(evt.confidence * 100)}%</td>
                        {/* Fight Detected column */}
                        <td className="py-4 px-4 text-center">
                          {evt.fight_detected
                            ? <span className="text-xs font-bold text-red-400" title="Fight detected">✓</span>
                            : <span className="text-zinc-700">—</span>
                          }
                        </td>
                        {/* Fight Confidence column */}
                        <td className="py-4 px-4 font-mono text-xs">
                          {evt.fight_detected && (evt.fight_confidence ?? 0) > 0
                            ? <span className="text-red-400 font-bold">{Math.round((evt.fight_confidence ?? 0) * 100)}%</span>
                            : <span className="text-zinc-700">—</span>
                          }
                        </td>
                        {/* Accident Detected column */}
                        <td className="py-4 px-4 text-center">
                          {evt.accident_detected
                            ? <span className="text-xs font-bold text-orange-400" title="Accident detected">✓</span>
                            : <span className="text-zinc-700">—</span>
                          }
                        </td>
                        {/* Accident Confidence column */}
                        <td className="py-4 px-4 font-mono text-xs">
                          {evt.accident_detected && (evt.accident_confidence ?? 0) > 0
                            ? <span className="text-orange-400 font-bold">{Math.round((evt.accident_confidence ?? 0) * 100)}%</span>
                            : <span className="text-zinc-700">—</span>
                          }
                        </td>
                        {/* Email Sent column */}
                        <td className="py-4 px-4 text-center">
                          {(evt.threat_level === 'HIGH' || evt.threat_level === 'CRITICAL') ? (
                            <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-950/60 text-emerald-400 border-emerald-800/40" title="Email notification dispatched">
                              <Mail className="h-3 w-3" />
                              <span>SENT</span>
                            </span>
                          ) : (
                            <span className="text-zinc-700">—</span>
                          )}
                        </td>
                        <td className="py-4 px-6 text-right">
                          <div className="flex items-center justify-end space-x-2.5">
                            <button
                              onClick={() => handleViewXAI(evt)}
                              className="flex items-center space-x-1.5 text-xs bg-zinc-900 hover:bg-zinc-800 border border-emerald-900/60 text-emerald-400 font-bold px-2.5 py-1.5 rounded transition uppercase tracking-wider"
                            >
                              <Brain className="h-3.5 w-3.5" />
                              <span>Reasoning</span>
                            </button>

                            {evt.image_path ? (
                              <button
                                onClick={() => setViewImageEvent(evt)}
                                className="flex items-center space-x-1.5 text-xs bg-zinc-900 hover:bg-zinc-800 border border-zinc-805 text-blue-400 font-bold px-2.5 py-1.5 rounded transition uppercase tracking-wider"
                              >
                                <Eye className="h-3.5 w-3.5" />
                                <span>Image</span>
                              </button>
                            ) : (
                              <span className="text-zinc-650 text-xs">No Capture</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={9} className="py-12 text-center text-zinc-650 uppercase tracking-widest text-xs font-bold">
                        No events found matching current query parameters
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination Footer */}
          {totalPages > 1 && (
            <div className="border-t border-zinc-900 bg-zinc-950/40 px-6 py-4 flex items-center justify-between">
              <span className="text-xs text-zinc-550 font-bold">
                Showing Page {page} of {totalPages} ({total} entries total)
              </span>
              <div className="flex items-center space-x-2">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage(prev => prev - 1)}
                  className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-3 py-1.5 rounded text-xs font-bold text-zinc-400 transition uppercase disabled:opacity-50"
                >
                  PREVIOUS
                </button>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage(prev => prev + 1)}
                  className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-3 py-1.5 rounded text-xs font-bold text-zinc-400 transition uppercase disabled:opacity-50"
                >
                  NEXT
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Modal: View Capture Image */}
        {viewImageEvent && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <div className="absolute inset-0 bg-black/90" onClick={() => setViewImageEvent(null)} />
            <div className="relative w-full max-w-2xl bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden z-10 p-4">
              <div className="flex justify-between items-center mb-3">
                <span className="text-xs font-bold text-zinc-450 uppercase tracking-widest">Surveillance Evidence Capture</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={async () => {
                      const idToDelete = viewImageEvent.event_id || viewImageEvent._id;
                      if (!window.confirm('Delete this surveillance evidence image?')) return;
                      try {
                        await eventService.deleteEvent(idToDelete);
                        setEvents(prev => prev.filter(ev => ev.event_id !== idToDelete && ev._id !== idToDelete));
                        setViewImageEvent(null);
                      } catch (err) {
                        console.error('Delete failed:', err);
                        alert('Failed to delete image.');
                      }
                    }}
                    className="bg-red-950/80 hover:bg-red-900 border border-red-800 text-red-300 text-xs px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition font-semibold"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Delete Image
                  </button>
                  <button 
                    onClick={() => setViewImageEvent(null)}
                    className="text-xs text-zinc-550 hover:text-zinc-300 font-bold uppercase tracking-wider px-2 py-1"
                  >
                    Close
                  </button>
                </div>
              </div>
              <img 
                src={`http://localhost:8000${viewImageEvent.image_path}`} 
                alt="Captured Surveillance Bounding Box Details" 
                className="w-full rounded-lg border border-zinc-850"
              />
            </div>
          </div>
        )}

        {/* Modal: XAI Explanation Reasoning */}
        <XAIExplanationModal
          isOpen={isXAIModalOpen}
          onClose={() => setIsXAIModalOpen(false)}
          explanation={xaiExplanation}
          event={selectedXAIEvent}
        />
      </div>
    </CommandCenterLayout>
  );
};
export default EventHistory;
