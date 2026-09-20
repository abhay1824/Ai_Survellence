import React, { useState, useEffect, useRef } from 'react';
import { Shield, AlertOctagon, Activity, Camera as CameraIcon, CheckCircle, RefreshCw, Layers } from 'lucide-react';
import type { Camera, SurveillanceEvent, ThreatLevel, HeatmapPoint } from '../types/surveillance';
import { cameraService } from '../services/camera';
import { eventService } from '../services/event';
import { WS_BASE_URL, API_BASE_URL } from '../services/api';
import HeatmapOverlay from '../components/map/HeatmapOverlay';
import { LiveHUDOverlay } from '../components/monitoring/LiveHUDOverlay';
import { PolygonEditorModal } from '../components/monitoring/PolygonEditorModal';
import { SirenAudioPlayer } from '../components/monitoring/SirenAudioPlayer';
import { EventTimeline } from '../components/common/EventTimeline';

export const CommandCenter: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-001');
  const [events, setEvents] = useState<SurveillanceEvent[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [heatmapPoints, setHeatmapPoints] = useState<HeatmapPoint[]>([]);

  // WebSocket Live Stream state
  const [frameSrc, setFrameSrc] = useState<string>('');
  const [liveScore, setLiveScore] = useState<number>(0);
  const [liveThreat, setLiveThreat] = useState<ThreatLevel>('LOW');
  const [livePersonCount, setLivePersonCount] = useState<number>(0);
  const [liveCorrelations, setLiveCorrelations] = useState<string[]>([]);

  // UI Toggles
  const [streamSource, setStreamSource] = useState<'webcam' | 'stream'>('stream');
  const [isPolygonEditorOpen, setIsPolygonEditorOpen] = useState(false);
  const [sirenMuted, setSirenMuted] = useState(false);
  const [evidenceModalEvent, setEvidenceModalEvent] = useState<SurveillanceEvent | null>(null);

  // References
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamIntervalRef = useRef<any>(null);

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(loadInitialData, 10000);
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      const cams = await cameraService.getCameras();
      setCameras(cams);

      const sum = await eventService.getDashboardSummary();
      setSummary(sum);

      const evts = await eventService.getEvents({ limit: 50 });
      setEvents(evts.events);

      const charts = await eventService.getAnalyticsCharts();
      if (charts && charts.heatmap_data) {
        setHeatmapPoints(charts.heatmap_data);
      }
    } catch (err) {
      console.error('Command center data fetch error:', err);
    }
  };

  // Connect WebSocket Stream to selected camera
  useEffect(() => {
    connectStream(selectedCameraId);
    return () => {
      disconnectStream();
    };
  }, [selectedCameraId]);

  const connectStream = (cameraId: string) => {
    disconnectStream();

    const token = localStorage.getItem('token') || '';
    const wsUrl = `${WS_BASE_URL}/ws/camera/${cameraId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`WebSocket connected to ${cameraId}`);
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.frame) {
          setFrameSrc(data.frame);
        }
        if (data.threat_score !== undefined) setLiveScore(data.threat_score);
        if (data.highest_threat) setLiveThreat(data.highest_threat);
        if (data.person_count !== undefined) setLivePersonCount(data.person_count);
        if (data.correlated) setLiveCorrelations(data.correlated);
      } catch (e) {
        console.error('WS JSON parse error:', e);
      }
    };

    ws.onerror = (err) => {
      console.warn('WS Stream error:', err);
    };

    wsRef.current = ws;

    // Handle Client Webcam streaming if source is set to webcam
    if (streamSource === 'webcam') {
      startBrowserWebcam(ws);
    }
  };

  const disconnectStream = () => {
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const startBrowserWebcam = async (ws: WebSocket) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = stream;
      await video.play().catch(() => {});
      videoRef.current = video;

      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');

      webcamIntervalRef.current = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN && ctx && video && video.videoWidth > 0 && video.videoHeight > 0) {
          ctx.drawImage(video, 0, 0, 640, 480);
          const frameData = canvas.toDataURL('image/jpeg', 0.6);
          ws.send(JSON.stringify({ frame: frameData }));
        }
      }, 100);
    } catch (err) {
      console.warn('Browser webcam permission error:', err);
    }
  };

  const handleAcknowledge = async (eventId: string) => {
    try {
      await eventService.acknowledgeAlarm(eventId);
      loadInitialData();
    } catch (err) {
      console.error('Acknowledge failed:', err);
    }
  };

  const handleResolve = async (eventId: string) => {
    try {
      await eventService.resolveAlarm(eventId);
      loadInitialData();
    } catch (err) {
      console.error('Resolve failed:', err);
    }
  };

  const handleEscalate = async (eventId: string) => {
    try {
      await eventService.escalateAlarm(eventId);
      loadInitialData();
    } catch (err) {
      console.error('Escalate failed:', err);
    }
  };

  const activeAlarms = events.filter((e) => e.status === 'ACTIVE');
  const selectedCamObj = cameras.find((c) => c.camera_id === selectedCameraId) || cameras[0];

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Audio Siren Synthesizer */}
      <SirenAudioPlayer threatLevel={liveThreat} muted={sirenMuted} />

      {/* Top SOC Metrics Header */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-zinc-900/60 border border-zinc-800 p-4 rounded-xl flex items-center justify-between shadow-xl">
          <div>
            <div className="text-xs text-zinc-400 font-mono uppercase">Surveillance Nodes</div>
            <div className="text-2xl font-black text-white font-mono">{summary?.total_cameras || cameras.length || 5}</div>
          </div>
          <CameraIcon className="w-8 h-8 text-emerald-400 opacity-80" />
        </div>

        <div className="bg-zinc-900/60 border border-zinc-800 p-4 rounded-xl flex items-center justify-between shadow-xl">
          <div>
            <div className="text-xs text-zinc-400 font-mono uppercase">Active Alarms</div>
            <div className="text-2xl font-black text-red-400 font-mono">{activeAlarms.length}</div>
          </div>
          <AlertOctagon className="w-8 h-8 text-red-500 animate-pulse" />
        </div>

        <div className="bg-zinc-900/60 border border-zinc-800 p-4 rounded-xl flex items-center justify-between shadow-xl">
          <div>
            <div className="text-xs text-zinc-400 font-mono uppercase">Today's Incidents</div>
            <div className="text-2xl font-black text-amber-400 font-mono">{summary?.today_events || 0}</div>
          </div>
          <Activity className="w-8 h-8 text-amber-400 opacity-80" />
        </div>

        <div className="bg-zinc-900/60 border border-zinc-800 p-4 rounded-xl flex items-center justify-between shadow-xl">
          <div>
            <div className="text-xs text-zinc-400 font-mono uppercase">System Threat Status</div>
            <div className={`text-lg font-black font-mono ${liveThreat === 'CRITICAL' ? 'text-red-400 animate-pulse' : 'text-emerald-400'}`}>
              {liveThreat}
            </div>
          </div>
          <Shield className="w-8 h-8 text-cyan-400 opacity-80" />
        </div>
      </div>

      {/* Main 3-Column Command Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Interactive Incident Heatmap (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col gap-4 bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" /> GIS Incident Heatmap & Nodes
            </h2>
            <span className="text-[10px] font-mono text-zinc-400 bg-zinc-800 px-2 py-0.5 rounded">Live Map</span>
          </div>

          <div className="h-[380px] w-full">
            <HeatmapOverlay
              cameras={cameras}
              heatmapPoints={heatmapPoints}
              onSelectCamera={(cid) => setSelectedCameraId(cid)}
            />
          </div>

          {/* Camera Selection Dropdown */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-400 font-mono">Select Feed:</label>
            <select
              value={selectedCameraId}
              onChange={(e) => setSelectedCameraId(e.target.value)}
              className="flex-1 bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-3 py-1.5 font-mono focus:outline-none focus:border-emerald-500"
            >
              {cameras.map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id} - {c.camera_name} [{c.status}]
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Center Column: Selected Live Video Stream + Live HUD (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4 bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 shadow-xl">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-zinc-800">
            {/* Live HUD Overlay */}
            <LiveHUDOverlay
              cameraName={selectedCamObj?.camera_name || 'Surveillance Feed'}
              cameraId={selectedCameraId}
              threatScore={liveScore}
              threatLevel={liveThreat}
              personCount={livePersonCount}
              correlatedEvents={liveCorrelations}
              streamSource={streamSource}
              onSourceChange={(src) => {
                setStreamSource(src);
                connectStream(selectedCameraId);
              }}
              onOpenPolygonEditor={() => setIsPolygonEditorOpen(true)}
              sirenMuted={sirenMuted}
              onToggleSiren={() => setSirenMuted(!sirenMuted)}
            />

            {/* Video Frame Display */}
            {frameSrc ? (
              <img src={frameSrc} alt="Live Surveillance Stream" className="w-full h-full object-contain" />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-zinc-500 font-mono text-xs gap-2">
                <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
                <span>Connecting WebSocket feed to {selectedCameraId}...</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Active Alarms List (3 Cols) */}
        <div className="lg:col-span-3 flex flex-col gap-4 bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 shadow-xl max-h-[460px] overflow-hidden">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <AlertOctagon className="w-4 h-4 text-red-500 animate-pulse" /> Active Alarms ({activeAlarms.length})
            </h2>
          </div>

          <div className="flex flex-col gap-2 overflow-y-auto pr-1">
            {activeAlarms.length === 0 ? (
              <div className="py-12 text-center text-xs text-zinc-500 font-mono flex flex-col items-center gap-2">
                <CheckCircle className="w-8 h-8 text-emerald-500 opacity-60" />
                <span>All sectors clear. No active alarms.</span>
              </div>
            ) : (
              activeAlarms.map((a) => (
                <div key={a.event_id} className="bg-zinc-900/90 border border-red-900/60 rounded-lg p-3 flex flex-col gap-2 shadow">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white tracking-wide">{a.event_type}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-red-950 text-red-400 border border-red-800 rounded font-bold">
                      {a.threat_level}
                    </span>
                  </div>

                  <div className="text-[11px] font-mono text-zinc-400 flex justify-between">
                    <span>Node: <strong className="text-white">{a.camera_id}</strong></span>
                    <span>Score: <strong className="text-emerald-400">+{a.threat_score}</strong></span>
                  </div>

                  <div className="flex items-center gap-2 mt-1">
                    <button
                      onClick={() => handleAcknowledge(a.event_id)}
                      className="flex-1 bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-[11px] py-1 rounded font-mono font-bold transition"
                    >
                      ACK
                    </button>
                    <button
                      onClick={() => handleResolve(a.event_id)}
                      className="flex-1 bg-emerald-950 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 text-[11px] py-1 rounded font-mono font-bold transition"
                    >
                      RESOLVE
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section: Real-Time Event Timeline Stream */}
      <EventTimeline
        events={events}
        onAcknowledge={handleAcknowledge}
        onResolve={handleResolve}
        onEscalate={handleEscalate}
        onViewEvidence={(evt) => setEvidenceModalEvent(evt)}
        onExportCSV={() => {
          window.open(eventService.getExportCSVUrl(), '_blank');
        }}
      />

      {/* Polygon Editor Modal */}
      <PolygonEditorModal
        isOpen={isPolygonEditorOpen}
        onClose={() => setIsPolygonEditorOpen(false)}
        cameraId={selectedCameraId}
        cameraName={selectedCamObj?.camera_name || 'Camera'}
        onZoneSaved={loadInitialData}
      />

      {/* Evidence Media Lightbox Modal */}
      {evidenceModalEvent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-6 max-w-2xl w-full flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="text-sm font-bold text-white uppercase font-mono">
                EVIDENCE MEDIA: {evidenceModalEvent.event_type}
              </div>
              <button
                onClick={() => setEvidenceModalEvent(null)}
                className="text-zinc-400 hover:text-white text-xs bg-zinc-800 px-3 py-1 rounded"
              >
                Close
              </button>
            </div>

            {evidenceModalEvent.video_path ? (
              <video
                src={`${API_BASE_URL.replace('/api', '')}${evidenceModalEvent.video_path}`}
                controls
                autoPlay
                className="w-full rounded border border-zinc-800 bg-black max-h-[400px]"
              />
            ) : evidenceModalEvent.image_path ? (
              <img
                src={`${API_BASE_URL.replace('/api', '')}${evidenceModalEvent.image_path}`}
                alt="Event Evidence"
                className="w-full rounded border border-zinc-800 object-contain max-h-[400px]"
              />
            ) : (
              <div className="py-8 text-center text-xs text-zinc-500 font-mono">No evidence media file available.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
export default CommandCenter;
