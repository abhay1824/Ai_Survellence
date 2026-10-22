import React, { useState, useEffect, useRef } from 'react';
import { MapPin, Compass, Camera as CameraIcon, RefreshCw, Edit3, Check, X, Crosshair, Search, Building2, Navigation } from 'lucide-react';
import type { Camera, ThreatLevel, HeatmapPoint } from '../types/surveillance';
import { cameraService } from '../services/camera';
import { eventService } from '../services/event';
import { WS_BASE_URL } from '../services/api';
import HeatmapOverlay from '../components/map/HeatmapOverlay';
import { LiveHUDOverlay } from '../components/monitoring/LiveHUDOverlay';
import { PolygonEditorModal } from '../components/monitoring/PolygonEditorModal';

const QUICK_LOCATIONS = [
  { name: 'Vijayapura Center', lat: 16.8302, lng: 75.7100 },
  { name: 'Gol Gumbaz', lat: 16.8295, lng: 75.7354 },
  { name: 'Station Road', lat: 16.8220, lng: 75.7150 },
  { name: 'BLDE Campus', lat: 16.8400, lng: 75.7200 },
  { name: 'Ibrahim Rouza', lat: 16.8285, lng: 75.7015 },
  { name: 'Navbagh', lat: 16.8350, lng: 75.7050 },
];

export const CommandCenter: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-001');
  const [heatmapPoints, setHeatmapPoints] = useState<HeatmapPoint[]>([]);

  // GPS Geolocation state (default to Vijayapura)
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number; accuracy?: number } | null>(null);
  const [mapCenter, setMapCenter] = useState<[number, number]>([16.8302, 75.7100]);
  const [mapZoom, setMapZoom] = useState<number>(14);

  // Search & Geocoding state
  const [searchQuery, setSearchQuery] = useState<string>('Vijayapura');
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [resolvedAddress, setResolvedAddress] = useState<string>('Vijayapura, Karnataka');

  // Location Calibration Modal / Mode
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isClickSettingLocation, setIsClickSettingLocation] = useState(false);
  const [editLat, setEditLat] = useState<string>('16.8302');
  const [editLng, setEditLng] = useState<string>('75.7100');
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);

  // WebSocket Live Stream state
  const [frameSrc, setFrameSrc] = useState<string>('');
  const [liveScore, setLiveScore] = useState<number>(0);
  const [liveThreat, setLiveThreat] = useState<ThreatLevel>('LOW');
  const [livePersonCount, setLivePersonCount] = useState<number>(0);
  const [liveCorrelations, setLiveCorrelations] = useState<string[]>([]);

  // UI Toggles
  const [streamSource, setStreamSource] = useState<'webcam' | 'stream'>('stream');
  const [isPolygonEditorOpen, setIsPolygonEditorOpen] = useState(false);

  // References
  const wsRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamIntervalRef = useRef<any>(null);

  // Request device GPS Geolocation
  const requestGPS = (manual: boolean = false) => {
    if (!navigator.geolocation) {
      setUserLocation({ lat: 16.8302, lng: 75.7100 });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        let { latitude, longitude, accuracy } = pos.coords;
        // Desktop wired/fiber ISPs (like Jio Fiber) in North Karnataka route through Bengaluru core gateway (~12.97)
        // If ISP routes to Bengaluru/South Karnataka (lat < 15.0), calibrate to local deployment location in Vijayapura
        if (latitude < 15.0 || Math.abs(latitude - 12.97) < 2.0) {
          latitude = 16.8302;
          longitude = 75.7100;
        }
        setUserLocation({ lat: latitude, lng: longitude, accuracy });
        if (manual && selectedCamObj) {
          saveNewLocation(latitude, longitude);
          setMapCenter([latitude, longitude]);
          setMapZoom(16);
          setSaveSuccessMsg(`📍 Set ${selectedCamObj.camera_name} to local access location (Vijayapura)`);
          setTimeout(() => setSaveSuccessMsg(null), 4000);
        }
      },
      (err) => {
        console.warn('GPS location request error, defaulting to Vijayapura:', err);
        setUserLocation({ lat: 16.8302, lng: 75.7100 });
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  useEffect(() => {
    requestGPS();
    loadInitialData();
    const interval = setInterval(loadInitialData, 10000);
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      const cams = await cameraService.getCameras();
      setCameras(cams);

      // Initialize map center to selected camera
      const current = cams.find(c => c.camera_id === selectedCameraId) || cams[0];
      if (current) {
        setMapCenter([current.latitude, current.longitude]);
        reverseGeocode(current.latitude, current.longitude);
      }

      const charts = await eventService.getAnalyticsCharts();
      if (charts && charts.heatmap_data) {
        setHeatmapPoints(charts.heatmap_data);
      }
    } catch (err) {
      console.error('GPS tracking data fetch error:', err);
    }
  };

  // Reverse geocode lat/lng to human-readable address
  const reverseGeocode = async (lat: number, lon: number) => {
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`, {
        headers: { 'User-Agent': 'AegisSurveillance/1.0' }
      });
      const data = await res.json();
      if (data && data.display_name) {
        let text = data.display_name;
        // Clean out confusing street or clinic names
        text = text.replace(/Warad Hi -Tech Dental Clinic,\s*/gi, '').replace(/Mahatma Gandhi Road,\s*/gi, '');
        const parts = text.split(',').slice(0, 3).join(',');
        setResolvedAddress(parts || 'Vijayapura, Karnataka');
      }
    } catch (e) {
      setResolvedAddress('Vijayapura, Karnataka');
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

  const selectedCamObj = cameras.find((c) => c.camera_id === selectedCameraId) || cameras[0];

  const handleSelectCamera = (cid: string) => {
    setSelectedCameraId(cid);
    const cam = cameras.find(c => c.camera_id === cid);
    if (cam) {
      setMapCenter([cam.latitude, cam.longitude]);
      setMapZoom(15);
      reverseGeocode(cam.latitude, cam.longitude);
    }
  };

  // Search location via OpenStreetMap Nominatim
  const handleSearchLocation = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery.trim())}&format=json&limit=1`;
      const res = await fetch(url, { headers: { 'User-Agent': 'AegisSurveillance/1.0' } });
      const data = await res.json();
      if (data && data.length > 0) {
        const lat = parseFloat(data[0].lat);
        const lon = parseFloat(data[0].lon);
        await saveNewLocation(lat, lon);
        setMapCenter([lat, lon]);
        setMapZoom(16);
        const shortName = data[0].display_name.split(',').slice(0, 2).join(',');
        setResolvedAddress(shortName);
        setSaveSuccessMsg(`Moved ${selectedCamObj?.camera_name || 'Camera'} to ${shortName}!`);
        setTimeout(() => setSaveSuccessMsg(null), 4000);
      } else {
        alert('Location not found. Please try another place name, street, or city.');
      }
    } catch (err) {
      console.error('Location search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  // Handle quick city select
  const handleQuickCity = async (city: typeof QUICK_LOCATIONS[0]) => {
    await saveNewLocation(city.lat, city.lng);
    setMapCenter([city.lat, city.lng]);
    setMapZoom(15);
    setResolvedAddress(city.name);
    setSaveSuccessMsg(`Moved ${selectedCamObj?.camera_name || 'Camera'} to ${city.name}!`);
    setTimeout(() => setSaveSuccessMsg(null), 3000);
  };

  // Handle map click to update camera position
  const handleMapClick = async (lat: number, lng: number) => {
    if (!selectedCamObj) return;
    setEditLat(lat.toFixed(6));
    setEditLng(lng.toFixed(6));
    await saveNewLocation(lat, lng);
    setIsClickSettingLocation(false);
  };

  // Save new location to database and local state
  const saveNewLocation = async (lat: number, lng: number) => {
    if (!selectedCamObj?._id) return;
    try {
      await cameraService.updateCamera(selectedCamObj._id, {
        latitude: lat,
        longitude: lng,
      });
      setCameras(prev => prev.map(c => c.camera_id === selectedCamObj.camera_id ? { ...c, latitude: lat, longitude: lng } : c));
      setMapCenter([lat, lng]);
      reverseGeocode(lat, lng);
      setSaveSuccessMsg(`Updated ${selectedCamObj.camera_name} location!`);
      setTimeout(() => setSaveSuccessMsg(null), 3000);
    } catch (err) {
      console.error('Failed to update camera location:', err);
    }
  };

  const openCalibrationModal = () => {
    if (selectedCamObj) {
      setEditLat(selectedCamObj.latitude.toString());
      setEditLng(selectedCamObj.longitude.toString());
    }
    setIsCalibrating(true);
  };

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-950 border border-emerald-800 rounded-xl text-emerald-400">
            <MapPin className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-black text-white tracking-wide uppercase">GPS TRACKING &amp; CAMERA SCREEN</h1>
            <p className="text-xs text-zinc-400 font-mono">Real-time GPS geolocation, access node tracking, and live video surveillance</p>
          </div>
        </div>

        {/* GPS Location Status Indicator & Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          {saveSuccessMsg && (
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 border border-emerald-800 px-3 py-1 rounded-lg flex items-center gap-1.5 animate-in fade-in">
              <Check className="w-3.5 h-3.5" /> {saveSuccessMsg}
            </span>
          )}

          {isClickSettingLocation && (
            <div className="flex items-center gap-2 bg-amber-950 border border-amber-600 px-3 py-1.5 rounded-lg text-xs font-mono text-amber-300 animate-pulse">
              <Crosshair className="w-4 h-4" />
              <span>Click anywhere on the map to place camera</span>
              <button
                onClick={() => setIsClickSettingLocation(false)}
                className="ml-1 text-zinc-400 hover:text-white"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <button
            onClick={() => requestGPS(true)}
            className="bg-emerald-950 hover:bg-emerald-900 text-emerald-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold flex items-center gap-1.5 border border-emerald-700 transition cursor-pointer"
          >
            <Navigation className="w-3.5 h-3.5 text-emerald-400" />
            Locate Device (GPS)
          </button>

          <button
            onClick={openCalibrationModal}
            className="bg-cyan-950 hover:bg-cyan-900 text-cyan-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold flex items-center gap-1.5 border border-cyan-700 transition cursor-pointer"
          >
            <Edit3 className="w-3.5 h-3.5 text-cyan-400" />
            Set Coordinates
          </button>
        </div>
      </div>

      {/* Main 2-Column Grid: Left GPS Map Screen, Right Camera Screen */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {/* Left Column: GPS Map Screen */}
        <div className="flex flex-col gap-3 bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2.5">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <MapPin className="w-4 h-4 text-emerald-400" /> GPS Node Map &amp; Access Localization
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (selectedCamObj) {
                    setMapCenter([selectedCamObj.latitude, selectedCamObj.longitude]);
                    setMapZoom(16);
                  }
                }}
                className="text-[10px] font-mono bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded transition flex items-center gap-1"
              >
                <Compass className="w-3 h-3" /> Focus {selectedCamObj?.camera_name || 'Node'}
              </button>

              <button
                onClick={() => setIsClickSettingLocation(prev => !prev)}
                className={`text-[10px] font-mono px-2 py-0.5 rounded border transition flex items-center gap-1 ${
                  isClickSettingLocation
                    ? 'bg-amber-600 text-white border-amber-500 animate-pulse'
                    : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border-zinc-700'
                }`}
              >
                <Crosshair className="w-3 h-3 text-cyan-400" />
                {isClickSettingLocation ? 'Click on Map...' : 'Pin on Map'}
              </button>
            </div>
          </div>

          {/* Search Location Bar */}
          <div className="flex items-center gap-2 bg-zinc-950 border border-zinc-800 rounded-lg p-1.5 shadow-md">
            <Search className="w-4 h-4 text-zinc-400 ml-1.5 shrink-0" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearchLocation()}
              placeholder="Search city, neighborhood, or street (e.g. Gol Gumbaz, Station Road, Vijayapura)..."
              className="flex-1 bg-transparent text-xs text-white placeholder-zinc-500 focus:outline-none font-mono"
            />
            <button
              onClick={handleSearchLocation}
              disabled={isSearching}
              className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-1 rounded font-mono font-bold transition flex items-center gap-1 shrink-0 cursor-pointer"
            >
              {isSearching ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <MapPin className="w-3.5 h-3.5" />}
              Find &amp; Set
            </button>
          </div>

          {/* Quick Location Presets */}
          <div className="flex items-center gap-1.5 flex-wrap text-[10px] font-mono">
            <span className="text-zinc-500 flex items-center gap-1 font-bold">
              <Building2 className="w-3 h-3 text-zinc-400" /> Quick Areas:
            </span>
            {QUICK_LOCATIONS.map((c) => (
              <button
                key={c.name}
                onClick={() => handleQuickCity(c)}
                className="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white px-2 py-0.5 rounded border border-zinc-800 transition"
              >
                {c.name}
              </button>
            ))}
          </div>

          {/* GPS Map Container */}
          <div className="h-[430px] w-full rounded-lg overflow-hidden border border-zinc-800">
            <HeatmapOverlay
              cameras={cameras}
              selectedCameraId={selectedCameraId}
              heatmapPoints={heatmapPoints}
              center={mapCenter}
              zoom={mapZoom}
              userLocation={userLocation}
              isSettingLocation={isClickSettingLocation}
              onSelectCamera={handleSelectCamera}
              onMapClick={handleMapClick}
            />
          </div>

          {/* Bottom GPS Location & Camera Selection Info */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div className="bg-zinc-950/80 border border-zinc-800 p-2.5 rounded-lg text-xs font-mono flex flex-col justify-center">
              <span className="text-zinc-500 text-[10px] uppercase font-bold">Accessed Camera Node Location:</span>
              <span className="text-emerald-400 font-bold truncate">
                {selectedCamObj
                  ? `${selectedCamObj.camera_name} — ${selectedCamObj.latitude.toFixed(5)}, ${selectedCamObj.longitude.toFixed(5)}`
                  : 'No camera selected'}
              </span>
              {resolvedAddress && (
                <span className="text-zinc-400 text-[10px] truncate mt-0.5">
                  📍 {resolvedAddress}
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 bg-zinc-950/80 border border-zinc-800 p-2.5 rounded-lg">
              <label className="text-[10px] text-zinc-400 font-mono uppercase font-bold shrink-0">Active Feed:</label>
              <select
                value={selectedCameraId}
                onChange={(e) => handleSelectCamera(e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2.5 py-1 font-mono focus:outline-none focus:border-emerald-500 cursor-pointer"
              >
                {cameras.map((c) => (
                  <option key={c.camera_id} value={c.camera_id}>
                    {c.camera_name} [{c.status}]
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Right Column: Live Camera Screen */}
        <div className="flex flex-col gap-3 bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2.5">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <CameraIcon className="w-4 h-4 text-cyan-400" /> Camera Screen &mdash; {selectedCamObj?.camera_name || selectedCameraId}
            </h2>
            <div className="flex items-center gap-2">
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border uppercase ${
                selectedCamObj?.status?.toLowerCase() === 'online' ? 'bg-emerald-950 text-emerald-400 border-emerald-800' : 'bg-zinc-800 text-zinc-400 border-zinc-700'
              }`}>
                {selectedCamObj?.status || 'Active'}
              </span>
            </div>
          </div>

          {/* Live Video Container */}
          <div className="relative w-full h-[480px] bg-black rounded-lg overflow-hidden border border-zinc-800 flex items-center justify-center">
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
              sirenMuted={true}
              onToggleSiren={() => {}}
            />

            {/* Video Frame Display */}
            {frameSrc ? (
              <img src={frameSrc} alt="Live Surveillance Stream" className="w-full h-full object-contain" />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-zinc-500 font-mono text-xs gap-2">
                <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
                <span>Connecting live camera screen to {selectedCamObj?.camera_name || selectedCameraId}...</span>
              </div>
            )}
          </div>

          {/* Selected Node Telemetry Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-1 text-xs font-mono">
            <div className="bg-zinc-950/80 border border-zinc-800 p-2.5 rounded-lg">
              <div className="text-zinc-500 text-[10px] uppercase font-bold">Node Identifier</div>
              <div className="text-white font-bold">{selectedCamObj?.camera_id || selectedCameraId}</div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 p-2.5 rounded-lg">
              <div className="text-zinc-500 text-[10px] uppercase font-bold">Node Geolocation</div>
              <div className="text-emerald-400 font-bold truncate">
                {selectedCamObj ? `${selectedCamObj.latitude.toFixed(5)}, ${selectedCamObj.longitude.toFixed(5)}` : 'N/A'}
              </div>
            </div>
            <div className="bg-zinc-950/80 border border-zinc-800 p-2.5 rounded-lg col-span-2 sm:col-span-1">
              <div className="text-zinc-500 text-[10px] uppercase font-bold">Stream Mode</div>
              <div className="text-cyan-400 font-bold uppercase">{streamSource === 'webcam' ? 'Client Device Webcam' : 'RTSP / Video Stream'}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Location Calibration Modal */}
      {isCalibrating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-6 max-w-md w-full flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2 text-white font-bold text-sm uppercase">
                <MapPin className="w-4 h-4 text-emerald-400" />
                Set Coordinates: {selectedCamObj?.camera_name}
              </div>
              <button
                onClick={() => setIsCalibrating(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed">
              Enter exact latitude and longitude for this camera node, or pin it directly on the map.
            </p>

            {/* Manual Lat/Lon Inputs */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] font-mono uppercase text-zinc-400 block mb-1">Latitude</label>
                <input
                  type="number"
                  step="0.000001"
                  value={editLat}
                  onChange={(e) => setEditLat(e.target.value)}
                  placeholder="e.g. 17.3850"
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-[10px] font-mono uppercase text-zinc-400 block mb-1">Longitude</label>
                <input
                  type="number"
                  step="0.000001"
                  value={editLng}
                  onChange={(e) => setEditLng(e.target.value)}
                  placeholder="e.g. 78.4867"
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-col gap-2 pt-2 border-t border-zinc-800/80">
              <button
                type="button"
                onClick={() => {
                  setIsCalibrating(false);
                  setIsClickSettingLocation(true);
                }}
                className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-amber-300 text-xs py-2 px-3 rounded flex items-center justify-between transition font-mono"
              >
                <span className="flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5 text-amber-400" /> Click Directly on Map to Place
                </span>
                <span className="text-[10px] text-zinc-500">Interactive</span>
              </button>
            </div>

            {/* Save Button */}
            <div className="flex items-center justify-end gap-2 pt-2 border-t border-zinc-800">
              <button
                type="button"
                onClick={() => setIsCalibrating(false)}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs py-2 px-4 rounded font-mono transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={async () => {
                  const lat = parseFloat(editLat);
                  const lng = parseFloat(editLng);
                  if (!isNaN(lat) && !isNaN(lng)) {
                    await saveNewLocation(lat, lng);
                    setIsCalibrating(false);
                  }
                }}
                className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs py-2 px-4 rounded font-mono font-bold transition flex items-center gap-1.5"
              >
                <Check className="w-3.5 h-3.5" /> Save Location
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Polygon Editor Modal */}
      <PolygonEditorModal
        isOpen={isPolygonEditorOpen}
        onClose={() => setIsPolygonEditorOpen(false)}
        cameraId={selectedCameraId}
        cameraName={selectedCamObj?.camera_name || 'Camera'}
        onZoneSaved={loadInitialData}
      />
    </div>
  );
};
export default CommandCenter;
