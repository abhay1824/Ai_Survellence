import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import type { Camera, HeatmapPoint } from '../../types/surveillance';
import 'leaflet/dist/leaflet.css';

interface HeatmapOverlayProps {
  cameras: Camera[];
  selectedCameraId?: string;
  heatmapPoints?: HeatmapPoint[];
  center?: [number, number];
  zoom?: number;
  userLocation?: { lat: number; lng: number; accuracy?: number } | null;
  isSettingLocation?: boolean;
  onSelectCamera?: (cameraId: string) => void;
  onMapClick?: (lat: number, lng: number) => void;
}

const ChangeView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  React.useEffect(() => {
    if (center && !isNaN(center[0]) && !isNaN(center[1])) {
      map.flyTo([center[0], center[1]], zoom, { duration: 1.0 });
      map.invalidateSize();
    }
  }, [center[0], center[1], zoom, map]);
  return null;
};

const MapEventsHandler: React.FC<{
  onMapClick?: (lat: number, lng: number) => void;
  isSettingLocation?: boolean;
}> = ({ onMapClick, isSettingLocation }) => {
  useMapEvents({
    click(e) {
      if (isSettingLocation && onMapClick) {
        onMapClick(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
};

const createAccessLocationIcon = () => {
  return L.divIcon({
    className: 'custom-access-marker',
    html: `<div class="relative flex items-center justify-center w-8 h-8 pointer-events-auto">
      <span class="absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-60 animate-ping"></span>
      <span class="relative inline-flex items-center justify-center rounded-full h-5 w-5 bg-cyan-500 border-2 border-white shadow-xl text-black">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3 text-black" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
      </span>
    </div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
};

const createSelectedCameraIcon = (cameraName: string) => {
  return L.divIcon({
    className: 'custom-selected-camera-marker',
    html: `<div class="relative flex flex-col items-center pointer-events-auto">
      <div class="relative flex items-center justify-center w-8 h-8">
        <span class="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-80 animate-ping"></span>
        <span class="relative inline-flex items-center justify-center rounded-full h-6 w-6 bg-emerald-500 border-2 border-white shadow-2xl text-black font-black text-[10px]">
          CAM
        </span>
      </div>
      <div class="bg-zinc-950/95 border-2 border-emerald-400 text-emerald-300 text-[11px] font-mono font-black px-2.5 py-0.5 rounded-md shadow-2xl whitespace-nowrap mt-1 uppercase tracking-wider">
        📍 ${cameraName} (ACCESSED)
      </div>
    </div>`,
    iconSize: [140, 56],
    iconAnchor: [70, 16],
  });
};

const createCameraIcon = (status: Camera['status']) => {
  const statusLower = String(status || '').toLowerCase();
  const isOnline = statusLower === 'online' || statusLower === 'active';
  const isWarning = statusLower === 'warning';
  const isAlert = statusLower === 'critical alert' || statusLower === 'critical';
  const isTampered = statusLower === 'tampered';

  let color = 'bg-zinc-600 border-zinc-500';
  if (isOnline) color = 'bg-emerald-500 border-emerald-400';
  else if (isWarning) color = 'bg-amber-500 border-amber-400';
  else if (isAlert) color = 'bg-red-600 border-red-400 animate-pulse';
  else if (isTampered) color = 'bg-purple-600 border-purple-400 animate-bounce';

  return L.divIcon({
    className: 'custom-camera-marker',
    html: `<div class="relative flex items-center justify-center w-5 h-5 pointer-events-auto">
      <span class="absolute inline-flex h-full w-full rounded-full border opacity-75 ${color}"></span>
      <span class="relative inline-flex rounded-full h-3 w-3 ${color} shadow-lg"></span>
    </div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

export const HeatmapOverlay: React.FC<HeatmapOverlayProps> = ({
  cameras,
  selectedCameraId,
  heatmapPoints: _heatmapPoints = [],
  center = [16.8302, 75.7100],
  zoom = 14,
  userLocation,
  isSettingLocation = false,
  onSelectCamera,
  onMapClick,
}) => {
  return (
    <div className={`w-full h-full min-h-[360px] rounded-xl overflow-hidden border border-zinc-800 shadow-2xl relative ${isSettingLocation ? 'cursor-crosshair' : ''}`}>
      <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
        <ChangeView center={center} zoom={zoom} />
        <MapEventsHandler onMapClick={onMapClick} isSettingLocation={isSettingLocation} />
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Camera Nodes */}
        {cameras.map((camera) => {
          const isSelected = camera.camera_id === selectedCameraId;
          return (
            <Marker
              key={camera._id || camera.camera_id}
              position={[camera.latitude, camera.longitude]}
              icon={isSelected ? createSelectedCameraIcon(camera.camera_name) : createCameraIcon(camera.status)}
              zIndexOffset={isSelected ? 1000 : 0}
              eventHandlers={{
                click: () => onSelectCamera && onSelectCamera(camera.camera_id),
              }}
            >
              <Popup>
                <div className="p-2 space-y-1.5 font-sans text-xs">
                  <div className="font-black text-emerald-400 uppercase flex items-center gap-1">
                    🎥 {camera.camera_name} {isSelected ? '(ACCESSED)' : ''}
                  </div>
                  <div className="text-zinc-300 font-mono text-[11px]">
                    Node ID: <strong>{camera.camera_id}</strong><br />
                    Location: <strong>{camera.latitude.toFixed(5)}, {camera.longitude.toFixed(5)}</strong>
                  </div>
                  <div className="text-[10px] font-bold text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
                    Status: {camera.status} &bull; Stream Active
                  </div>
                  {onSelectCamera && !isSelected && (
                    <button
                      onClick={() => onSelectCamera(camera.camera_id)}
                      className="w-full text-center bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1 px-2 rounded text-[10px] uppercase transition mt-1"
                    >
                      Access Camera Feed
                    </button>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* Live Device Access Location Marker */}
        {userLocation && (
          <>
            <CircleMarker
              center={[userLocation.lat, userLocation.lng]}
              radius={Math.max(16, Math.min(45, (userLocation.accuracy || 20) / 2))}
              pathOptions={{
                color: '#06b6d4',
                fillColor: '#06b6d4',
                fillOpacity: 0.15,
                weight: 1.5,
                dashArray: '4, 4',
              }}
            />
            <Marker
              position={[userLocation.lat, userLocation.lng]}
              icon={createAccessLocationIcon()}
            >
              <Popup>
                <div className="p-2 space-y-1 font-sans text-xs">
                  <div className="font-extrabold text-cyan-400 uppercase flex items-center gap-1">
                    📍 Device Geolocation
                  </div>
                  <div className="text-zinc-200 font-mono text-[11px]">
                    Lat: {userLocation.lat.toFixed(5)}<br />
                    Lon: {userLocation.lng.toFixed(5)}
                  </div>
                  {userLocation.accuracy !== undefined && (
                    <div className="text-[10px] text-zinc-400 font-mono">
                      GPS Accuracy: ±{Math.round(userLocation.accuracy)}m
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          </>
        )}
      </MapContainer>
    </div>
  );
};
export default HeatmapOverlay;
