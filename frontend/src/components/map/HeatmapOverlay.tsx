import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { Camera, HeatmapPoint } from '../../types/surveillance';
import 'leaflet/dist/leaflet.css';

interface HeatmapOverlayProps {
  cameras: Camera[];
  heatmapPoints: HeatmapPoint[];
  center?: [number, number];
  zoom?: number;
  onSelectCamera?: (cameraId: string) => void;
}

const ChangeView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
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
    html: `<div class="relative flex items-center justify-center w-5 h-5">
      <span class="absolute inline-flex h-full w-full rounded-full border opacity-75 ${color}"></span>
      <span class="relative inline-flex rounded-full h-3 w-3 ${color} shadow-lg"></span>
    </div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

export const HeatmapOverlay: React.FC<HeatmapOverlayProps> = ({
  cameras,
  heatmapPoints = [],
  center = [12.9716, 77.5946],
  zoom = 14,
  onSelectCamera,
}) => {
  return (
    <div className="w-full h-full min-h-[360px] rounded-xl overflow-hidden border border-zinc-800 shadow-2xl relative">
      <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
        <ChangeView center={center} zoom={zoom} />
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Heatmap Incident Circles (Density Visualizer) */}
        {heatmapPoints.map((hp, idx) => {
          const radius = Math.max(15, Math.min(45, hp.intensity * 35));
          const circleColor = hp.intensity > 0.8 ? '#ef4444' : hp.intensity > 0.5 ? '#f59e0b' : '#10b981';
          return (
            <CircleMarker
              key={`heat-${idx}`}
              center={[hp.lat, hp.lng]}
              radius={radius}
              pathOptions={{
                color: circleColor,
                fillColor: circleColor,
                fillOpacity: 0.35,
                weight: 1,
              }}
            >
              <Popup>
                <div className="text-xs space-y-1 font-mono">
                  <div className="font-bold text-red-400">{hp.event_type || 'INCIDENT CLUSTER'}</div>
                  <div>Camera: {hp.camera_id || 'NODE'}</div>
                  <div>Threat Level: {hp.threat_level || 'HIGH'}</div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Camera Nodes */}
        {cameras.map((camera) => (
          <Marker
            key={camera._id || camera.camera_id}
            position={[camera.latitude, camera.longitude]}
            icon={createCameraIcon(camera.status)}
            eventHandlers={{
              click: () => onSelectCamera && onSelectCamera(camera.camera_id),
            }}
          >
            <Popup>
              <div className="p-2 space-y-1.5 font-sans">
                <div className="font-bold text-xs text-white uppercase">{camera.camera_name}</div>
                <div className="text-[11px] text-zinc-400 font-mono">ID: {camera.camera_id}</div>
                <div className="text-[11px] font-bold text-emerald-400">Status: {camera.status}</div>
                {onSelectCamera && (
                  <button
                    onClick={() => onSelectCamera(camera.camera_id)}
                    className="w-full text-center bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1 px-2 rounded text-[10px] uppercase transition"
                  >
                    Select Feed
                  </button>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};
export default HeatmapOverlay;
