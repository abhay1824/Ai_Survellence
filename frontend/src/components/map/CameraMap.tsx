import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import { Link } from 'react-router-dom';
import L from 'leaflet';
import type { Camera } from '../../types';
import 'leaflet/dist/leaflet.css';

// Leaflet map center helper
const ChangeView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

// Fix Leaflet assets in bundler - use custom glowing DivIcons instead
const createCameraIcon = (status: Camera['status']) => {
  const statusLower = String(status || '').toLowerCase();
  const isOnline = statusLower === 'online' || statusLower === 'active';
  const isWarning = statusLower === 'warning';
  const isAlert = statusLower === 'critical alert' || statusLower === 'critical';
  const isTampered = statusLower === 'tampered';

  let ringColor = 'border-zinc-700';
  let dotColor = 'bg-zinc-600';
  let animation = '';

  if (isOnline) {
    ringColor = 'border-emerald-500';
    dotColor = 'bg-emerald-500';
    animation = 'animate-ping';
  } else if (isWarning) {
    ringColor = 'border-amber-500';
    dotColor = 'bg-amber-500';
    animation = 'animate-ping';
  } else if (isAlert) {
    ringColor = 'border-red-500';
    dotColor = 'bg-red-500';
    animation = 'animate-pulse scale-125 border-red-400';
  } else if (isTampered) {
    ringColor = 'border-purple-500';
    dotColor = 'bg-purple-500';
    animation = 'animate-bounce';
  }

  return L.divIcon({
    className: 'custom-camera-marker',
    html: `
      <div class="relative flex items-center justify-center w-6 h-6">
        <span class="absolute inline-flex h-full w-full rounded-full border ${ringColor} opacity-75 ${animation}"></span>
        <span class="relative inline-flex rounded-full h-3 w-3 ${dotColor} border border-zinc-950 shadow-lg"></span>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
};

interface CameraMapProps {
  cameras: Camera[];
  center?: [number, number];
  zoom?: number;
}

export const CameraMap: React.FC<CameraMapProps> = ({ 
  cameras, 
  center = [16.8302, 75.7100], // Default Vijayapura Karnataka India
  zoom = 12 
}) => {
  return (
    <div className="w-full h-[500px] rounded-xl overflow-hidden border border-zinc-800 shadow-2xl relative z-10">
      <MapContainer 
        center={center} 
        zoom={zoom} 
        scrollWheelZoom={true} 
        style={{ height: '100%', width: '100%' }}
      >
        <ChangeView center={center} zoom={zoom} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {cameras.map((camera) => (
          <Marker
            key={camera._id}
            position={[camera.latitude, camera.longitude]}
            icon={createCameraIcon(camera.status)}
          >
            <Popup>
              <div className="p-2 space-y-2 text-zinc-200">
                <div className="flex items-center justify-between gap-4">
                  <h3 className="font-extrabold text-sm text-zinc-150 uppercase tracking-wide">
                    {camera.camera_name}
                  </h3>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                    camera.status === 'Online' ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/40' :
                    camera.status === 'Warning' ? 'bg-amber-950/60 text-amber-400 border border-amber-850/40' :
                    camera.status === 'Critical Alert' ? 'bg-red-950/60 text-red-400 border border-red-800/40' :
                    'bg-zinc-800 text-zinc-400 border border-zinc-700/40'
                  }`}>
                    {camera.status}
                  </span>
                </div>
                
                <div className="text-xs text-zinc-400 space-y-1">
                  <p><span className="text-zinc-500 font-semibold uppercase">ID:</span> {camera.camera_id}</p>
                  <p><span className="text-zinc-500 font-semibold uppercase">Coords:</span> {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}</p>
                </div>
                
                {camera.status !== 'Offline' ? (
                  <Link
                    to={`/monitoring?camera=${camera.camera_id}`}
                    className="block w-full text-center bg-blue-600 hover:bg-blue-500 text-white font-bold py-1.5 px-3 rounded text-xs tracking-wider uppercase transition-colors"
                  >
                    Open Live Feed
                  </Link>
                ) : (
                  <span className="block w-full text-center bg-zinc-800 text-zinc-500 py-1.5 px-3 rounded text-xs uppercase cursor-not-allowed">
                    Feed Offline
                  </span>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};
export default CameraMap;
