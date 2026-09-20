import React, { useEffect, useState, useRef } from 'react';
import { Loader, Heart, ShieldAlert, WifiOff } from 'lucide-react';
import { WS_BASE_URL } from '../../services/api';

interface Detections {
  bbox: number[];
  class: string;
  confidence: number;
  track_id: number | null;
  threat_level: string;
}

interface CameraGridCellProps {
  cameraId: string;
  cameraName: string;
  streamUrl: string;
  onAlertTriggered: (alert: { event_type: string; threat_level: string; confidence: number; frame: string }) => void;
}

export const CameraGridCell: React.FC<CameraGridCellProps> = ({
  cameraId,
  cameraName,
  streamUrl,
  onAlertTriggered,
}) => {
  const [frameData, setFrameData] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detections[]>([]);
  const [streamHealth, setStreamHealth] = useState<'CONNECTING' | 'STABLE' | 'DEGRADED' | 'DISCONNECTED'>('DISCONNECTED');
  const wsRef = useRef<WebSocket | null>(null);

  // Keep alert callback inside a mutable ref so changes to onAlertTriggered don't retrigger the main WebSocket hook
  const onAlertTriggeredRef = useRef(onAlertTriggered);
  useEffect(() => {
    onAlertTriggeredRef.current = onAlertTriggered;
  }, [onAlertTriggered]);

  useEffect(() => {
    setStreamHealth('CONNECTING');
    setFrameData(null);
    setDetections([]);

    const token = localStorage.getItem('token') || '';
    const wsUrl = `${WS_BASE_URL}/ws/camera/${cameraId}?token=${token}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStreamHealth('STABLE');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setFrameData(data.frame);
        setDetections(data.detections || []);
        
        if (data.alerts && data.alerts.length > 0) {
          data.alerts.forEach((alert: any) => {
            onAlertTriggeredRef.current({
              event_type: alert.event_type,
              threat_level: alert.threat_level,
              confidence: alert.confidence,
              frame: data.frame
            });
          });
        }
      } catch (err) {
        console.error(`Error on cell ${cameraId} WebSocket msg:`, err);
      }
    };

    ws.onerror = () => {
      setStreamHealth('DEGRADED');
    };

    ws.onclose = () => {
      setStreamHealth('DISCONNECTED');
    };

    return () => {
      ws.close();
    };
  }, [cameraId, streamUrl]);

  return (
    <div className="relative aspect-video rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 flex flex-col justify-between group select-none shadow-lg">
      {/* HUD Header */}
      <div className="absolute top-2 left-2 z-20 bg-black/60 backdrop-blur-md border border-zinc-850 px-2 py-0.5 rounded text-[10px] font-bold text-zinc-100 flex items-center gap-1.5 uppercase tracking-wider">
        <span className={`h-1.5 w-1.5 rounded-full ${
          streamHealth === 'STABLE' ? 'bg-emerald-400 animate-pulse' :
          streamHealth === 'CONNECTING' ? 'bg-blue-400 animate-spin' :
          streamHealth === 'DEGRADED' ? 'bg-amber-400 animate-ping' : 'bg-red-500'
        }`} />
        {cameraName} ({cameraId})
      </div>

      {/* Main Image Player */}
      <div className="flex-1 w-full h-full flex items-center justify-center relative">
        {streamHealth === 'CONNECTING' ? (
          <div className="flex flex-col items-center space-y-2">
            <Loader className="h-6 w-6 text-blue-500 animate-spin" />
            <p className="text-zinc-650 text-[10px] font-bold uppercase tracking-widest">Handshaking...</p>
          </div>
        ) : streamHealth === 'DISCONNECTED' ? (
          <div className="flex flex-col items-center space-y-2">
            <WifiOff className="h-6 w-6 text-red-500 animate-pulse" />
            <p className="text-red-500/80 text-[10px] font-bold uppercase tracking-widest">Disconnected</p>
          </div>
        ) : frameData ? (
          <img
            src={frameData}
            alt={cameraName}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center space-y-2">
            <Heart className="h-6 w-6 text-zinc-800 animate-pulse" />
            <p className="text-zinc-650 text-[10px] font-bold uppercase tracking-widest">Awaiting Feed...</p>
          </div>
        )}

        {/* Local Detection Count Overlay */}
        {detections.length > 0 && (
          <div className="absolute bottom-2 right-2 z-20 bg-black/75 backdrop-blur-md border border-zinc-800 px-2 py-0.5 rounded text-[9px] font-bold text-zinc-300 flex items-center gap-1 uppercase tracking-wider">
            <ShieldAlert className="h-3 w-3 text-amber-400" />
            Targets: {detections.length}
          </div>
        )}
      </div>
    </div>
  );
};
