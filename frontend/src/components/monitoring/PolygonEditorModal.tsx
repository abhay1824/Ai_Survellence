import React, { useState, useEffect, useRef } from 'react';
import { Shield, Save, Trash2, X, RotateCcw, AlertTriangle } from 'lucide-react';
import type { RestrictedZone } from '../../types/surveillance';
import { zoneService } from '../../services/zone';

interface PolygonEditorModalProps {
  isOpen: boolean;
  onClose: () => void;
  cameraId: string;
  cameraName: string;
  onZoneSaved?: () => void;
}

export const PolygonEditorModal: React.FC<PolygonEditorModalProps> = ({
  isOpen,
  onClose,
  cameraId,
  cameraName,
  onZoneSaved
}) => {
  const [zones, setZones] = useState<RestrictedZone[]>([]);
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState('Restricted Zone Alpha');
  const [threatLevel, setThreatLevel] = useState<'HIGH' | 'CRITICAL' | 'MEDIUM'>('HIGH');
  const [loading, setLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (isOpen && cameraId) {
      loadZones();
    }
  }, [isOpen, cameraId]);

  const loadZones = async () => {
    try {
      setLoading(true);
      const data = await zoneService.getZonesByCamera(cameraId);
      setZones(data);
    } catch (err) {
      console.error('Failed to load zones:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    drawCanvas();
  }, [zones, currentPoints]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background grid
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw Existing Saved Zones
    zones.forEach((z) => {
      if (z.points.length > 2) {
        ctx.beginPath();
        ctx.moveTo(z.points[0][0], z.points[0][1]);
        for (let i = 1; i < z.points.length; i++) {
          ctx.lineTo(z.points[i][0], z.points[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.fill();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ef4444';
        ctx.font = '12px sans-serif';
        ctx.fillText(`ZONE: ${z.name.toUpperCase()}`, z.points[0][0], Math.max(16, z.points[0][1] - 6));
      }
    });

    // Draw Currently Drawing Points
    if (currentPoints.length > 0) {
      ctx.fillStyle = '#38bdf8';
      currentPoints.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
      });

      if (currentPoints.length > 1) {
        ctx.beginPath();
        ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
        for (let i = 1; i < currentPoints.length; i++) {
          ctx.lineTo(currentPoints[i][0], currentPoints[i][1]);
        }
        if (currentPoints.length > 2) {
          ctx.closePath();
          ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
          ctx.fill();
        }
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    setCurrentPoints((prev) => [...prev, [x, y]]);
  };

  const handleSaveCurrentZone = async () => {
    if (currentPoints.length < 3) {
      alert('Please click at least 3 points to form a polygon zone!');
      return;
    }

    try {
      setLoading(true);
      await zoneService.createZone({
        camera_id: cameraId,
        name: zoneName,
        points: currentPoints,
        threat_level: threatLevel,
      });
      setCurrentPoints([]);
      setZoneName(`Restricted Zone ${zones.length + 2}`);
      await loadZones();
      if (onZoneSaved) onZoneSaved();
    } catch (err) {
      console.error('Failed to create zone:', err);
      alert('Error saving zone');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteZone = async (zoneId: string) => {
    if (!confirm('Are you sure you want to delete this restricted zone?')) return;
    try {
      setLoading(true);
      await zoneService.deleteZone(zoneId);
      await loadZones();
      if (onZoneSaved) onZoneSaved();
    } catch (err) {
      console.error('Failed to delete zone:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
      <div className="bg-zinc-950 border border-emerald-500/40 rounded-xl shadow-2xl w-full max-w-4xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/60">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-emerald-400" />
            <div>
              <h2 className="text-lg font-bold text-white tracking-wide">RESTRICTED AREA POLYGON EDITOR</h2>
              <p className="text-xs text-zinc-400">Node: <span className="text-emerald-400 font-mono">{cameraId}</span> ({cameraName})</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 flex flex-col md:flex-row gap-6">
          {/* Canvas Draw Area */}
          <div className="relative flex-1 bg-black border border-zinc-800 rounded-lg overflow-hidden flex flex-col items-center justify-center">
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              onClick={handleCanvasClick}
              className="cursor-crosshair w-full h-auto max-w-full"
            />
            <div className="absolute bottom-2 left-2 bg-zinc-900/90 text-xs font-mono text-zinc-300 px-3 py-1 rounded border border-zinc-800">
              Click canvas to add polygon vertices. Min 3 points.
            </div>
          </div>

          {/* Controls & Saved Zones sidebar */}
          <div className="w-full md:w-80 flex flex-col gap-4">
            <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-lg p-4 flex flex-col gap-3">
              <h3 className="text-xs font-semibold tracking-wider text-emerald-400 uppercase">Draw New Polygon</h3>

              <div>
                <label className="text-xs text-zinc-400 block mb-1">Zone Label</label>
                <input
                  type="text"
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-700 text-white rounded px-3 py-1.5 text-xs focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-400 block mb-1">Threat Level</label>
                <select
                  value={threatLevel}
                  onChange={(e) => setThreatLevel(e.target.value as any)}
                  className="w-full bg-zinc-900 border border-zinc-700 text-white rounded px-3 py-1.5 text-xs focus:outline-none focus:border-emerald-500"
                >
                  <option value="HIGH">HIGH (+50 Points)</option>
                  <option value="CRITICAL">CRITICAL (+100 Points)</option>
                  <option value="MEDIUM">MEDIUM (+20 Points)</option>
                </select>
              </div>

              <div className="text-xs font-mono text-cyan-400 flex items-center gap-2">
                <span>Vertices Clicked:</span>
                <span className="font-bold text-white bg-zinc-800 px-2 py-0.5 rounded">{currentPoints.length}</span>
              </div>

              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={handleSaveCurrentZone}
                  disabled={loading || currentPoints.length < 3}
                  className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white px-3 py-2 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Save className="w-3.5 h-3.5" /> Save Zone
                </button>
                <button
                  onClick={() => setCurrentPoints([])}
                  className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-2 rounded text-xs font-semibold flex items-center justify-center gap-1 transition"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Clear
                </button>
              </div>
            </div>

            {/* Saved Zones List */}
            <div className="flex-1 bg-zinc-900/40 border border-zinc-800/80 rounded-lg p-4 overflow-y-auto max-h-56">
              <h3 className="text-xs font-semibold tracking-wider text-zinc-400 uppercase mb-3">Active Camera Zones ({zones.length})</h3>
              {zones.length === 0 ? (
                <p className="text-xs text-zinc-500 italic">No polygon zones defined for this camera feed.</p>
              ) : (
                <div className="flex flex-col gap-2">
                  {zones.map((z) => (
                    <div key={z.zone_id} className="bg-zinc-900/80 border border-zinc-800 rounded p-2.5 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-white">{z.name}</div>
                        <div className="text-[10px] font-mono text-red-400 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" /> {z.threat_level} INTRUSION
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteZone(z.zone_id)}
                        className="text-zinc-500 hover:text-red-400 p-1 rounded transition"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-zinc-800 bg-zinc-900/40 flex justify-end">
          <button
            onClick={onClose}
            className="bg-zinc-800 hover:bg-zinc-700 text-white text-xs px-4 py-2 rounded-lg font-medium transition"
          >
            Close Polygon Editor
          </button>
        </div>
      </div>
    </div>
  );
};
