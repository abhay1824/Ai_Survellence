import React, { useEffect, useState, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { Camera, SurveillanceEvent } from '../types';
import { cameraService } from '../services/camera';
import { WS_BASE_URL } from '../services/api';
import { Video, AlertOctagon, Heart, MapPin, Eye, Settings, Loader, Brain, Mail, Activity, AlertTriangle, CheckCircle, Radio, Zap, ShieldAlert, VideoOff } from 'lucide-react';
import CommandCenterLayout from '../layouts/CommandCenterLayout';
import { CameraGridCell } from '../components/monitoring/CameraGridCell';
import { XAIExplanationModal } from '../components/modals/XAIExplanationModal';
import type { XAIExplanation, CameraHealth } from '../types/surveillance';
import { SIREN_TRIGGER_LEVELS } from '../constants/alertLevels';

export const LiveMonitoring: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [layout, setLayout] = useState<number>(1); // 1 = Single feed, 2 = 1x2 split, 4 = 2x2 grid, etc.
  
  // Stream states
  const [frameData, setFrameData] = useState<string | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [highestThreat, setHighestThreat] = useState<string>('LOW');
  const [streamHealth, setStreamHealth] = useState<'EXCELLENT' | 'STABLE' | 'DEGRADED' | 'DISCONNECTED'>('DISCONNECTED');
  const [fps, setFps] = useState(15);
  const [connecting, setConnecting] = useState(false);

  // Fight Detection States
  const [fightDetected, setFightDetected] = useState(false);
  const [fightConfidence, setFightConfidence] = useState(0);
  const [fightParticipants, setFightParticipants] = useState<number[]>([]);
  const [fightDuration, setFightDuration] = useState(0);

  // Accident Detection States
  const [accidentDetected, setAccidentDetected] = useState(false);
  const [accidentConfidence, setAccidentConfidence] = useState(0);
  
  // Alarm states
  const [alarmActive, setAlarmActive] = useState(false);
  const [criticalThreatName, setCriticalThreatName] = useState('');
  const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);
  const [alertModalImage, setAlertModalImage] = useState<string | null>(null);
  
  // Camera Health & Telemetry State
  const [healthList, setHealthList] = useState<CameraHealth[]>([]);

  const loadHealth = async () => {
    try {
      const data = await cameraService.getCameraHealth();
      setHealthList(data);
    } catch (err) {
      console.error('Camera health fetch error:', err);
    }
  };

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: string, freeze: boolean, tamper: boolean) => {
    if (status === 'No Video Input') {
      return (
        <span className="bg-amber-950/80 text-amber-400 border border-amber-800 text-[10px] px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1">
          <VideoOff className="w-3 h-3" /> NO VIDEO INPUT
        </span>
      );
    }
    if (tamper) {
      return (
        <span className="bg-purple-950 text-purple-400 border border-purple-800 text-[10px] px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1 animate-pulse">
          <ShieldAlert className="w-3 h-3" /> TAMPERED / COVERED
        </span>
      );
    }
    if (freeze) {
      return (
        <span className="bg-yellow-950 text-yellow-400 border border-yellow-800 text-[10px] px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" /> FRAME FREEZE
        </span>
      );
    }
    if (status === 'Offline') {
      return (
        <span className="bg-zinc-800 text-zinc-400 border border-zinc-700 text-[10px] px-2 py-0.5 rounded font-mono font-bold">
          OFFLINE
        </span>
      );
    }
    return (
      <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1">
        <CheckCircle className="w-3 h-3" /> ONLINE & STABLE
      </span>
    );
  };

  // XAI Explanation States
  const [activeXAIExplanation, setActiveXAIExplanation] = useState<XAIExplanation | null>(null);
  const [isXAIModalOpen, setIsXAIModalOpen] = useState<boolean>(false);
  const [activeXAIEvent, setActiveXAIEvent] = useState<SurveillanceEvent | null>(null);

  // Email Alert States
  const [emailAlertSent, setEmailAlertSent] = useState(false);
  const [emailRecipientCount, setEmailRecipientCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const lastFrameTime = useRef<number>(Date.now());
  const fpsInterval = useRef<any>(null);
  const frameCount = useRef<number>(0);

  // Alert dismiss latch (keyed by camera_id|event_type)
  const dismissedAlertsRef = useRef<{ [key: string]: number }>({});
  const activeAlertKeyRef = useRef<string | null>(null);
  const recentAlertKeysRef = useRef<{ [key: string]: number }>({});

  const isAlertDismissed = (camId: string, eventType: string): boolean => {
    const key = `${camId}|${eventType}`;
    const now = Date.now();
    const lastSeen = dismissedAlertsRef.current[key];
    if (lastSeen !== undefined) {
      if (now - lastSeen > 3000) {
        delete dismissedAlertsRef.current[key];
        return false;
      }
      dismissedAlertsRef.current[key] = now;
      return true;
    }
    return false;
  };

  const handleDismissModal = () => {
    setIsAlertModalOpen(false);
    setAlarmActive(false);
    const now = Date.now();
    if (activeAlertKeyRef.current) {
      dismissedAlertsRef.current[activeAlertKeyRef.current] = now;
    }
    Object.entries(recentAlertKeysRef.current).forEach(([key, timestamp]) => {
      if (now - timestamp <= 1500) {
        dismissedAlertsRef.current[key] = now;
      }
    });
  };

  // Synthesize alarm sound using Web Audio API (completely client-side, offline ready!)
  const triggerAudioSiren = () => {
    try {
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContext) return;
      
      const audioCtx = new AudioContext();
      
      // Dual oscillators for classic wailing siren sound
      const osc1 = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      osc1.type = 'sawtooth';
      osc2.type = 'sine';
      
      // Siren frequencies slide
      osc1.frequency.setValueAtTime(600, audioCtx.currentTime);
      osc1.frequency.linearRampToValueAtTime(1000, audioCtx.currentTime + 0.3);
      osc1.frequency.linearRampToValueAtTime(600, audioCtx.currentTime + 0.6);
      osc1.frequency.linearRampToValueAtTime(1000, audioCtx.currentTime + 0.9);
      
      osc2.frequency.setValueAtTime(590, audioCtx.currentTime);
      osc2.frequency.linearRampToValueAtTime(990, audioCtx.currentTime + 0.3);
      osc2.frequency.linearRampToValueAtTime(590, audioCtx.currentTime + 0.6);
      osc2.frequency.linearRampToValueAtTime(990, audioCtx.currentTime + 0.9);

      // volume setting
      gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gainNode.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 1.2);
      
      osc1.connect(gainNode);
      osc2.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      osc1.start();
      osc2.start();
      
      osc1.stop(audioCtx.currentTime + 1.2);
      osc2.stop(audioCtx.currentTime + 1.2);
    } catch (e) {
      console.warn('Web Audio API is blocked or failed to initialize:', e);
    }
  };

  // Load Cameras List
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const list = await cameraService.getCameras();
        setCameras(list);
        
        // Match camera from query param
        const camParam = searchParams.get('camera');
        if (camParam) {
          const match = list.find(c => c.camera_id === camParam);
          if (match) setSelectedCamera(match);
          else if (list.length > 0) setSelectedCamera(list[0]);
        } else if (list.length > 0) {
          setSelectedCamera(list[0]);
        }
      } catch (err) {
        console.error('Failed to query camera registry:', err);
      }
    };
    loadCameras();
  }, [searchParams]);

  const isMountedRef = useRef(true);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Handle WebSocket Connection (Pure Receive-Only Stream from Backend CameraManager)
  useEffect(() => {
    if (!selectedCamera) return;

    if (layout > 1) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnecting(false);
      setStreamHealth('DISCONNECTED');
      return;
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    setFrameData(null);
    setDetections([]);
    setHighestThreat('LOW');
    setConnecting(true);
    setStreamHealth('DISCONNECTED');

    const token = localStorage.getItem('token') || '';
    const wsUrl = `${WS_BASE_URL}/ws/camera/${selectedCamera.camera_id}?token=${token}`;
    
    console.log(`Establishing stream: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket stream established.');
      setConnecting(false);
      setStreamHealth('EXCELLENT');
      frameCount.current = 0;
      lastFrameTime.current = Date.now();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setFrameData(data.frame);
        setDetections(data.detections || []);
        setHighestThreat(data.highest_threat || 'LOW');
        
        // Fight Detection
        setFightDetected(data.fight_detected || false);
        setFightConfidence(data.fight_confidence || 0);
        setFightParticipants(data.fight_participants || []);
        setFightDuration(data.fight_duration_sec || 0);

        // Accident Detection
        setAccidentDetected(data.accident_detected || false);
        setAccidentConfidence(data.accident_confidence || 0);
        
        // Track frame counts for FPS
        frameCount.current += 1;
        
        // Process XAI explanation
        if (data.xai_explanation) {
          setActiveXAIExplanation(data.xai_explanation);
        }

        // Process alerts
        if (data.alerts && data.alerts.length > 0) {
          data.alerts.forEach((alert: any) => {
            // Trigger visual alarms and modal if threat is in trigger levels and not latched/dismissed
            const camId = selectedCamera?.camera_id || 'CAM-001';
            const alertKey = `${camId}|${alert.event_type}`;
            recentAlertKeysRef.current[alertKey] = Date.now();
            const dismissed = isAlertDismissed(camId, alert.event_type);

            if (!dismissed && SIREN_TRIGGER_LEVELS.includes(alert.threat_level)) {
              activeAlertKeyRef.current = alertKey;
              setAlarmActive(true);
              setCriticalThreatName(alert.event_type);
              triggerAudioSiren();
              
              // Mark email alert as sent (backend sends automatically)
              setEmailAlertSent(true);
              // Fetch recipient count from settings
              import('../services/user').then(mod => {
                mod.userService.getEmailSettings().then(settings => {
                  let count = 0;
                  if (settings.email) count++;
                  count += (settings.notification_emails || []).length;
                  setEmailRecipientCount(count);
                }).catch(() => {});
              });

              // Only pop modal if not open
              setAlertModalImage(data.frame);
              setIsAlertModalOpen(true);
            }
          });
        }
      } catch (e) {
        console.error('Error decoding WS payload:', e);
      }
    };

    ws.onerror = (e) => {
      console.error('WebSocket stream error:', e);
      setStreamHealth('DEGRADED');
    };

    ws.onclose = () => {
      console.log('WebSocket stream terminated.');
      setStreamHealth('DISCONNECTED');
      setConnecting(false);
    };

    // Calculate FPS every 2 seconds
    fpsInterval.current = setInterval(() => {
      const now = Date.now();
      const diffSeconds = (now - lastFrameTime.current) / 1000;
      if (diffSeconds > 0) {
        setFps(Math.round(frameCount.current / diffSeconds));
      }
      frameCount.current = 0;
      lastFrameTime.current = now;
      
      if (ws.readyState === WebSocket.OPEN) {
        setStreamHealth(prev => prev === 'DEGRADED' ? 'STABLE' : prev);
      }
    }, 2000);

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (fpsInterval.current) {
        clearInterval(fpsInterval.current);
      }
    };
  }, [selectedCamera?.camera_id, selectedCamera?.stream_url, layout]);

  const handleCameraChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const cam = cameras.find(c => c.camera_id === e.target.value);
    if (!cam) return;

    if (cam.stream_url === 'webcam_prompt') {
      const url = prompt(`Enter Phone Camera URL for ${cam.camera_name} (e.g. http://192.168.1.50:8080/video or webcam):`);
      if (url) {
        try {
          const updated = await cameraService.updateCamera(cam._id || cam.camera_id, { stream_url: url });
          setCameras(prev => prev.map(c => (c._id || c.camera_id) === (cam._id || cam.camera_id) ? updated : c));
          setSelectedCamera(updated);
          setSearchParams({ camera: updated.camera_id });
          setAlarmActive(false);
          return;
        } catch (err) {
          console.error("Failed to update camera stream URL:", err);
          alert("Failed to update camera stream URL.");
        }
      }
    }

    setSelectedCamera(cam);
    setSearchParams({ camera: cam.camera_id });
    setAlarmActive(false);
  };

  const handleAcknowledgeAlarm = () => {
    handleDismissModal();
    setEmailAlertSent(false);
    setEmailRecipientCount(0);
  };

  return (
    <CommandCenterLayout>
      <div className="space-y-6">
        {/* TOP: Camera Information & Selector */}
        <div className="glass-panel p-5 rounded-xl border border-zinc-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-zinc-950 border border-zinc-800 rounded-lg text-blue-500">
              <Video className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <select
                  value={selectedCamera?.camera_id || ''}
                  onChange={handleCameraChange}
                  className="bg-zinc-900 border border-zinc-850 hover:border-zinc-700 text-zinc-100 font-extrabold text-sm py-1.5 px-3 rounded-lg focus:outline-none focus:border-blue-500 transition cursor-pointer uppercase tracking-wider"
                >
                  {cameras.map((c) => (
                    <option key={c._id || c.camera_id} value={c.camera_id}>
                      {c.camera_name}
                    </option>
                  ))}
                </select>

                <select
                  value={layout}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setLayout(val);
                  }}
                  className="bg-zinc-900 border border-zinc-850 hover:border-zinc-700 text-zinc-100 font-extrabold text-sm py-1.5 px-3 rounded-lg focus:outline-none focus:border-blue-500 transition cursor-pointer uppercase tracking-wider"
                >
                  <option value={1}>1 Screen</option>
                  <option value={2}>2 Screens (1x2)</option>
                  <option value={4}>4 Screens (2x2)</option>
                  <option value={9}>9 Screens (3x3)</option>
                  <option value={16}>16 Screens (4x4)</option>
                  <option value={25}>25 Screens (5x5)</option>
                  <option value={32}>32 Screens (8x4)</option>
                </select>

                {selectedCamera && layout === 1 && (
                  <div className="flex items-center gap-2">
                    <select
                      value={selectedCamera.stream_url === 'webcam' ? 'webcam' : selectedCamera.stream_url === '0' ? '0' : 'custom'}
                      onChange={async (e) => {
                        const val = e.target.value;
                        let targetUrl = val;
                        if (val === 'custom') {
                          const inputUrl = prompt(`Configure stream URL for ${selectedCamera.camera_name} (e.g. http://192.168.1.50:8080/video or rtsp://...):`, selectedCamera.stream_url === '0' || selectedCamera.stream_url === 'webcam' ? '' : selectedCamera.stream_url);
                          if (!inputUrl) return;
                          targetUrl = inputUrl;
                        }
                        try {
                          const updated = await cameraService.updateCamera(selectedCamera._id || selectedCamera.camera_id, { stream_url: targetUrl });
                          setCameras(prev => prev.map(c => (c._id || c.camera_id) === (selectedCamera._id || selectedCamera.camera_id) ? updated : c));
                          setSelectedCamera(updated);
                        } catch (err) {
                          console.error("Failed to update camera URL:", err);
                          alert("Failed to update stream source.");
                        }
                      }}
                      className="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-zinc-300 font-bold text-xs py-1.5 px-3 rounded-lg focus:outline-none focus:border-blue-500 transition cursor-pointer uppercase tracking-wider"
                    >
                      <option value="webcam">🎥 Browser Webcam</option>
                      <option value="0">💻 Local Camera (OpenCV 0)</option>
                      <option value="custom">🌐 Camera Stream URL / RTSP...</option>
                    </select>
                  </div>
                )}

                <span className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider flex items-center gap-1.5 ${
                  streamHealth === 'EXCELLENT' ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/40' :
                  streamHealth === 'STABLE' ? 'bg-blue-950/60 text-blue-400 border-blue-800/40' :
                  streamHealth === 'DEGRADED' ? 'bg-amber-950/60 text-amber-400 border-amber-800/40' :
                  'bg-zinc-900 text-zinc-500 border-zinc-800'
                }`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${
                    streamHealth === 'EXCELLENT' ? 'bg-emerald-400 animate-pulse' :
                    streamHealth === 'STABLE' ? 'bg-blue-400' :
                    streamHealth === 'DEGRADED' ? 'bg-amber-400 animate-ping' : 'bg-zinc-600'
                  }`} />
                  {streamHealth}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-zinc-550 mt-1 font-semibold">
                <span className="flex items-center gap-1"><MapPin className="h-3 w-3" /> GPS: {selectedCamera?.latitude.toFixed(4)}, {selectedCamera?.longitude.toFixed(4)}</span>
                <span className="flex items-center gap-1"><Eye className="h-3 w-3" /> Source: {selectedCamera?.stream_url}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6 self-end md:self-auto text-xs text-zinc-400 font-mono">
            <div className="text-right">
              <p className="text-zinc-650 font-bold uppercase tracking-wider">Frame Stream Rate</p>
              <p className="text-zinc-200 font-black text-md mt-0.5">{streamHealth !== 'DISCONNECTED' ? `${fps} FPS` : '0 FPS'}</p>
            </div>
            <div className="text-right">
              <p className="text-zinc-650 font-bold uppercase tracking-wider">Alert status</p>
              <p className={`font-black text-md mt-0.5 ${
                highestThreat === 'CRITICAL' ? 'text-red-400 animate-pulse' :
                highestThreat === 'HIGH' ? 'text-amber-400' :
                highestThreat === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'
              }`}>{highestThreat}</p>
            </div>
          </div>
        </div>

        {/* Flashing Alert Banner for High/Critical Threats */}
        {alarmActive && (
          <div className="critical-flash-bg border border-red-500/50 rounded-xl p-4 flex items-center justify-between text-red-200">
            <div className="flex items-center space-x-3">
              <AlertOctagon className="h-6 w-6 text-red-400 animate-bounce shrink-0" />
              <div>
                <p className="font-extrabold uppercase tracking-widest text-sm text-red-400">
                  EMERGENCY RED ALARM TRIGGERED
                </p>
                <p className="text-xs font-bold text-zinc-300 mt-0.5">
                  Critical Event detected: <b>{criticalThreatName.toUpperCase()}</b> on {selectedCamera?.camera_name}. Local Response unit notifications in progress.
                </p>
                {emailAlertSent && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className="flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-0.5 rounded border bg-emerald-950/60 text-emerald-400 border-emerald-800/40 uppercase tracking-wider">
                      <Mail className="h-3 w-3" />
                      EMAIL ALERT SENT
                    </span>
                    {emailRecipientCount > 0 && (
                      <span className="text-[10px] font-mono font-bold text-zinc-400">
                        Recipients: {emailRecipientCount}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
            <button
              onClick={handleAcknowledgeAlarm}
              className="bg-red-600 hover:bg-red-500 text-white font-extrabold px-4 py-2 rounded-lg text-xs tracking-wider uppercase transition shadow-md shadow-red-900/40"
            >
              ACKNOWLEDGE
            </button>
          </div>
        )}

        {/* Main Feed + Detection Panel Grid */}
        {layout === 1 ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* LEFT: Live Video Feed */}
            <div className="lg:col-span-2 space-y-4">
              <div className="relative aspect-video rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950 flex items-center justify-center shadow-2xl">
                {/* Flashing Red Border overlay for alert HUD */}
                {alarmActive && <div className="absolute inset-0 siren-overlay z-20" />}
                
                {connecting ? (
                  <div className="flex flex-col items-center space-y-3">
                    <Loader className="h-8 w-8 text-blue-500 animate-spin" />
                    <p className="text-zinc-550 uppercase tracking-widest text-xs font-semibold">Initiating video handshake...</p>
                  </div>
                ) : frameData ? (
                  <img 
                    src={frameData} 
                    alt="Live Surveillance Stream" 
                    className="w-full h-full object-cover relative z-10"
                  />
                ) : (
                  <div className="flex flex-col items-center space-y-3 text-zinc-650 uppercase tracking-widest text-xs font-semibold">
                    <Heart className="h-8 w-8 text-zinc-700 animate-pulse" />
                    <p>Awaiting video frame sequence...</p>
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT: Detection Panel */}
            <div className="glass-panel p-6 rounded-2xl border border-zinc-800 flex flex-col h-[360px] lg:h-auto min-h-[300px]">
              <div className="flex items-center justify-between border-b border-zinc-900 pb-3 mb-4 shrink-0">
                <div className="flex items-center space-x-2">
                  <Settings className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-black uppercase tracking-wider text-zinc-300">
                    Tracking Analysis
                  </h3>
                </div>
                <button
                  onClick={() => {
                    setActiveXAIEvent(null);
                    setIsXAIModalOpen(true);
                  }}
                  className="bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 text-xs px-2.5 py-1 rounded-lg font-mono font-bold flex items-center gap-1.5 transition shadow-sm"
                >
                  <Brain className="w-3.5 h-3.5 text-emerald-400" /> XAI Reason
                </button>
              </div>

              {/* FIGHT DETECTED Badge */}
              {fightDetected && fightConfidence >= 0.70 && (
                <div className="mb-3 p-3 rounded-xl border-2 border-red-500 bg-red-950/40 animate-pulse">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-full bg-red-500 animate-ping" />
                      <span className="text-xs font-black uppercase tracking-widest text-red-400">⚠ FIGHT DETECTED</span>
                    </div>
                    <span className="text-xs font-mono font-bold text-red-300">{Math.round(fightConfidence * 100)}%</span>
                  </div>
                  {fightParticipants.length > 0 && (
                    <p className="text-[10px] text-red-300/80 font-mono mt-1">
                      Participants: {fightParticipants.map(p => `#${p}`).join(', ')}
                    </p>
                  )}
                  {fightDuration > 0 && (
                    <p className="text-[10px] text-red-400/70 font-mono">
                      Duration: {fightDuration}s
                    </p>
                  )}
                </div>
              )}

              {/* ACCIDENT DETECTED Badge */}
              {accidentDetected && accidentConfidence >= 0.70 && (
                <div className="mb-3 p-3 rounded-xl border-2 border-orange-500 bg-orange-950/40 animate-pulse">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-full bg-orange-500 animate-ping" />
                      <span className="text-xs font-black uppercase tracking-widest text-orange-400">🚗 ACCIDENT DETECTED</span>
                    </div>
                    <span className="text-xs font-mono font-bold text-orange-300">{Math.round(accidentConfidence * 100)}%</span>
                  </div>
                  <p className="text-[10px] text-orange-300/80 font-mono mt-1">
                    Vehicle collision pattern detected &bull; Threat Level: HIGH
                  </p>
                </div>
              )}

              <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
                {detections.length > 0 ? (
                  detections.map((det, idx) => (
                    <div 
                      key={idx} 
                      className={`p-3 rounded-lg border flex items-center justify-between transition ${
                        det.threat_level === 'CRITICAL' ? 'bg-red-950/20 border-red-900/40 text-red-200' :
                        det.threat_level === 'HIGH' ? 'bg-amber-950/20 border-amber-800/40 text-amber-200' :
                        det.threat_level === 'MEDIUM' ? 'bg-yellow-950/20 border-yellow-800/40 text-yellow-200' :
                        'bg-zinc-900/40 border-zinc-850/80 text-zinc-300'
                      }`}
                    >
                      <div>
                        <p className="font-extrabold text-sm tracking-wide uppercase">
                          {det.track_id !== null ? `#${det.track_id} ` : ''}{det.class}
                        </p>
                        <p className="text-[10px] text-zinc-550 mt-0.5 font-mono">
                          Threat: {det.threat_level}
                        </p>
                      </div>
                      <span className="font-mono font-bold text-xs">
                        {Math.round(det.confidence * 100)}%
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="h-full flex items-center justify-center text-zinc-650 text-xs uppercase tracking-widest font-semibold text-center">
                    No active tracking targets
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* MULTI-CAMERA SPLIT GRID */
          <div className={`grid gap-4 ${
            layout === 2 ? 'grid-cols-1 md:grid-cols-2' :
            layout === 4 ? 'grid-cols-2' :
            layout === 9 ? 'grid-cols-3' :
            layout === 16 ? 'grid-cols-4' :
            layout === 25 ? 'grid-cols-5' :
            'grid-cols-8'
          }`}>
            {Array.from({ length: layout }).map((_, index) => {
              // Get camera properties
              const camObj = cameras[index % (cameras.length || 1)];
              const camCode = camObj ? camObj.camera_id : `CAM-${String(index + 1).padStart(3, '0')}`;
              const camName = camObj ? camObj.camera_name : `Cam ${index + 1}`;
              const stream = camObj ? camObj.stream_url : '0';

              return (
                <CameraGridCell
                  key={index}
                  cameraId={camCode}
                  cameraName={camName}
                  streamUrl={stream}
                  onAlertTriggered={(alert) => {
                    const alertKey = `${camCode}|${alert.event_type}`;
                    recentAlertKeysRef.current[alertKey] = Date.now();
                    const dismissed = isAlertDismissed(camCode, alert.event_type);

                    // Update global alert state and open modal if threat is in trigger levels and not latched/dismissed
                    if (!dismissed && SIREN_TRIGGER_LEVELS.includes(alert.threat_level)) {
                      activeAlertKeyRef.current = alertKey;
                      setCriticalThreatName(alert.event_type);
                      setAlertModalImage(alert.frame);
                      setAlarmActive(true);
                      setIsAlertModalOpen(true);
                      triggerAudioSiren();
                    }
                  }}
                />
              );
            })}
          </div>
        )}

        {/* BOTTOM: Camera Health & Telemetry (Integrated) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-emerald-400" />
              <h2 className="text-md font-bold tracking-wider text-zinc-300 uppercase">
                Camera Health & Real-Time Telemetry
              </h2>
            </div>
            <button
              onClick={loadHealth}
              className="bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-400 hover:text-white px-3 py-1 rounded-lg border border-zinc-800 transition"
            >
              Refresh Telemetry
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {healthList.map((cam) => (
              <div
                key={cam.id || cam.camera_id}
                className="bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 rounded-xl p-4 shadow-lg flex flex-col justify-between gap-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-[11px] font-mono text-emerald-400 font-bold">{cam.camera_id}</div>
                    <h3 className="text-sm font-bold text-white truncate max-w-[160px]">{cam.camera_name}</h3>
                  </div>
                  {getStatusBadge(cam.status, cam.freeze_detected, cam.tamper_detected)}
                </div>

                <div className="grid grid-cols-2 gap-2 bg-zinc-950/80 p-2.5 rounded-lg border border-zinc-800/80 text-xs font-mono">
                  <div>
                    <div className="text-zinc-500 text-[10px] flex items-center gap-1">
                      <Radio className="w-3 h-3 text-cyan-400" /> FPS
                    </div>
                    <div className={`text-sm font-bold ${cam.fps < 10 ? 'text-red-400 animate-pulse' : 'text-emerald-400'}`}>
                      {cam.fps.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div className="text-zinc-500 text-[10px] flex items-center gap-1">
                      <Zap className="w-3 h-3 text-yellow-400" /> Latency
                    </div>
                    <div className="text-sm font-bold text-white">
                      {cam.latency_ms} ms
                    </div>
                  </div>
                </div>

                <div className="text-[10px] font-mono text-zinc-500 flex justify-between border-t border-zinc-800/80 pt-2">
                  <span>Heartbeat:</span>
                  <span className="text-zinc-300">
                    {cam.last_ping ? new Date(cam.last_ping).toLocaleTimeString() : 'Active'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Modal: Red Emergency Alert Pop-up */}
        {isAlertModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <div className="absolute inset-0 bg-black/90 backdrop-blur-md" onClick={handleDismissModal} />
            <div className="relative w-full max-w-xl bg-zinc-950 border-2 border-red-500 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(239,68,68,0.3)] z-10 animate-in fade-in zoom-in duration-200">
              
              {/* Header banner */}
              <div className="bg-red-600 text-white px-6 py-4 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <AlertOctagon className="h-6 w-6 text-white animate-bounce shrink-0" />
                  <h3 className="text-md font-black tracking-widest uppercase">
                    CRITICAL INCIDENT ENFORCEMENT WARN
                  </h3>
                </div>
              </div>

              {/* Body */}
              <div className="p-6 space-y-4">
                <p className="text-sm font-bold text-zinc-355 leading-relaxed">
                  FastAPI threat engine reports: <b>{criticalThreatName.toUpperCase()}</b> visual detection. Auto-archiving footage stream. Verify capture below:
                </p>
                
                {/* Captured Frame display */}
                <div className="aspect-video bg-zinc-900 rounded-xl overflow-hidden border border-zinc-800 relative">
                  {alertModalImage ? (
                    <img 
                      src={alertModalImage} 
                      alt="Threat Capture" 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-zinc-650 text-xs">
                      No capture image
                    </div>
                  )}
                  <span className="absolute bottom-3 left-3 bg-black/60 backdrop-blur-md border border-zinc-800 px-3 py-1 rounded text-[10px] font-mono text-zinc-300">
                    TARGET LOC: {selectedCamera?.latitude.toFixed(4)}, {selectedCamera?.longitude.toFixed(4)}
                  </span>
                </div>

                <div className="flex items-center justify-end space-x-3 pt-2">
                  <button
                    onClick={() => {
                      handleDismissModal();
                      setIsXAIModalOpen(true);
                    }}
                    className="bg-emerald-950 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 font-bold py-3 px-4 rounded-xl tracking-wider uppercase transition flex items-center justify-center gap-2"
                  >
                    <Brain className="w-4 h-4 text-emerald-400" /> VIEW XAI REASONING
                  </button>
                  <button
                    onClick={handleAcknowledgeAlarm}
                    className="bg-red-600 hover:bg-red-500 text-white font-extrabold py-3 px-4 rounded-xl tracking-wider uppercase transition shadow-lg shadow-red-900/30"
                  >
                    DISMISS & ACKNOWLEDGE
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Modal: XAI Explanation Reasoning */}
        <XAIExplanationModal
          isOpen={isXAIModalOpen}
          onClose={() => setIsXAIModalOpen(false)}
          explanation={activeXAIExplanation}
          event={activeXAIEvent}
        />
        
      </div>
    </CommandCenterLayout>
  );
};
export default LiveMonitoring;
