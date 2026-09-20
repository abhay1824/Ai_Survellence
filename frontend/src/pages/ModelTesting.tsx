import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Upload,
  Play,
  Download,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
  Zap,
  Layers,
  Crosshair,
  Car,
  Swords,
  Eye,
  Columns,
  Image as ImageIcon,
  Check,
  Sliders
} from 'lucide-react';
import api from '../services/api';

interface ModelStatus {
  loaded: boolean;
  type: string;
  classes: string[];
  model_file: string;
}

interface AvailableSample {
  id: string;
  title: string;
  category: string;
  description: string;
  available: boolean;
}

interface StatusResponse {
  service: string;
  models: {
    fight_classifier: ModelStatus;
    weapon_detector: ModelStatus;
    accident_detector: ModelStatus;
  };
  available_samples: AvailableSample[];
}

interface DetectionBox {
  class: string;
  confidence: number;
  box: [number, number, number, number];
}

interface PredictResponse {
  success: boolean;
  inference_time_ms: number;
  image_dimensions: { width: number; height: number };
  original_image: string;
  annotated_image: string;
  fight_result: {
    top_class?: string;
    confidence?: number;
    is_fight?: boolean;
    probabilities?: Record<string, number>;
    model?: string;
    error?: string;
  } | null;
  weapon_result: {
    count: number;
    detections: DetectionBox[];
  };
  accident_result: {
    count: number;
    detections: { class: string; confidence: number; box: [number, number, number, number] }[];
  };
  threat_assessment: {
    score: number;
    level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'SAFE';
    summary: string;
  };
}

export const ModelTesting: React.FC = () => {
  // State
  const [statusData, setStatusData] = useState<StatusResponse | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(true);

  // Selected input
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('fight_sample_1');

  // Parameters
  const [modelType, setModelType] = useState<string>('all');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.35);

  // Inference state
  const [runningInference, setRunningInference] = useState<boolean>(false);
  const [inferenceResult, setInferenceResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // View state: 'annotated' | 'side-by-side' | 'original'
  const [viewMode, setViewMode] = useState<'annotated' | 'side-by-side' | 'original'>('annotated');

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load Model Status on Mount
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    setLoadingStatus(true);
    try {
      const res = await api.get<StatusResponse>('/model-test/status');
      setStatusData(res.data);
    } catch (err: any) {
      console.error('Failed to fetch model test status:', err);
    } finally {
      setLoadingStatus(false);
    }
  };

  // Handle File Upload Change
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setSelectedSampleId(''); // Clear preset if file is uploaded
      const reader = new FileReader();
      reader.onload = () => {
        setFilePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle Preset Select
  const handleSelectPreset = (sampleId: string) => {
    setSelectedSampleId(sampleId);
    setSelectedFile(null);
    setFilePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Execute Model Inference
  const handleRunInference = async () => {
    setRunningInference(true);
    setErrorMsg(null);
    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else if (selectedSampleId) {
        formData.append('sample_id', selectedSampleId);
      } else {
        setErrorMsg('Please select a preset sample or upload an image.');
        setRunningInference(false);
        return;
      }

      formData.append('model_type', modelType);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const res = await api.post<PredictResponse>('/model-test/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setInferenceResult(res.data);
    } catch (err: any) {
      console.error('Inference error:', err);
      const msg = err.response?.data?.detail || 'Failed to execute inference on the model.';
      setErrorMsg(msg);
    } finally {
      setRunningInference(false);
    }
  };

  // Download Annotated Image
  const handleDownload = () => {
    if (!inferenceResult?.annotated_image) return;
    const link = document.createElement('a');
    link.href = inferenceResult.annotated_image;
    link.download = `eagle_ai_detection_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Helper for threat level styles
  const getThreatBadge = (level: string) => {
    switch (level) {
      case 'CRITICAL':
        return 'bg-red-950/80 text-red-400 border-red-800 animate-pulse';
      case 'HIGH':
        return 'bg-amber-950/80 text-amber-400 border-amber-800';
      case 'MEDIUM':
        return 'bg-yellow-950/80 text-yellow-400 border-yellow-800';
      case 'LOW':
        return 'bg-blue-950/80 text-blue-400 border-blue-800';
      default:
        return 'bg-emerald-950/80 text-emerald-400 border-emerald-800';
    }
  };

  const presetButtons = [
    {
      id: 'fight_sample_1',
      title: 'Real Fight Scene',
      icon: Swords,
      color: 'text-red-400 border-red-900/50 bg-red-950/20 hover:border-red-600',
      badge: '🥊 Altercation',
    },
    {
      id: 'normal_scene_1',
      title: 'Normal Scene',
      icon: ShieldCheck,
      color: 'text-emerald-400 border-emerald-900/50 bg-emerald-950/20 hover:border-emerald-600',
      badge: '🛡️ Baseline',
    },
    {
      id: 'accident_sample_1',
      title: 'Traffic Accident',
      icon: Car,
      color: 'text-amber-400 border-amber-900/50 bg-amber-950/20 hover:border-amber-600',
      badge: '🚗 Vehicle Hazard',
    },
    {
      id: 'weapon_sample_1',
      title: 'Armed Weapon',
      icon: Crosshair,
      color: 'text-purple-400 border-purple-900/50 bg-purple-950/20 hover:border-purple-600',
      badge: '🗡️ Edged Threat',
    },
    {
      id: 'surveillance_cam',
      title: 'Live CCTV Frame',
      icon: Layers,
      color: 'text-cyan-400 border-cyan-900/50 bg-cyan-950/20 hover:border-cyan-600',
      badge: '📹 Stream Snap',
    },
  ];

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Top Header & Operational Telemetry */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-950/50 border border-emerald-800/80 rounded-xl text-emerald-400 shadow-lg shadow-emerald-950/30">
            <Cpu className="w-7 h-7" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-wider text-white uppercase font-mono">
                AI MODEL TESTING & VERIFICATION SANDBOX
              </h1>
              <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase">
                v2.0 ONLINE
              </span>
            </div>
            <p className="text-xs text-zinc-400">
              Interactive test bench for YOLO Fight Classifier, Weapon Detection, and Accident Recognition models
            </p>
          </div>
        </div>

        {/* Loaded Models Status Bar */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs font-mono">
            <Swords className="w-3.5 h-3.5 text-red-400" />
            <span className="text-zinc-400">Fight Classifier:</span>
            <span className={statusData?.models.fight_classifier.loaded ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
              {statusData?.models.fight_classifier.loaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs font-mono">
            <Crosshair className="w-3.5 h-3.5 text-orange-400" />
            <span className="text-zinc-400">Weapon Detector:</span>
            <span className={statusData?.models.weapon_detector.loaded ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
              {statusData?.models.weapon_detector.loaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs font-mono">
            <Car className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-zinc-400">Accident Detector:</span>
            <span className={statusData?.models.accident_detector.loaded ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
              {statusData?.models.accident_detector.loaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          <button
            onClick={fetchStatus}
            disabled={loadingStatus}
            className="p-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg border border-zinc-700 transition"
            title="Refresh Models Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingStatus ? 'animate-spin text-emerald-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Main Sandbox Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: Input Source & Controls (5 Cols) */}
        <div className="lg:col-span-4 xl:col-span-4 flex flex-col gap-5">
          {/* Section 1: Preset Sample Library */}
          <div className="bg-zinc-900/70 border border-zinc-800/90 rounded-xl p-4 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-mono font-bold tracking-wider text-zinc-300 uppercase flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-emerald-400" />
                1-Click Preset Samples
              </h2>
              <span className="text-[10px] text-zinc-500 font-mono">Ground Truth Assets</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {presetButtons.map((sample) => {
                const Icon = sample.icon;
                const isSelected = selectedSampleId === sample.id && !selectedFile;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectPreset(sample.id)}
                    className={`flex flex-col items-start p-3 rounded-lg border text-left transition-all relative overflow-hidden ${sample.color} ${
                      isSelected ? 'ring-2 ring-emerald-400 border-transparent shadow-md shadow-emerald-950/50' : 'opacity-85 hover:opacity-100'
                    }`}
                  >
                    {isSelected && (
                      <div className="absolute top-1.5 right-1.5">
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      </div>
                    )}
                    <div className="flex items-center gap-1.5 text-xs font-bold text-zinc-200">
                      <Icon className="w-3.5 h-3.5" />
                      <span>{sample.title}</span>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400 mt-1">{sample.badge}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Custom Image Upload */}
          <div className="bg-zinc-900/70 border border-zinc-800/90 rounded-xl p-4 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-mono font-bold tracking-wider text-zinc-300 uppercase flex items-center gap-1.5">
                <Upload className="w-4 h-4 text-cyan-400" />
                Custom Image Upload
              </h2>
              {selectedFile && (
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setFilePreview(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  className="text-[10px] text-red-400 hover:text-red-300 font-mono underline"
                >
                  Clear File
                </button>
              )}
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition flex flex-col items-center justify-center gap-2 ${
                selectedFile
                  ? 'border-emerald-500/80 bg-emerald-950/10'
                  : 'border-zinc-700/80 hover:border-zinc-500 bg-zinc-950/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png, image/jpeg, image/webp"
                className="hidden"
                onChange={handleFileChange}
              />
              {filePreview ? (
                <div className="relative w-full max-h-32 overflow-hidden rounded-md border border-zinc-800">
                  <img src={filePreview} alt="Preview" className="w-full h-32 object-cover" />
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center text-xs font-mono text-white">
                    Click to replace
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="w-6 h-6 text-zinc-400" />
                  <div className="text-xs font-semibold text-zinc-300">
                    Drop image here or <span className="text-emerald-400 underline">browse files</span>
                  </div>
                  <div className="text-[10px] font-mono text-zinc-500">Supports JPG, PNG, WEBP (Max 15MB)</div>
                </>
              )}
            </div>

            {selectedFile && (
              <div className="mt-2 text-[11px] font-mono text-emerald-400 flex items-center justify-between">
                <span className="truncate max-w-[200px]">📁 {selectedFile.name}</span>
                <span className="text-zinc-500">{(selectedFile.size / 1024).toFixed(0)} KB</span>
              </div>
            )}
          </div>

          {/* Section 3: Model Parameters & Pipeline Controls */}
          <div className="bg-zinc-900/70 border border-zinc-800/90 rounded-xl p-4 shadow-xl flex flex-col gap-4">
            <h2 className="text-xs font-mono font-bold tracking-wider text-zinc-300 uppercase flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-amber-400" />
              Inference Configuration
            </h2>

            {/* Target Model Selection Tabs */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-mono text-zinc-400">Target Model Filter:</label>
              <div className="grid grid-cols-2 gap-1.5 bg-zinc-950 p-1.5 rounded-lg border border-zinc-800 text-xs font-mono">
                {[
                  { id: 'all', label: 'All Models' },
                  { id: 'fight', label: 'Fight Classifier' },
                  { id: 'weapon', label: 'Weapon Detector' },
                  { id: 'accident', label: 'Accident Detector' },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setModelType(item.id)}
                    className={`px-2 py-1.5 rounded text-center transition ${
                      modelType === item.id
                        ? 'bg-emerald-950 text-emerald-400 font-bold border border-emerald-800'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Confidence Threshold Slider */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-zinc-400">Confidence Threshold:</span>
                <span className="text-emerald-400 font-bold bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/80">
                  {Math.round(confidenceThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 bg-zinc-800 rounded-lg cursor-pointer h-1.5"
              />
              <div className="flex justify-between text-[10px] font-mono text-zinc-500">
                <span>20% (Sensitive)</span>
                <span>90% (Strict)</span>
              </div>
            </div>

            {/* Action Trigger Button */}
            <button
              onClick={handleRunInference}
              disabled={runningInference}
              className={`w-full py-3 px-4 rounded-xl font-bold font-mono text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg ${
                runningInference
                  ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-600 via-emerald-500 to-cyan-500 hover:from-emerald-500 hover:to-cyan-400 text-zinc-950 shadow-emerald-950/40 hover:scale-[1.01]'
              }`}
            >
              {runningInference ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span className="text-white">Executing YOLO Inference...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Inspect Image with Model</span>
                </>
              )}
            </button>

            {errorMsg && (
              <div className="p-3 bg-red-950/80 border border-red-800 rounded-lg text-xs font-mono text-red-300 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Output Preview & Analytical Breakdown (8 Cols) */}
        <div className="lg:col-span-8 xl:col-span-8 flex flex-col gap-5">
          {inferenceResult ? (
            <>
              {/* Verdict Header Alert Banner */}
              <div
                className={`p-4 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl ${
                  inferenceResult.threat_assessment.level === 'CRITICAL'
                    ? 'bg-gradient-to-r from-red-950/90 via-red-900/60 to-zinc-950 border-red-700 text-red-200'
                    : inferenceResult.threat_assessment.level === 'HIGH'
                    ? 'bg-gradient-to-r from-amber-950/90 via-amber-900/60 to-zinc-950 border-amber-700 text-amber-200'
                    : 'bg-gradient-to-r from-emerald-950/90 via-emerald-900/60 to-zinc-950 border-emerald-700 text-emerald-200'
                }`}
              >
                <div className="flex items-center gap-3">
                  {inferenceResult.threat_assessment.level === 'CRITICAL' ? (
                    <ShieldAlert className="w-8 h-8 text-red-400 animate-bounce" />
                  ) : inferenceResult.threat_assessment.level === 'HIGH' ? (
                    <AlertTriangle className="w-8 h-8 text-amber-400" />
                  ) : (
                    <CheckCircle2 className="w-8 h-8 text-emerald-400" />
                  )}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm sm:text-base font-black tracking-wide uppercase font-mono">
                        {inferenceResult.threat_assessment.summary}
                      </span>
                    </div>
                    <p className="text-[11px] opacity-80 font-mono mt-0.5">
                      Threat Score: {inferenceResult.threat_assessment.score} / 100 • Evaluated via Weighted Multi-Model Normalization
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 self-start sm:self-center">
                  <span
                    className={`px-3 py-1 rounded-lg text-xs font-mono font-bold border uppercase tracking-wider ${getThreatBadge(
                      inferenceResult.threat_assessment.level
                    )}`}
                  >
                    {inferenceResult.threat_assessment.level} PRIORITY
                  </span>
                </div>
              </div>

              {/* View Mode Bar & Telemetry */}
              <div className="flex flex-wrap items-center justify-between gap-3 bg-zinc-900/70 border border-zinc-800 p-2.5 rounded-xl">
                {/* Mode Selector */}
                <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-mono">
                  <button
                    onClick={() => setViewMode('annotated')}
                    className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition ${
                      viewMode === 'annotated'
                        ? 'bg-emerald-950 text-emerald-400 font-bold border border-emerald-800'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>AI Annotated Output</span>
                  </button>
                  <button
                    onClick={() => setViewMode('side-by-side')}
                    className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition ${
                      viewMode === 'side-by-side'
                        ? 'bg-emerald-950 text-emerald-400 font-bold border border-emerald-800'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    <Columns className="w-3.5 h-3.5" />
                    <span>Side-by-Side</span>
                  </button>
                  <button
                    onClick={() => setViewMode('original')}
                    className={`px-3 py-1.5 rounded flex items-center gap-1.5 transition ${
                      viewMode === 'original'
                        ? 'bg-emerald-950 text-emerald-400 font-bold border border-emerald-800'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    <ImageIcon className="w-3.5 h-3.5" />
                    <span>Original Source</span>
                  </button>
                </div>

                {/* Telemetry badges & Download */}
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1 text-[11px] font-mono bg-zinc-950 px-2.5 py-1.5 rounded border border-zinc-800 text-cyan-400">
                    <Zap className="w-3 h-3" />
                    <span>{inferenceResult.inference_time_ms} ms</span>
                  </div>
                  <div className="text-[11px] font-mono bg-zinc-950 px-2.5 py-1.5 rounded border border-zinc-800 text-zinc-400">
                    {inferenceResult.image_dimensions.width}×{inferenceResult.image_dimensions.height} px
                  </div>
                  <button
                    onClick={handleDownload}
                    className="p-1.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded border border-zinc-700 transition"
                    title="Download Annotated Image"
                  >
                    <Download className="w-3.5 h-3.5 text-emerald-400" />
                  </button>
                </div>
              </div>

              {/* Visual Display Canvas Container */}
              <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden shadow-2xl relative">
                {viewMode === 'annotated' && (
                  <div className="w-full flex items-center justify-center p-2 bg-zinc-950">
                    <img
                      src={inferenceResult.annotated_image}
                      alt="Annotated Model Inference"
                      className="max-h-[500px] w-auto object-contain rounded-lg border border-zinc-800"
                    />
                  </div>
                )}

                {viewMode === 'original' && (
                  <div className="w-full flex items-center justify-center p-2 bg-zinc-950">
                    <img
                      src={inferenceResult.original_image}
                      alt="Original Raw Input"
                      className="max-h-[500px] w-auto object-contain rounded-lg border border-zinc-800"
                    />
                  </div>
                )}

                {viewMode === 'side-by-side' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 p-2 bg-zinc-950">
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider px-1">
                        Original Input
                      </span>
                      <img
                        src={inferenceResult.original_image}
                        alt="Original Input"
                        className="max-h-[380px] w-full object-contain rounded-lg border border-zinc-800"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider px-1">
                        AI Annotated Output
                      </span>
                      <img
                        src={inferenceResult.annotated_image}
                        alt="AI Annotated Output"
                        className="max-h-[380px] w-full object-contain rounded-lg border border-emerald-950/80 shadow"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Detailed Breakdown Panels */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Card 1: Fight Classification */}
                <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3">
                  <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
                    <div className="flex items-center gap-1.5">
                      <Swords className="w-4 h-4 text-red-400" />
                      <span className="text-xs font-mono font-bold uppercase text-zinc-200">Fight Classifier</span>
                    </div>
                    {inferenceResult.fight_result?.is_fight ? (
                      <span className="bg-red-950 text-red-400 border border-red-800 text-[10px] font-mono px-2 py-0.5 rounded font-bold">
                        VIOLENCE DETECTED
                      </span>
                    ) : (
                      <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-mono px-2 py-0.5 rounded font-bold">
                        NON-FIGHT NORMAL
                      </span>
                    )}
                  </div>

                  {inferenceResult.fight_result ? (
                    <div className="flex flex-col gap-2 text-xs font-mono">
                      <div className="flex justify-between text-zinc-400 text-[11px]">
                        <span>Top Class:</span>
                        <span className="text-white font-bold">{inferenceResult.fight_result.top_class}</span>
                      </div>
                      <div className="flex justify-between text-zinc-400 text-[11px]">
                        <span>Confidence:</span>
                        <span className="text-emerald-400 font-bold">
                          {((inferenceResult.fight_result.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* Probabilities Bars */}
                      {inferenceResult.fight_result.probabilities && (
                        <div className="flex flex-col gap-1.5 mt-1 pt-2 border-t border-zinc-800/80">
                          {Object.entries(inferenceResult.fight_result.probabilities).map(([cls, prob]) => {
                            const isFightClass = cls.toLowerCase() === 'fight';
                            const percentage = Math.round(prob * 100);
                            return (
                              <div key={cls} className="flex flex-col gap-0.5">
                                <div className="flex justify-between text-[10px] text-zinc-400">
                                  <span>{cls}</span>
                                  <span className="font-bold">{percentage}%</span>
                                </div>
                                <div className="w-full bg-zinc-950 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className={`h-full rounded-full transition-all ${
                                      isFightClass ? 'bg-red-500' : 'bg-emerald-500'
                                    }`}
                                    style={{ width: `${percentage}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-zinc-500 font-mono italic">Fight model skipped or disabled</div>
                  )}
                </div>

                {/* Card 2: Weapon Detection */}
                <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3">
                  <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
                    <div className="flex items-center gap-1.5">
                      <Crosshair className="w-4 h-4 text-orange-400" />
                      <span className="text-xs font-mono font-bold uppercase text-zinc-200">Weapon Detector</span>
                    </div>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                        inferenceResult.weapon_result.count > 0
                          ? 'bg-red-950 text-red-400 border-red-800 animate-pulse'
                          : 'bg-zinc-800 text-zinc-400 border-zinc-700'
                      }`}
                    >
                      {inferenceResult.weapon_result.count} FOUND
                    </span>
                  </div>

                  {inferenceResult.weapon_result.count > 0 ? (
                    <div className="flex flex-col gap-1.5 max-h-44 overflow-y-auto">
                      {inferenceResult.weapon_result.detections.map((det, idx) => (
                        <div
                          key={idx}
                          className="bg-zinc-950/80 border border-red-900/40 p-2 rounded-lg text-xs font-mono flex flex-col gap-1"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-red-400 font-bold uppercase">{det.class}</span>
                            <span className="text-white">{(det.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="text-[10px] text-zinc-500">
                            Box: [{det.box.join(', ')}]
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs text-zinc-500 font-mono py-4 text-center">
                      🛡️ Zero weapon signatures detected at &gt;{Math.round(confidenceThreshold * 100)}% confidence
                    </div>
                  )}
                </div>

                {/* Card 3: Accident Detection */}
                <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3">
                  <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
                    <div className="flex items-center gap-1.5">
                      <Car className="w-4 h-4 text-amber-400" />
                      <span className="text-xs font-mono font-bold uppercase text-zinc-200">Accident Detector</span>
                    </div>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold border ${
                        inferenceResult.accident_result.count > 0
                          ? 'bg-amber-950 text-amber-400 border-amber-800'
                          : 'bg-zinc-800 text-zinc-400 border-zinc-700'
                      }`}
                    >
                      {inferenceResult.accident_result.count} HAZARDS
                    </span>
                  </div>

                  {inferenceResult.accident_result.count > 0 ? (
                    <div className="flex flex-col gap-1.5 max-h-44 overflow-y-auto">
                      {inferenceResult.accident_result.detections.map((det, idx) => (
                        <div
                          key={idx}
                          className="bg-zinc-950/80 border border-amber-900/40 p-2 rounded-lg text-xs font-mono flex flex-col gap-1"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-amber-400 font-bold uppercase">Accident Collision</span>
                            <span className="text-white">{(det.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="text-[10px] text-zinc-500">
                            Box: [{det.box.join(', ')}]
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs text-zinc-500 font-mono py-4 text-center">
                      🚗 No traffic collisions or vehicular hazards identified
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            /* Blank state waiting for inspection */
            <div className="bg-zinc-900/40 border border-dashed border-zinc-800 rounded-xl p-12 flex flex-col items-center justify-center text-center min-h-[460px] gap-4">
              <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-2xl text-emerald-400 shadow-xl">
                <Cpu className="w-10 h-10" />
              </div>
              <div className="max-w-md">
                <h3 className="text-base font-bold text-white font-mono tracking-wide uppercase">
                  Awaiting Inference Execution
                </h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Select a 1-click ground truth preset from the library or upload a custom camera frame, then click{' '}
                  <span className="text-emerald-400 font-semibold">"Inspect Image with Model"</span> to visualize detections.
                </p>
              </div>

              <div className="flex flex-wrap items-center justify-center gap-3 mt-2">
                <div className="flex items-center gap-1 text-[11px] font-mono text-zinc-500 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800/80">
                  <Swords className="w-3.5 h-3.5 text-red-400" /> Fight Classification
                </div>
                <div className="flex items-center gap-1 text-[11px] font-mono text-zinc-500 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800/80">
                  <Crosshair className="w-3.5 h-3.5 text-orange-400" /> Bounding Boxes
                </div>
                <div className="flex items-center gap-1 text-[11px] font-mono text-zinc-500 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800/80">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" /> Threat Scoring
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelTesting;
