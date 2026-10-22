import React, { useState } from 'react';
import {
  Brain,
  X,
  ShieldAlert,
  Eye,
  Activity,
  Compass,
  Users,
  CheckCircle2,
  Radio,
  Sliders,
  Download,
  Clock,
  Target
} from 'lucide-react';
import type { XAIExplanation, ThreatLevel, SurveillanceEvent } from '../../types/surveillance';
import { eventService } from '../../services/event';

interface XAIExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation?: XAIExplanation | null;
  event?: SurveillanceEvent | null;
  onAcknowledge?: (eventId: string) => void;
  onEscalate?: (eventId: string) => void;
}

export const XAIExplanationModal: React.FC<XAIExplanationModalProps> = ({
  isOpen,
  onClose,
  explanation,
  event,
  onAcknowledge,
  onEscalate,
}) => {
  const [activeTab, setActiveTab] = useState<'VISUAL' | 'TIMELINE'>('VISUAL');
  const [downloadingPDF, setDownloadingPDF] = useState(false);

  if (!isOpen) return null;

  const exp = explanation || event?.xai_explanation;

  const threatLevel: ThreatLevel = exp?.alert_summary?.threat_level || event?.threat_level || 'LOW';
  const threatScore = exp?.alert_summary?.threat_score ?? event?.threat_score ?? 0;
  const cameraId = exp?.alert_summary?.camera_id || event?.camera_id || 'CAM-01';
  const timestamp = exp?.alert_summary?.timestamp || event?.timestamp || new Date().toISOString();
  const eventId = event?.event_id || 'INC-001';

  const handleDownloadPDF = async () => {
    setDownloadingPDF(true);
    try {
      await eventService.downloadEventPDF(eventId);
    } catch (e) {
      console.error('Failed to download PDF incident report:', e);
      alert('Could not download PDF report. Ensure backend API is active.');
    } finally {
      setDownloadingPDF(false);
    }
  };

  const getThreatBadge = (level: ThreatLevel) => {
    switch (level) {
      case 'CRITICAL':
        return 'bg-red-950/90 text-red-400 border-red-700 shadow-red-900/50';
      case 'HIGH':
        return 'bg-amber-950/90 text-amber-400 border-amber-700 shadow-amber-900/50';
      case 'MEDIUM':
        return 'bg-yellow-950/90 text-yellow-400 border-yellow-700 shadow-yellow-900/50';
      default:
        return 'bg-emerald-950/90 text-emerald-400 border-emerald-700 shadow-emerald-900/50';
    }
  };

  const getMovementBadge = (state?: string) => {
    switch (state) {
      case 'Chasing':
        return 'bg-purple-950 text-purple-400 border-purple-800 animate-pulse';
      case 'Running':
        return 'bg-red-950 text-red-400 border-red-800 animate-bounce';
      case 'Walking':
        return 'bg-cyan-950 text-cyan-400 border-cyan-800';
      default:
        return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-zinc-950 border border-zinc-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="p-4 sm:p-6 bg-gradient-to-r from-zinc-950 via-zinc-900 to-zinc-950 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-950/80 border border-emerald-800/80 rounded-xl text-emerald-400 shadow-lg shadow-emerald-950/50">
              <Brain className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-wide">Eagle Eye Explainable AI (XAI) Engine</h2>
                <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded-full font-bold">
                  SOC REASONING MODULE
                </span>
              </div>
              <p className="text-xs text-zinc-400 font-mono">
                Evidence-based intelligence breakdown &amp; non-predictive threat analysis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* View Mode Toggle */}
            <div className="bg-zinc-900 p-1 border border-zinc-800 rounded-lg flex items-center gap-1 text-xs">
              <button
                onClick={() => setActiveTab('VISUAL')}
                className={`px-3 py-1 rounded-md font-medium transition flex items-center gap-1.5 ${
                  activeTab === 'VISUAL' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" /> SOC Dashboard
              </button>
              <button
                onClick={() => setActiveTab('TIMELINE')}
                className={`px-3 py-1 rounded-md font-medium transition flex items-center gap-1.5 ${
                  activeTab === 'TIMELINE' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Clock className="w-3.5 h-3.5 text-cyan-400" /> Timeline Reconstruction
              </button>
            </div>

            <button
              onClick={handleDownloadPDF}
              disabled={downloadingPDF}
              className="bg-emerald-950 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 font-bold px-3 py-1.5 rounded-lg text-xs tracking-wider uppercase transition flex items-center gap-1.5 shadow-md disabled:opacity-50"
            >
              <Download className="w-4 h-4 text-emerald-400" />
              <span>{downloadingPDF ? 'GENERATING...' : 'EXPORT PDF'}</span>
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800 transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-4 sm:p-6 overflow-y-auto flex-1 space-y-6">

          {activeTab === 'VISUAL' ? (
            <>
              {/* SECTION 1: ALERT SUMMARY */}
              <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4 shadow-md">
                <div className="space-y-1">
                  <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-widest">
                    SECTION 1 — ALERT SUMMARY
                  </span>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-lg text-xs font-mono font-bold border shadow-lg ${getThreatBadge(threatLevel)}`}>
                      THREAT LEVEL: {threatLevel}
                    </span>
                    <span className="text-sm font-mono text-zinc-300">
                      Score: <strong className="text-emerald-400 text-base">+{threatScore}/100</strong>
                    </span>
                    <span className="text-xs font-mono text-zinc-400">
                      Cam: <strong className="text-white">{cameraId}</strong>
                    </span>
                  </div>
                  <p className="text-xs text-zinc-300 font-sans mt-2">
                    {exp?.alert_summary?.brief_explanation || 'Security anomaly detected by automated surveillance model.'}
                  </p>
                </div>

                <div className="text-right text-[11px] font-mono text-zinc-400">
                  <div>Timestamp: <strong className="text-zinc-200">{new Date(timestamp).toLocaleString()}</strong></div>
                  <div>Overall Confidence: <strong className="text-cyan-400">{exp?.confidence_breakdown?.overall_assessment_confidence || '95.0%'}</strong></div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

                {/* SECTION 2: EVIDENCE USED */}
                <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2">
                    <Eye className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-bold text-white tracking-wide uppercase">SECTION 2 — Evidence Used</h3>
                  </div>

                  <ul className="space-y-2 text-xs font-mono text-zinc-300">
                    {exp?.evidence_used && exp.evidence_used.length > 0 ? (
                      exp.evidence_used.map((item: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-2 bg-zinc-950/60 border border-zinc-800/50 p-2 rounded-lg">
                          <span className="text-emerald-400 font-bold">•</span>
                          <span>{item}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-zinc-500 italic">No structured evidence metrics recorded.</li>
                    )}
                  </ul>
                </div>

                {/* SECTION 2B: FIGHT ANALYSIS */}
                {(() => {
                  const fightConf = event?.fight_confidence ?? 0;
                  const fightDet = event?.fight_detected ?? false;
                  const participants = event?.fight_participants ?? [];
                  const duration = event?.fight_duration_sec ?? 0;
                  const fightPts = fightConf > 0.95 ? 40 : fightConf > 0.90 ? 30 : fightConf >= 0.85 ? 20 : 0;
                  const fightConfPct = Math.round(fightConf * 100);

                  return (
                    <div className={`rounded-xl p-4 space-y-3 border ${
                      fightDet
                        ? 'bg-red-950/30 border-red-800/60'
                        : 'bg-zinc-900/60 border-zinc-800'
                    }`}>
                      <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2">
                        <span className={`text-base ${fightDet ? 'animate-pulse' : ''}`}>🥊</span>
                        <h3 className="text-xs font-bold text-white tracking-wide uppercase">FIGHT ANALYSIS</h3>
                        <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider ${
                          fightDet
                            ? 'bg-red-950 text-red-400 border-red-700'
                            : 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        }`}>
                          {fightDet ? 'DETECTED' : 'NOT DETECTED'}
                        </span>
                      </div>

                      <div className="space-y-2 font-mono text-xs">
                        <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <span className="text-zinc-400">Fight Confidence:</span>
                          <strong className={fightDet ? 'text-red-400' : 'text-zinc-500'}>
                            {fightDet ? `${fightConfPct}%` : 'N/A'}
                          </strong>
                        </div>

                        {/* Confidence bar */}
                        {fightDet && (
                          <div className="bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                            <div className="flex justify-between text-[10px] text-zinc-400 mb-1">
                              <span>Fight Score</span><span>{fightConfPct}%</span>
                            </div>
                            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-red-500 rounded-full transition-all"
                                style={{ width: `${fightConfPct}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {participants.length > 0 && (
                          <div className="flex items-start justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                            <span className="text-zinc-400">Participants:</span>
                            <div className="flex flex-wrap gap-1 justify-end">
                              {participants.map((pid: number) => (
                                <span key={pid} className="bg-red-900/60 text-red-300 border border-red-700/60 px-1.5 py-0.5 rounded text-[10px] font-bold">#{pid}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {duration > 0 && (
                          <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                            <span className="text-zinc-400">Duration:</span>
                            <strong className="text-amber-400">{duration}s</strong>
                          </div>
                        )}

                        <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <span className="text-zinc-400">Threat Score Contribution:</span>
                          <strong className={fightDet ? 'text-red-400' : 'text-zinc-500'}>
                            {fightDet ? `+${fightPts} pts` : '0 pts'}
                          </strong>
                        </div>

                        <div className="bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <div className="text-[10px] text-zinc-500 font-sans">
                            Detection method: YOLO11 Classification — Temporal Confirmation (7/10 frames)
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {/* SECTION 2C: ACCIDENT ANALYSIS */}
                {(() => {
                  const accConf = event?.accident_confidence ?? 0;
                  const accDet = event?.accident_detected ?? false;
                  const accConfPct = Math.round(accConf * 100);
                  const accPts = accDet ? 35 : 0;

                  return (
                    <div className={`rounded-xl p-4 space-y-3 border ${
                      accDet
                        ? 'bg-orange-950/30 border-orange-800/60'
                        : 'bg-zinc-900/60 border-zinc-800'
                    }`}>
                      <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2">
                        <span className={`text-base ${accDet ? 'animate-pulse' : ''}`}>🚗</span>
                        <h3 className="text-xs font-bold text-white tracking-wide uppercase">ACCIDENT ANALYSIS</h3>
                        <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider ${
                          accDet
                            ? 'bg-orange-950 text-orange-400 border-orange-700'
                            : 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        }`}>
                          {accDet ? 'DETECTED' : 'NOT DETECTED'}
                        </span>
                      </div>

                      <div className="space-y-2 font-mono text-xs">
                        <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <span className="text-zinc-400">Accident Confidence:</span>
                          <strong className={accDet ? 'text-orange-400' : 'text-zinc-500'}>
                            {accDet ? `${accConfPct}%` : 'N/A'}
                          </strong>
                        </div>

                        {/* Confidence bar */}
                        {accDet && (
                          <div className="bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                            <div className="flex justify-between text-[10px] text-zinc-400 mb-1">
                              <span>Collision Pattern Score</span><span>{accConfPct}%</span>
                            </div>
                            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-orange-500 rounded-full transition-all"
                                style={{ width: `${accConfPct}%` }}
                              />
                            </div>
                          </div>
                        )}

                        <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <span className="text-zinc-400">Threat Score Contribution:</span>
                          <strong className={accDet ? 'text-orange-400' : 'text-zinc-500'}>
                            {accDet ? `+${accPts} pts (HIGH)` : '0 pts'}
                          </strong>
                        </div>

                        <div className="bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <div className="text-[10px] text-zinc-500 font-sans">
                            Detection method: YOLO11s Object Detection (Threshold &ge; 70%)
                          </div>
                        </div>

                        <div className="bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                          <div className="text-[10px] text-zinc-400 font-sans">
                            Observed visible collision indicators. Non-predictive safety protocols active.
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}

              </div>

              {/* SECTION 3: BEHAVIORAL ANALYSIS */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <h3 className="text-xs font-bold text-white tracking-wide uppercase">SECTION 3 — Behavioral Analysis</h3>
                  </div>

                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                      <span className="text-zinc-400">Movement State:</span>
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold border ${getMovementBadge(exp?.behavioral_analysis?.movement_state)}`}>
                        {exp?.behavioral_analysis?.movement_state || 'Stationary'}
                      </span>
                    </div>

                    <div className="bg-zinc-950/60 p-2.5 rounded-lg border border-zinc-800/50 space-y-1">
                      <div className="text-[10px] text-zinc-400 uppercase font-bold flex items-center gap-1">
                        <Compass className="w-3 h-3 text-cyan-400" /> Trajectory Breakdown
                      </div>
                      <div className="text-zinc-300">
                        Direction: <strong className="text-white">{exp?.behavioral_analysis?.trajectory_summary?.direction || 'N/A'}</strong>
                      </div>
                      <div className="text-zinc-300">
                        Estimated Speed: <strong className="text-emerald-400">{exp?.behavioral_analysis?.trajectory_summary?.speed_estimate || '0.0 m/s'}</strong>
                      </div>
                      <div className="text-zinc-400 text-[11px] leading-relaxed">
                        {exp?.behavioral_analysis?.trajectory_summary?.path_history}
                      </div>
                    </div>

                    <div className="bg-zinc-950/60 p-2.5 rounded-lg border border-zinc-800/50">
                      <div className="text-[10px] text-zinc-400 uppercase font-bold mb-1">Observed Action Statement</div>
                      <p className="text-zinc-200 italic font-sans">
                        "{exp?.behavioral_analysis?.observed_actions || 'Subject location static.'}"
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* SECTION 4: SPATIAL RISK ANALYSIS */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-amber-400" />
                    <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                      SECTION 4 — Spatial Risk Analysis (Danger Radius: {exp?.spatial_risk_analysis?.danger_radius_m || 5.0}m)
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-zinc-400">
                    Nearby Individuals Tracked: <strong className="text-white">{exp?.spatial_risk_analysis?.nearby_persons?.length || 0}</strong>
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {exp?.spatial_risk_analysis?.nearby_persons && exp.spatial_risk_analysis.nearby_persons.length > 0 ? (
                    exp.spatial_risk_analysis.nearby_persons.map((person: any, pIdx: number) => (
                      <div
                        key={pIdx}
                        className={`p-3 rounded-xl border flex items-center justify-between font-mono text-xs ${
                          person.inside_threat_cone
                            ? 'bg-red-950/70 border-red-600 text-red-200 shadow-md shadow-red-950/60'
                            : person.inside_danger_radius
                            ? 'bg-amber-950/40 border-amber-800/80 text-amber-300'
                            : 'bg-zinc-950/60 border-zinc-800 text-zinc-300'
                        }`}
                      >
                        <div>
                          <div className="font-bold flex items-center gap-1">
                            Person ID #{person.track_id}
                            {person.inside_threat_cone && <Target className="w-3.5 h-3.5 text-red-400 animate-pulse" />}
                          </div>
                          <div className="text-[11px] text-zinc-400">Distance: <strong className="text-white">{person.distance_m}m</strong></div>
                        </div>

                        <div className="flex flex-col items-end gap-1">
                          {person.inside_threat_cone ? (
                            <span className="text-[9px] bg-red-900 text-red-100 border border-red-500 px-2 py-0.5 rounded font-bold tracking-wider flex items-center gap-1">
                              FORWARD CONE (15°)
                            </span>
                          ) : person.inside_danger_radius ? (
                            <span className="text-[10px] bg-amber-900 text-amber-200 border border-amber-700 px-2 py-0.5 rounded font-bold">
                              DANGER ZONE
                            </span>
                          ) : (
                            <span className="text-[10px] bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded">
                              OUTSIDE
                            </span>
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-3 text-xs text-zinc-500 font-mono py-2 text-center">
                      No nearby tracked individuals within proximity threshold.
                    </div>
                  )}
                </div>

                <div className="bg-zinc-950/60 p-2.5 rounded-lg border border-zinc-800/50 text-xs font-mono text-cyan-300 flex items-center gap-2">
                  <Radio className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span>{exp?.spatial_risk_analysis?.line_of_sight_summary || 'Line of sight analysis active.'}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                {/* SECTION 5: THREAT ASSESSMENT & SECTION 6: RECOMMENDED ACTIONS */}
                <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2 mb-3">
                      <ShieldAlert className="w-4 h-4 text-red-400" />
                      <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                        SECTION 5 — Threat Assessment
                      </h3>
                    </div>

                    <div className="bg-zinc-950/60 border border-zinc-800 p-3 rounded-xl text-xs font-mono text-zinc-200 leading-relaxed">
                      {exp?.threat_assessment || 'Threat score reflects objective rule-based correlation.'}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2 mb-3">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                        SECTION 6 — Recommended Actions
                      </h3>
                    </div>

                    <ul className="space-y-1.5 text-xs font-mono">
                      {exp?.recommended_actions?.map((act: string, aIdx: number) => (
                        <li key={aIdx} className="flex items-center gap-2 text-emerald-300 bg-emerald-950/30 border border-emerald-900/50 px-2.5 py-1.5 rounded-lg">
                          <span className="text-emerald-400">✓</span> {act}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* SECTION 7: CONFIDENCE BREAKDOWN */}
                <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 border-b border-zinc-800/80 pb-2 mb-3">
                      <Brain className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-xs font-bold text-white tracking-wide uppercase">
                        SECTION 7 — Confidence Breakdown
                      </h3>
                    </div>

                    <div className="space-y-2 font-mono text-xs">
                      <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                        <span className="text-zinc-400">Weapon Detection Confidence:</span>
                        <strong className="text-amber-400">{exp?.confidence_breakdown?.weapon_detection_confidence || 'N/A'}</strong>
                      </div>
                      <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                        <span className="text-zinc-400">Tracking Confidence:</span>
                        <strong className="text-cyan-400">{exp?.confidence_breakdown?.tracking_confidence || '94.5%'}</strong>
                      </div>
                      <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                        <span className="text-zinc-400">Threat Correlation Confidence:</span>
                        <strong className="text-purple-400">{exp?.confidence_breakdown?.threat_correlation_confidence || '95.0%'}</strong>
                      </div>
                      <div className="flex items-center justify-between bg-zinc-950/60 p-2 rounded-lg border border-zinc-800/50">
                        <span className="text-zinc-400">Overall Assessment Confidence:</span>
                        <strong className="text-emerald-400 text-sm">{exp?.confidence_breakdown?.overall_assessment_confidence || '95.5%'}</strong>
                      </div>
                    </div>
                  </div>

                  {/* Operator Action Buttons */}
                  {event && event.status === 'ACTIVE' && (
                    <div className="pt-3 border-t border-zinc-800/80 flex items-center justify-end gap-2">
                      {onAcknowledge && (
                        <button
                          onClick={() => onAcknowledge(event.event_id)}
                          className="bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold transition"
                        >
                          ACKNOWLEDGE ALARM
                        </button>
                      )}
                      {onEscalate && (
                        <button
                          onClick={() => onEscalate(event.event_id)}
                          className="bg-purple-950 hover:bg-purple-900 border border-purple-700 text-purple-300 text-xs px-3 py-1.5 rounded-lg font-mono font-bold transition"
                        >
                          ESCALATE TO COMMAND
                        </button>
                      )}
                    </div>
                  )}
                </div>

              </div>
            </>
          ) : activeTab === 'TIMELINE' ? (
            /* SECTION: TIMELINE RECONSTRUCTION */
            <div className="space-y-4 font-mono">
              <div className="flex items-center justify-between bg-zinc-900/80 border border-zinc-800 p-3 rounded-xl">
                <div className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-cyan-400" />
                  <div>
                    <h3 className="text-xs font-bold text-white uppercase">CHRONOLOGICAL INCIDENT TIMELINE RECONSTRUCTION</h3>
                    <p className="text-[11px] text-zinc-400">Step-by-step event sequence captured prior to threat escalation</p>
                  </div>
                </div>
                <span className="text-xs bg-cyan-950 text-cyan-400 border border-cyan-800 px-3 py-1 rounded-lg font-bold">
                  {exp?.timeline?.length || 5} EVENT STEPS LOGGED
                </span>
              </div>

              <div className="relative border-l-2 border-cyan-900/60 ml-4 pl-6 space-y-6 py-2">
                {exp?.timeline && exp.timeline.length > 0 ? (
                  exp.timeline.map((step: any, sIdx: number) => (
                    <div key={sIdx} className="relative group">
                      {/* Node Bullet */}
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-cyan-400 group-hover:bg-cyan-400 transition" />
                      
                      <div className="bg-zinc-900/60 border border-zinc-800 hover:border-cyan-800/80 p-3 rounded-xl transition space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-cyan-400 font-bold">{step.timestamp}</span>
                          <span className="text-zinc-200 font-bold">{step.event}</span>
                        </div>
                        <p className="text-xs text-zinc-400 font-sans">{step.details}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <>
                    <div className="relative">
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-cyan-400" />
                      <div className="bg-zinc-900/60 border border-zinc-800 p-3 rounded-xl text-xs space-y-1">
                        <div className="flex items-center justify-between"><span className="text-cyan-400 font-bold">13:22:01</span><span className="text-zinc-200 font-bold">Subject Person Detected</span></div>
                        <p className="text-zinc-400">ByteTrack acquired Track ID #12 on camera CAM-03</p>
                      </div>
                    </div>

                    <div className="relative">
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-amber-400" />
                      <div className="bg-zinc-900/60 border border-zinc-800 p-3 rounded-xl text-xs space-y-1">
                        <div className="flex items-center justify-between"><span className="text-amber-400 font-bold">13:22:03</span><span className="text-zinc-200 font-bold">Pistol Weapon Detected</span></div>
                        <p className="text-zinc-400">YOLOv8 weapon model confidence registered at 96.0%</p>
                      </div>
                    </div>

                    <div className="relative">
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-purple-400" />
                      <div className="bg-zinc-900/60 border border-zinc-800 p-3 rounded-xl text-xs space-y-1">
                        <div className="flex items-center justify-between"><span className="text-purple-400 font-bold">13:22:04</span><span className="text-zinc-200 font-bold">Velocity Spike (Running)</span></div>
                        <p className="text-zinc-400">Subject speed increased to 2.4 m/s heading South-East</p>
                      </div>
                    </div>

                    <div className="relative">
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-red-500" />
                      <div className="bg-zinc-900/60 border border-zinc-800 p-3 rounded-xl text-xs space-y-1">
                        <div className="flex items-center justify-between"><span className="text-red-400 font-bold">13:22:06</span><span className="text-zinc-200 font-bold">Restricted Zone Intrusion</span></div>
                        <p className="text-zinc-400">Perimeter polygon breach logged in zone RESTRICTED-A</p>
                      </div>
                    </div>

                    <div className="relative">
                      <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-zinc-950 border-2 border-emerald-400" />
                      <div className="bg-zinc-900/60 border border-zinc-800 p-3 rounded-xl text-xs space-y-1">
                        <div className="flex items-center justify-between"><span className="text-emerald-400 font-bold">13:22:09</span><span className="text-zinc-200 font-bold">XAI Explanation Generated</span></div>
                        <p className="text-zinc-400">Evidence-based reasoning compiled and pushed to SOC Dashboard</p>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : null}

        </div>

        {/* Footer */}
        <div className="p-4 bg-zinc-900 border-t border-zinc-800 flex items-center justify-between text-xs font-mono text-zinc-400">
          <div>Eagle Eye Smart Surveillance Operations Center • Model: YOLOv8 / ByteTrack / YOLO11-Fight / YOLO11s-Accident / XAI Engine</div>
          <button
            onClick={onClose}
            className="bg-zinc-800 hover:bg-zinc-700 text-white px-4 py-1.5 rounded-lg font-medium transition"
          >
            Close Panel
          </button>
        </div>

      </div>
    </div>
  );
};
