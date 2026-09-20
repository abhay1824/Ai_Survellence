export type ThreatLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export type AlarmStatus = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED' | 'ESCALATED';

export interface RestrictedZone {
  zone_id: string;
  camera_id: string;
  name: string;
  points: [number, number][]; // Polygon coordinates [[x, y], ...]
  threat_level: ThreatLevel;
  created_at?: string;
}

export interface Camera {
  _id?: string;
  camera_id: string;
  camera_name: string;
  latitude: number;
  longitude: number;
  stream_url: string;
  status: 'Online' | 'No Video Input' | 'Warning' | 'Critical Alert' | 'Offline' | 'Tampered' | 'Frozen';
  fps?: number;
  latency_ms?: number;
  freeze_detected?: boolean;
  tamper_detected?: boolean;
  last_ping?: string;
  created_at?: string;
}

export interface NearbyPerson {
  track_id: number;
  distance_m: number;
  inside_danger_radius: boolean;
  inside_threat_cone?: boolean;
  angle_offset_deg?: number;
}

export interface SpatialRiskAnalysis {
  nearby_persons: NearbyPerson[];
  danger_radius_m: number;
  threat_cone_occupants_count?: number;
  forward_cone_angle_deg?: number;
  line_of_sight_summary: string;
}

export interface TimelineStep {
  timestamp: string;
  event: string;
  details: string;
}

export interface ThreatFactor {
  factor: string;
  points: number;
}

export interface ThreatHistoryPoint {
  time: string;
  score: number;
}

export interface ThreatEvolution {
  score_start: number;
  score_end: number;
  delta_str: string;
  time_frame_seconds: number;
  delta_explanation: string;
  delta_factors: ThreatFactor[];
  history_points: ThreatHistoryPoint[];
}

export interface TrajectorySummary {
  direction: string;
  speed_estimate: string;
  path_history: string;
}

export interface BehavioralAnalysis {
  movement_state: 'Stationary' | 'Walking' | 'Running' | 'Chasing';
  trajectory_summary: TrajectorySummary;
  observed_actions: string;
}

export interface AlertSummary {
  threat_level: ThreatLevel;
  threat_score: number;
  camera_id: string;
  timestamp: string;
  brief_explanation: string;
}

export interface ConfidenceBreakdown {
  weapon_detection_confidence: string;
  fight_detection_confidence?: string;
  accident_detection_confidence?: string;
  tracking_confidence: string;
  threat_correlation_confidence: string;
  overall_assessment_confidence: string;
}

export interface XAIExplanation {
  alert_summary: AlertSummary;
  evidence_used: string[];
  behavioral_analysis: BehavioralAnalysis;
  spatial_risk_analysis: SpatialRiskAnalysis;
  threat_assessment: string;
  recommended_actions: string[];
  confidence_breakdown: ConfidenceBreakdown;
  timeline?: TimelineStep[];
  threat_evolution?: ThreatEvolution;
  formatted_text: string;
}

export interface SurveillanceEvent {
  _id: string;
  event_id: string;
  camera_id: string;
  event_type: string;
  threat_level: ThreatLevel;
  threat_score: number;
  confidence: number;
  latitude: number;
  longitude: number;
  person_id?: number;
  zone_id?: string;
  duration?: number;
  correlated_events?: string[];
  xai_explanation?: XAIExplanation;
  status: AlarmStatus;
  acknowledged_by?: string;
  acknowledged_at?: string;
  image_path?: string | null;
  video_path?: string | null;
  timestamp: string;
  // Fight Detection Fields
  fight_detected?: boolean;
  fight_confidence?: number;
  fight_participants?: number[];
  fight_duration_sec?: number;
  // Accident Detection Fields
  accident_detected?: boolean;
  accident_confidence?: number;
}

export interface DashboardSummary {
  total_cameras: number;
  active_cameras: number;
  today_events: number;
  critical_alerts: number;
  active_alarms_count?: number;
  total_accidents_today?: number;
  recent_events: SurveillanceEvent[];
  threat_distribution: Record<ThreatLevel, number>;
}

export interface CameraHealth {
  id: string;
  camera_id: string;
  camera_name: string;
  status: string;
  fps: number;
  latency_ms: number;
  freeze_detected: boolean;
  tamper_detected: boolean;
  last_ping: string;
}

export interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number;
  event_type?: string;
  camera_id?: string;
  threat_level?: ThreatLevel;
}
