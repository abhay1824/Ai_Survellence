/**
 * Legacy/Alternative Dashboard View
 * Note: CommandCenter.tsx is the primary live command center route ('/').
 * This component is retained for standalone analytical overview reporting.
 */
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Video, ShieldAlert, Activity, AlertCircle, RefreshCw, Layers, Car } from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import CommandCenterLayout from '../layouts/CommandCenterLayout';
import { CameraMap } from '../components/map/CameraMap';
import { eventService } from '../services/event';
import { cameraService } from '../services/camera';
import type { Camera, DashboardSummary } from '../types';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [mapCenter, setMapCenter] = useState<[number, number]>([0, 0]);

  const fetchDashboardData = async () => {
    try {
      const summaryData = await eventService.getDashboardSummary();
      const camerasList = await cameraService.getCameras();
      setSummary(summaryData);
      setCameras(camerasList);
      if (camerasList.length > 0 && camerasList[0].latitude && camerasList[0].longitude) {
        setMapCenter([camerasList[0].latitude, camerasList[0].longitude]);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000);
    return () => clearInterval(interval);
  }, []);


  const handleRefresh = () => {
    setRefreshing(true);
    fetchDashboardData();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#09090b] flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-zinc-500 uppercase tracking-widest text-xs font-semibold">
            Connecting to Command Center...
          </p>
        </div>
      </div>
    );
  }

  // Formatting chart data
  const chartData = summary ? [
    { name: 'CRITICAL', value: summary.threat_distribution.CRITICAL, color: '#ef4444' },
    { name: 'HIGH', value: summary.threat_distribution.HIGH, color: '#f59e0b' },
    { name: 'MEDIUM', value: summary.threat_distribution.MEDIUM, color: '#eab308' },
    { name: 'LOW', value: summary.threat_distribution.LOW, color: '#10b981' }
  ] : [];

  const kpis = [
    {
      title: 'Total Systems',
      value: summary?.total_cameras || 0,
      icon: Video,
      color: 'text-blue-400 border-blue-500/20 bg-blue-500/5',
      glow: 'shadow-[0_0_15px_rgba(59,130,246,0.05)]'
    },
    {
      title: 'Online Feeds',
      value: summary?.active_cameras || 0,
      icon: Activity,
      color: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5',
      glow: 'shadow-[0_0_15px_rgba(16,185,129,0.05)]'
    },
    {
      title: "Today's Incidents",
      value: summary?.today_events || 0,
      icon: ShieldAlert,
      color: 'text-amber-400 border-amber-500/20 bg-amber-500/5',
      glow: 'shadow-[0_0_15px_rgba(245,158,11,0.05)]'
    },
    {
      title: 'Critical Threat Alerts',
      value: summary?.critical_alerts || 0,
      icon: AlertCircle,
      color: summary?.critical_alerts && summary.critical_alerts > 0
        ? 'text-red-400 border-red-500/30 bg-red-500/10 animate-pulse font-extrabold'
        : 'text-zinc-400 border-zinc-800 bg-zinc-900/5',
      glow: summary?.critical_alerts && summary.critical_alerts > 0
        ? 'shadow-[0_0_20px_rgba(239,68,68,0.2)]'
        : ''
    },
    {
      title: "Today's Accidents",
      value: summary?.total_accidents_today || 0,
      icon: Car,
      color: (summary?.total_accidents_today && summary.total_accidents_today > 0)
        ? 'text-orange-400 border-orange-500/30 bg-orange-500/10 font-extrabold'
        : 'text-zinc-400 border-zinc-800 bg-zinc-900/5',
      glow: (summary?.total_accidents_today && summary.total_accidents_today > 0)
        ? 'shadow-[0_0_20px_rgba(249,115,22,0.2)]'
        : ''
    }
  ];

  return (
    <CommandCenterLayout>
      <div className="space-y-8">
        {/* Header Block */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-zinc-900 pb-5">
          <div>
            <h1 className="text-3xl font-black tracking-wider text-zinc-100 uppercase">
              Operations Center
            </h1>
            <p className="text-zinc-500 text-sm mt-1">
              Real-time threat feeds, camera tracking, and emergency operations.
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleRefresh}
              className="flex items-center space-x-2 px-3 py-2 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 rounded-lg text-sm text-zinc-400 transition"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>REFRESH</span>
            </button>
          </div>
        </div>

        {/* KPI Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {kpis.map((kpi, idx) => {
            const Icon = kpi.icon;
            return (
              <div 
                key={idx} 
                className={`glass-panel p-6 rounded-xl border flex items-center justify-between transition duration-300 ${kpi.color} ${kpi.glow}`}
              >
                <div>
                  <p className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
                    {kpi.title}
                  </p>
                  <p className="text-3xl font-black tracking-tight mt-2 text-zinc-100">
                    {kpi.value}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950/60 border border-zinc-800">
                  <Icon className="h-6 w-6" />
                </div>
              </div>
            );
          })}
        </div>

        {/* Map and Side Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* GIS Camera Map (Leaflet) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center space-x-2">
              <Layers className="h-5 w-5 text-blue-500" />
              <h2 className="text-md font-bold tracking-wider text-zinc-300 uppercase">
                Geographic Command Map
              </h2>
            </div>
            <CameraMap cameras={cameras} center={mapCenter} />
          </div>

          {/* Threat Distribution Chart (Pie) */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <ShieldAlert className="h-5 w-5 text-amber-500" />
              <h2 className="text-md font-bold tracking-wider text-zinc-300 uppercase">
                Threat Breakdown
              </h2>
            </div>
            <div className="glass-panel p-6 rounded-xl border border-zinc-800 h-[500px] flex flex-col justify-between">
              <div className="h-[300px]">
                {chartData.every(item => item.value === 0) ? (
                  <div className="h-full flex items-center justify-center text-zinc-600 text-sm uppercase tracking-wider font-semibold">
                    No active threats reported
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={chartData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={90}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }}
                        itemStyle={{ color: '#fafafa' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                {chartData.map((item, idx) => (
                  <div key={idx} className="flex items-center space-x-2.5 p-2 bg-zinc-900/40 rounded border border-zinc-800/40">
                    <span className="w-3 h-3 rounded" style={{ backgroundColor: item.color }} />
                    <div className="text-[11px] leading-tight">
                      <p className="font-bold text-zinc-400 tracking-wider">{item.name}</p>
                      <p className="text-zinc-200 font-extrabold">{item.value}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent Security Events Logs */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-md font-bold tracking-wider text-zinc-300 uppercase">
              Recent Activity Feed
            </h2>
            <button 
              onClick={() => navigate('/events')}
              className="text-xs font-bold text-blue-400 hover:text-blue-300 uppercase tracking-widest"
            >
              View Full Audit Log &rarr;
            </button>
          </div>
          <div className="glass-panel rounded-xl border border-zinc-800 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/40 text-xs font-bold text-zinc-500 uppercase tracking-widest">
                    <th className="py-4 px-6">Timestamp</th>
                    <th className="py-4 px-6">Camera ID</th>
                    <th className="py-4 px-6">Alert Category</th>
                    <th className="py-4 px-6">Threat level</th>
                    <th className="py-4 px-6">Confidence</th>
                    <th className="py-4 px-6 text-right">Verification</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900 text-sm">
                  {summary?.recent_events && summary.recent_events.length > 0 ? (
                    summary.recent_events.map((evt) => (
                      <tr 
                        key={evt._id} 
                        className={`hover:bg-zinc-900/30 transition duration-150 ${
                          evt.threat_level === 'CRITICAL' ? 'bg-red-500/5' :
                          evt.threat_level === 'HIGH' ? 'bg-amber-500/5' : ''
                        }`}
                      >
                        <td className="py-4 px-6 text-zinc-400">
                          {new Date(evt.timestamp).toLocaleString()}
                        </td>
                        <td className="py-4 px-6 font-semibold text-zinc-300">{evt.camera_id}</td>
                        <td className="py-4 px-6">
                          <span className="flex items-center font-bold text-zinc-200">
                            {evt.event_type}
                          </span>
                        </td>
                        <td className="py-4 px-6">
                          <span className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider ${
                            evt.threat_level === 'CRITICAL' ? 'bg-red-950/60 text-red-400 border-red-800/40' :
                            evt.threat_level === 'HIGH' ? 'bg-amber-950/60 text-amber-400 border-amber-800/40' :
                            evt.threat_level === 'MEDIUM' ? 'bg-yellow-950/60 text-yellow-400 border-yellow-800/40' :
                            'bg-emerald-950/60 text-emerald-400 border-emerald-800/40'
                          }`}>
                            {evt.threat_level}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-zinc-300 font-semibold">{Math.round(evt.confidence * 100)}%</td>
                        <td className="py-4 px-6 text-right">
                          <button
                            onClick={() => navigate(`/monitoring?camera=${evt.camera_id}`)}
                            className="text-xs bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-blue-400 font-bold px-3 py-1.5 rounded transition uppercase tracking-wider"
                          >
                            Live Feeds
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-zinc-650 uppercase tracking-widest text-xs font-semibold">
                        Awaiting incident feeds...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </CommandCenterLayout>
  );
};
export default Dashboard;
