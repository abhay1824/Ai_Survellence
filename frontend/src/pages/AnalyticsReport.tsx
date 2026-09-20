import React, { useEffect, useState } from 'react';
import { BarChart3, Calendar, FileText, TrendingUp, ShieldAlert, Award, Activity, Download } from 'lucide-react';
import { AreaChart, Area, BarChart, Bar, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { eventService } from '../services/event';
import api from '../services/api';

export const AnalyticsReport: React.FC = () => {
  const [chartsData, setChartsData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // PDF Generator parameters
  const [pdfStartDate, setPdfStartDate] = useState('');
  const [pdfEndDate, setPdfEndDate] = useState('');
  const [compilingPdf, setCompilingPdf] = useState(false);

  const fetchCharts = async () => {
    try {
      const data = await eventService.getAnalyticsCharts();
      setChartsData(data);
    } catch (e) {
      console.error('Failed to load analytical charts data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCharts();
  }, []);

  const handleCompilePDF = async (e: React.FormEvent) => {
    e.preventDefault();
    setCompilingPdf(true);
    try {
      const response = await api.get('/events/pdf-report', {
        params: {
          start_date: pdfStartDate || undefined,
          end_date: pdfEndDate || undefined
        },
        responseType: 'blob'
      });

      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `executive_security_intelligence_report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error('Failed to compile PDF report:', err);
    } finally {
      setCompilingPdf(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="h-8 w-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-6 min-h-screen bg-zinc-950 text-white font-sans">
      {/* Header Title Section */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-wide text-white uppercase flex items-center gap-3">
            <BarChart3 className="h-7 w-7 text-emerald-400" />
            SURVEILLANCE INTELLIGENCE & HEATMAP REPORTING
          </h1>
          <p className="text-zinc-400 text-xs mt-1">
            Data visualizations, threat score analytics, camera ranking, and executive PDF briefing generator.
          </p>
        </div>
      </div>

      {/* Report Compiler Panel (PDF Builder) */}
      <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl">
        <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-5">
          <FileText className="h-5 w-5 text-emerald-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-white">
            MODULE 14: Executive Briefing & PDF Security Report Compiler
          </h2>
        </div>

        <form onSubmit={handleCompilePDF} className="flex flex-col sm:flex-row items-end gap-4 max-w-3xl">
          <div className="w-full">
            <label className="block text-xs font-mono text-zinc-400 mb-1.5 flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" /> Start Date
            </label>
            <input
              type="date"
              value={pdfStartDate}
              onChange={(e) => setPdfStartDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg py-2 px-3 text-white text-xs focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="w-full">
            <label className="block text-xs font-mono text-zinc-400 mb-1.5 flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" /> End Date
            </label>
            <input
              type="date"
              value={pdfEndDate}
              onChange={(e) => setPdfEndDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg py-2 px-3 text-white text-xs focus:outline-none focus:border-emerald-500"
            />
          </div>

          <button
            type="submit"
            disabled={compilingPdf}
            className="w-full sm:w-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-bold py-2.5 px-6 rounded-lg text-xs tracking-wider uppercase transition shrink-0 shadow-lg flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            {compilingPdf ? 'COMPILING PDF...' : 'GENERATE PDF REPORT'}
          </button>
        </form>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Chart 1: Daily incident frequency */}
        <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl flex flex-col justify-between">
          <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-4">
            <TrendingUp className="h-4 w-4 text-emerald-400" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
              Daily Incident Frequency
            </h3>
          </div>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartsData?.events_per_day || []}>
                <defs>
                  <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" stroke="#71717a" fontSize={10} />
                <YAxis stroke="#71717a" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }} />
                <Area type="monotone" dataKey="Events" stroke="#10b981" fillOpacity={1} fill="url(#colorEvents)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Threat Severity breakdown */}
        <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl flex flex-col justify-between">
          <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-4">
            <ShieldAlert className="h-4 w-4 text-amber-400" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
              Severity Breakdown
            </h3>
          </div>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartsData?.threat_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="name" stroke="#71717a" fontSize={10} />
                <YAxis stroke="#71717a" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }} />
                <Bar dataKey="value" name="Alert Count">
                  {(chartsData?.threat_distribution || []).map((entry: any, index: number) => {
                    let color = '#10b981';
                    if (entry.name === 'CRITICAL') color = '#ef4444';
                    else if (entry.name === 'HIGH') color = '#f59e0b';
                    else if (entry.name === 'MEDIUM') color = '#eab308';
                    return <Cell key={`cell-${index}`} fill={color} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: Top offending cameras */}
        <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl flex flex-col justify-between">
          <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-4">
            <Award className="h-4 w-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
              Most Incident-Prone Cameras
            </h3>
          </div>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartsData?.camera_ranking || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis type="number" stroke="#71717a" fontSize={10} />
                <YAxis dataKey="name" type="category" stroke="#71717a" fontSize={8} width={110} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }} />
                <Bar dataKey="incidents" fill="#06b6d4" radius={[0, 4, 4, 0]} name="Incidents" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Critical Alerts */}
        <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl flex flex-col justify-between">
          <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-4">
            <Activity className="h-4 w-4 text-red-400 animate-pulse" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
              Critical Threats Trend
            </h3>
          </div>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartsData?.critical_trend || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" stroke="#71717a" fontSize={10} />
                <YAxis stroke="#71717a" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }} />
                <Legend verticalAlign="top" height={36} />
                <Line type="monotone" dataKey="Critical" stroke="#ef4444" strokeWidth={2.5} dot={{ r: 4 }} activeDot={{ r: 6 }} name="Critical Threats" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 5: Accident Trend */}
        <div className="bg-zinc-900/60 p-6 rounded-xl border border-zinc-800 shadow-xl flex flex-col justify-between md:col-span-2">
          <div className="flex items-center space-x-2 border-b border-zinc-800 pb-3 mb-4">
            <Activity className="h-4 w-4 text-orange-400" />
            <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
              Vehicle Collision & Accident Trend (7-Day Aggregate)
            </h3>
          </div>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartsData?.accident_trend || []}>
                <defs>
                  <linearGradient id="colorAccidents" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="date" stroke="#71717a" fontSize={10} />
                <YAxis stroke="#71717a" fontSize={10} allowDecimals={false} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fafafa' }} />
                <Area type="monotone" dataKey="accidents" stroke="#f97316" fillOpacity={1} fill="url(#colorAccidents)" name="Accidents" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
export default AnalyticsReport;
