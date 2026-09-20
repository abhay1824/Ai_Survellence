import api, { API_BASE_URL } from './api';
import type { DashboardSummary, SurveillanceEvent, XAIExplanation } from '../types/surveillance';

export interface EventFilters {
  page?: number;
  limit?: number;
  start_date?: string;
  end_date?: string;
  camera_id?: string;
  threat_level?: string;
  event_type?: string;
  status?: string;
  search?: string;
}

export const eventService = {
  async getEvents(filters: EventFilters = {}): Promise<{ events: SurveillanceEvent[]; total: number; page: number; limit: number }> {
    const response = await api.get('/events', { params: filters });
    return response.data;
  },

  async getEventExplanation(eventId: string): Promise<XAIExplanation> {
    const response = await api.get(`/events/${eventId}/explain`);
    return response.data;
  },

  async generateExplanation(payload: any): Promise<XAIExplanation> {
    const response = await api.post('/events/explain', payload);
    return response.data;
  },

  async acknowledgeAlarm(eventId: string): Promise<SurveillanceEvent> {
    const response = await api.post(`/events/${eventId}/acknowledge`, { user: 'Operator' });
    return response.data;
  },

  async resolveAlarm(eventId: string): Promise<SurveillanceEvent> {
    const response = await api.post(`/events/${eventId}/resolve`);
    return response.data;
  },

  async escalateAlarm(eventId: string): Promise<SurveillanceEvent> {
    const response = await api.post(`/events/${eventId}/escalate`);
    return response.data;
  },

  async getDashboardSummary(): Promise<DashboardSummary> {
    const response = await api.get('/analytics/summary');
    return response.data;
  },

  async getAnalyticsCharts(): Promise<any> {
    const response = await api.get('/analytics/charts');
    return response.data;
  },

  getExportCSVUrl(filters: { start_date?: string; end_date?: string; camera_id?: string } = {}): string {
    const token = localStorage.getItem('token') || '';
    const params = new URLSearchParams({ ...filters, token });
    return `${API_BASE_URL}/events/export?${params.toString()}`;
  },

  async downloadEventPDF(eventId: string): Promise<void> {
    const response = await api.get(`/events/${eventId}/pdf`, {
      responseType: 'blob'
    });
    const blob = new Blob([response.data], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `incident_report_${eventId}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },

  getExportPDFUrl(filters: { start_date?: string; end_date?: string } = {}): string {
    const token = localStorage.getItem('token') || '';
    const params = new URLSearchParams({ ...filters, token });
    return `${API_BASE_URL}/events/pdf-report?${params.toString()}`;
  },
};
