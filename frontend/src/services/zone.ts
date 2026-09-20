import api from './api';
import type { RestrictedZone } from '../types/surveillance';

export const zoneService = {
  async getZonesByCamera(cameraId: string): Promise<RestrictedZone[]> {
    const response = await api.get(`/zones/camera/${cameraId}`);
    return response.data;
  },

  async createZone(data: { camera_id: string; name: string; points: [number, number][]; threat_level?: string }): Promise<RestrictedZone> {
    const response = await api.post('/zones', data);
    return response.data;
  },

  async updateZone(zoneId: string, data: Partial<RestrictedZone>): Promise<RestrictedZone> {
    const response = await api.put(`/zones/${zoneId}`, data);
    return response.data;
  },

  async deleteZone(zoneId: string): Promise<void> {
    await api.delete(`/zones/${zoneId}`);
  },
};
