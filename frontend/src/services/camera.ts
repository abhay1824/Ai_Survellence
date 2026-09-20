import api from './api';
import type { Camera, CameraHealth } from '../types/surveillance';

export const cameraService = {
  async getCameras(): Promise<Camera[]> {
    const response = await api.get('/cameras');
    return response.data;
  },

  async getCameraHealth(): Promise<CameraHealth[]> {
    const response = await api.get('/cameras/health');
    return response.data;
  },

  async getCameraById(id: string): Promise<Camera> {
    const response = await api.get(`/cameras/${id}`);
    return response.data;
  },

  async createCamera(camera: Partial<Camera>): Promise<Camera> {
    const response = await api.post('/cameras', camera);
    return response.data;
  },

  async updateCamera(id: string, camera: Partial<Camera>): Promise<Camera> {
    const response = await api.put(`/cameras/${id}`, camera);
    return response.data;
  },

  async deleteCamera(id: string): Promise<void> {
    await api.delete(`/cameras/${id}`);
  },
};
