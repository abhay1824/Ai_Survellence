import api from './api';

export const authService = {
  async login(username: string, password: string): Promise<any> {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);
    
    const response = await api.post('/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },

  async getMe(): Promise<any> {
    const response = await api.get('/auth/me');
    return response.data;
  },
};
