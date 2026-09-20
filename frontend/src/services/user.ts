import api from './api';
import type { EmailSettings, EmailNotificationResult } from '../types';

export const userService = {
  async getEmailSettings(): Promise<EmailSettings> {
    const response = await api.get('/user/email-settings');
    return response.data;
  },

  async updateEmailSettings(settings: Partial<EmailSettings>): Promise<EmailSettings> {
    const response = await api.put('/user/email-settings', settings);
    return response.data;
  },

  async addRecipient(email: string): Promise<EmailSettings> {
    const response = await api.post('/user/email-settings/add-recipient', { email });
    return response.data;
  },

  async removeRecipient(email: string): Promise<EmailSettings> {
    const response = await api.delete('/user/email-settings/remove-recipient', {
      data: { email },
    });
    return response.data;
  },

  async sendTestEmail(recipient?: string): Promise<EmailNotificationResult> {
    const response = await api.post('/user/email-settings/test', {
      recipient: recipient || null,
    });
    return response.data;
  },
};
