export * from './surveillance';

export interface User {
  username: string;
  role: 'Super Admin' | 'Police Admin' | 'Control Room Operator' | 'Viewer';
  token?: string;
}

export interface EmailSettings {
  email: string;
  notification_emails: string[];
  email_notifications_enabled: boolean;
  critical_alerts_enabled: boolean;
  daily_report_enabled: boolean;
}

export interface EmailNotificationResult {
  success: boolean;
  recipients_count: number;
  message: string;
}
