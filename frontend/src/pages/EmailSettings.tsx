import React, { useEffect, useState } from 'react';
import { Mail, Plus, Trash2, Save, Send, Bell, BellOff, AlertTriangle, FileText, Shield, CheckCircle, XCircle, Loader } from 'lucide-react';
import CommandCenterLayout from '../layouts/CommandCenterLayout';
import { userService } from '../services/user';
import type { EmailSettings as EmailSettingsType } from '../types';

export const EmailSettings: React.FC = () => {
  const [settings, setSettings] = useState<EmailSettingsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [newRecipient, setNewRecipient] = useState('');
  const [primaryEmail, setPrimaryEmail] = useState('');
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadSettings = async () => {
    try {
      const data = await userService.getEmailSettings();
      setSettings(data);
      setPrimaryEmail(data.email || '');
    } catch (err) {
      console.error('Failed to load email settings:', err);
      setStatusMessage({ type: 'error', text: 'Failed to load email settings' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, []);

  const showStatus = (type: 'success' | 'error', text: string) => {
    setStatusMessage({ type, text });
    setTimeout(() => setStatusMessage(null), 5000);
  };

  const handleSaveSettings = async () => {
    if (!settings) return;
    setSaving(true);
    try {
      const updated = await userService.updateEmailSettings({
        email: primaryEmail,
        email_notifications_enabled: settings.email_notifications_enabled,
        critical_alerts_enabled: settings.critical_alerts_enabled,
        daily_report_enabled: settings.daily_report_enabled,
      });
      setSettings(updated);
      setPrimaryEmail(updated.email);
      showStatus('success', 'Email settings saved successfully');
    } catch (err: any) {
      showStatus('error', err?.response?.data?.detail || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleAddRecipient = async () => {
    const email = newRecipient.trim();
    if (!email) return;

    // Basic email validation
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      showStatus('error', 'Please enter a valid email address');
      return;
    }

    try {
      const updated = await userService.addRecipient(email);
      setSettings(updated);
      setNewRecipient('');
      showStatus('success', `Added ${email} to notification recipients`);
    } catch (err: any) {
      showStatus('error', err?.response?.data?.detail || 'Failed to add recipient');
    }
  };

  const handleRemoveRecipient = async (email: string) => {
    try {
      const updated = await userService.removeRecipient(email);
      setSettings(updated);
      showStatus('success', `Removed ${email} from notification recipients`);
    } catch (err: any) {
      showStatus('error', err?.response?.data?.detail || 'Failed to remove recipient');
    }
  };

  const handleTestEmail = async () => {
    setTesting(true);
    try {
      const result = await userService.sendTestEmail();
      if (result.success) {
        showStatus('success', `Test email sent to ${result.recipients_count} recipient(s)`);
      } else {
        showStatus('error', result.message || 'Test email failed. Check SMTP configuration.');
      }
    } catch (err: any) {
      showStatus('error', err?.response?.data?.detail || 'Failed to send test email');
    } finally {
      setTesting(false);
    }
  };

  const handleToggle = (field: keyof EmailSettingsType) => {
    if (!settings) return;
    setSettings({ ...settings, [field]: !settings[field] });
  };

  if (loading) {
    return (
      <CommandCenterLayout>
        <div className="flex items-center justify-center h-96">
          <div className="h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </CommandCenterLayout>
    );
  }

  return (
    <CommandCenterLayout>
      <div className="space-y-6 max-w-4xl mx-auto">

        {/* Header */}
        <div className="border-b border-zinc-900 pb-5">
          <h1 className="text-3xl font-black tracking-wider text-zinc-100 uppercase flex items-center gap-3">
            <Mail className="h-8 w-8 text-blue-500" />
            Email Notification Settings
          </h1>
          <p className="text-zinc-500 text-sm mt-1">
            Configure email recipients and notification preferences for security alerts, XAI reports, and PDF incident reports.
          </p>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div className={`flex items-center gap-3 p-4 rounded-xl border ${
            statusMessage.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-800/40 text-emerald-400'
              : 'bg-red-950/40 border-red-800/40 text-red-400'
          }`}>
            {statusMessage.type === 'success'
              ? <CheckCircle className="h-5 w-5 shrink-0" />
              : <XCircle className="h-5 w-5 shrink-0" />
            }
            <span className="text-sm font-bold">{statusMessage.text}</span>
          </div>
        )}

        {/* Primary Email Section */}
        <div className="glass-panel p-6 rounded-xl border border-zinc-800">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="h-5 w-5 text-blue-400" />
            <h2 className="text-sm font-black uppercase tracking-wider text-zinc-300">Primary Email</h2>
          </div>
          <p className="text-xs text-zinc-500 mb-4">
            Your primary email address for login-associated notifications. This is the default alert recipient.
          </p>
          <div className="flex gap-3">
            <input
              type="email"
              value={primaryEmail}
              onChange={(e) => setPrimaryEmail(e.target.value)}
              placeholder="admin@company.com"
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg py-2.5 px-4 text-zinc-200 text-sm focus:outline-none focus:border-blue-500 transition"
            />
          </div>
        </div>

        {/* Additional Recipients Section */}
        <div className="glass-panel p-6 rounded-xl border border-zinc-800">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-amber-400" />
              <h2 className="text-sm font-black uppercase tracking-wider text-zinc-300">Additional Recipients</h2>
            </div>
            <span className="text-xs font-bold text-zinc-500 bg-zinc-900 px-2.5 py-1 rounded-lg border border-zinc-800">
              {settings?.notification_emails?.length || 0} configured
            </span>
          </div>
          <p className="text-xs text-zinc-500 mb-4">
            Security alerts, XAI reports, and PDF incident reports will be sent to all listed recipients.
          </p>

          {/* Recipient List */}
          <div className="space-y-2 mb-4">
            {settings?.notification_emails && settings.notification_emails.length > 0 ? (
              settings.notification_emails.map((email, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-zinc-900/60 border border-zinc-850 rounded-lg py-2.5 px-4 group hover:border-zinc-700 transition"
                >
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-emerald-500" />
                    <span className="text-sm text-zinc-300 font-mono">{email}</span>
                  </div>
                  <button
                    onClick={() => handleRemoveRecipient(email)}
                    className="text-zinc-600 hover:text-red-400 transition p-1.5 rounded-lg hover:bg-red-950/30"
                    title={`Remove ${email}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-zinc-650 text-xs uppercase tracking-widest font-semibold border border-dashed border-zinc-800 rounded-xl">
                No additional recipients configured
              </div>
            )}
          </div>

          {/* Add Recipient */}
          <div className="flex gap-3">
            <input
              type="email"
              value={newRecipient}
              onChange={(e) => setNewRecipient(e.target.value)}
              placeholder="security@company.com"
              onKeyDown={(e) => { if (e.key === 'Enter') handleAddRecipient(); }}
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg py-2.5 px-4 text-zinc-200 text-sm focus:outline-none focus:border-blue-500 transition"
            />
            <button
              onClick={handleAddRecipient}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-bold py-2.5 px-5 rounded-lg text-sm tracking-wider uppercase transition shadow-lg"
            >
              <Plus className="h-4 w-4" />
              Add
            </button>
          </div>
        </div>

        {/* Notification Preferences */}
        <div className="glass-panel p-6 rounded-xl border border-zinc-800">
          <div className="flex items-center gap-2 mb-4">
            <Bell className="h-5 w-5 text-emerald-400" />
            <h2 className="text-sm font-black uppercase tracking-wider text-zinc-300">Notification Preferences</h2>
          </div>

          <div className="space-y-4">
            {/* Email Notifications Toggle */}
            <div className="flex items-center justify-between py-3 border-b border-zinc-900">
              <div className="flex items-center gap-3">
                {settings?.email_notifications_enabled
                  ? <Bell className="h-4 w-4 text-emerald-400" />
                  : <BellOff className="h-4 w-4 text-zinc-600" />
                }
                <div>
                  <p className="text-sm font-bold text-zinc-300">Email Notifications</p>
                  <p className="text-xs text-zinc-550">Enable or disable all email notifications</p>
                </div>
              </div>
              <button
                onClick={() => handleToggle('email_notifications_enabled')}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings?.email_notifications_enabled ? 'bg-emerald-600' : 'bg-zinc-700'
                }`}
              >
                <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  settings?.email_notifications_enabled ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>

            {/* Critical Alerts Toggle */}
            <div className="flex items-center justify-between py-3 border-b border-zinc-900">
              <div className="flex items-center gap-3">
                <AlertTriangle className={`h-4 w-4 ${settings?.critical_alerts_enabled ? 'text-red-400' : 'text-zinc-600'}`} />
                <div>
                  <p className="text-sm font-bold text-zinc-300">Critical Alerts Only</p>
                  <p className="text-xs text-zinc-550">Receive emails only for CRITICAL threat level events</p>
                </div>
              </div>
              <button
                onClick={() => handleToggle('critical_alerts_enabled')}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings?.critical_alerts_enabled ? 'bg-red-600' : 'bg-zinc-700'
                }`}
              >
                <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  settings?.critical_alerts_enabled ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>

            {/* Daily Report Toggle */}
            <div className="flex items-center justify-between py-3">
              <div className="flex items-center gap-3">
                <FileText className={`h-4 w-4 ${settings?.daily_report_enabled ? 'text-blue-400' : 'text-zinc-600'}`} />
                <div>
                  <p className="text-sm font-bold text-zinc-300">Daily Summary Report</p>
                  <p className="text-xs text-zinc-550">Receive a daily PDF summary of all security events</p>
                </div>
              </div>
              <button
                onClick={() => handleToggle('daily_report_enabled')}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  settings?.daily_report_enabled ? 'bg-blue-600' : 'bg-zinc-700'
                }`}
              >
                <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  settings?.daily_report_enabled ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <button
            onClick={handleTestEmail}
            disabled={testing || (!primaryEmail && (!settings?.notification_emails || settings.notification_emails.length === 0))}
            className="flex items-center gap-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 font-bold py-3 px-6 rounded-xl text-sm tracking-wider uppercase transition shadow-lg disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {testing ? (
              <>
                <Loader className="h-4 w-4 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Test Email
              </>
            )}
          </button>

          <button
            onClick={handleSaveSettings}
            disabled={saving}
            className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 px-8 rounded-xl text-sm tracking-wider uppercase transition shadow-lg shadow-emerald-900/30 disabled:opacity-60"
          >
            {saving ? (
              <>
                <Loader className="h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4" />
                Save Settings
              </>
            )}
          </button>
        </div>

        {/* Info Card */}
        <div className="glass-panel p-5 rounded-xl border border-zinc-800 bg-zinc-900/30">
          <h3 className="text-xs font-black uppercase tracking-wider text-zinc-400 mb-3">How Email Notifications Work</h3>
          <ul className="space-y-2 text-xs text-zinc-500">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">1.</span>
              When a <strong className="text-zinc-400">HIGH</strong> or <strong className="text-red-400">CRITICAL</strong> threat is detected (weapon, fight, accident, zone intrusion), all configured recipients receive an email alert.
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">2.</span>
              XAI analysis summaries are included in alert emails with evidence-backed reasoning.
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">3.</span>
              PDF incident reports are automatically attached when generated for HIGH/CRITICAL events.
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-0.5">4.</span>
              Use the <strong className="text-zinc-400">Test Email</strong> button to verify delivery to all configured addresses.
            </li>
          </ul>
        </div>
      </div>
    </CommandCenterLayout>
  );
};

export default EmailSettings;
