import React, { useEffect, useState } from 'react';
import { Plus, Edit2, Trash2, Video, Sliders, Radio, AlertTriangle } from 'lucide-react';
import CommandCenterLayout from '../layouts/CommandCenterLayout';
import { cameraService } from '../services/camera';
import type { Camera } from '../types';

export const CameraManagement: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState('Viewer');
  
  // Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  
  // Form fields
  const [cameraIdField, setCameraIdField] = useState('');
  const [cameraNameField, setCameraNameField] = useState('');
  const [latField, setLatField] = useState(0.0);
  const [lonField, setLonField] = useState(0.0);
  const [streamUrlField, setStreamUrlField] = useState('');
  const [statusField, setStatusField] = useState<Camera['status']>('Offline');
  
  const [formError, setFormError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const fetchCameras = async () => {
    try {
      const data = await cameraService.getCameras();
      setCameras(data);
    } catch (err) {
      console.error('Failed to load cameras list:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCameras();
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        const u = JSON.parse(userStr);
        setUserRole(u.role);
      } catch (e) {
        // ignore
      }
    }
  }, []);

  const openCreateModal = () => {
    setModalMode('create');
    setCameraIdField(`CAM-${String(cameras.length + 1).padStart(3, '0')}`);
    setCameraNameField('');
    setLatField(16.8302);
    setLonField(75.7100);
    setStreamUrlField('');
    setStatusField('Offline');
    setFormError('');
    setIsModalOpen(true);
  };

  const openEditModal = (camera: Camera) => {
    setModalMode('edit');
    setSelectedCameraId(camera._id || camera.camera_id);
    setCameraIdField(camera.camera_id);
    setCameraNameField(camera.camera_name);
    setLatField(camera.latitude);
    setLonField(camera.longitude);
    setStreamUrlField(camera.stream_url);
    setStatusField(camera.status);
    setFormError('');
    setIsModalOpen(true);
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setFormError('');

    const payload = {
      camera_id: cameraIdField,
      camera_name: cameraNameField,
      latitude: Number(latField),
      longitude: Number(lonField),
      stream_url: streamUrlField,
      status: statusField,
    };

    try {
      if (modalMode === 'create') {
        await cameraService.createCamera(payload);
      } else if (modalMode === 'edit' && selectedCameraId) {
        await cameraService.updateCamera(selectedCameraId, payload);
      }
      setIsModalOpen(false);
      fetchCameras();
    } catch (err: any) {
      console.error(err);
      setFormError(err.response?.data?.detail || 'Failed to submit camera configurations.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteCamera = async (id: string) => {
    if (window.confirm('Are you sure you want to decommission this camera feed?')) {
      try {
        await cameraService.deleteCamera(id);
        fetchCameras();
      } catch (err) {
        console.error('Failed to delete camera:', err);
      }
    }
  };

  const isAuthorized = ['Super Admin', 'Police Admin'].includes(userRole);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#09090b] flex items-center justify-center">
        <div className="h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <CommandCenterLayout>
      <div className="space-y-8">
        {/* Header Section */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-900 pb-5">
          <div>
            <h1 className="text-3xl font-black tracking-wider text-zinc-100 uppercase flex items-center gap-3">
              <Sliders className="h-8 w-8 text-blue-500" />
              Camera Operations
            </h1>
            <p className="text-zinc-500 text-sm mt-1">
              Add, configure, or decommission surveillance nodes inside the smart city perimeter.
            </p>
          </div>
          {isAuthorized && (
            <button
              onClick={openCreateModal}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white font-bold py-2.5 px-4 rounded-lg text-sm tracking-wider uppercase transition shadow-lg shadow-blue-900/30 self-start sm:self-auto"
            >
              <Plus className="h-4 w-4" />
              <span>PROVISION CAMERA</span>
            </button>
          )}
        </div>

        {/* Read-Only Restriction Warning Banner */}
        {!isAuthorized && (
          <div className="flex items-center space-x-2.5 p-4 rounded-xl border border-zinc-800 bg-zinc-900/20 text-zinc-400 text-sm">
            <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0" />
            <span>
              Your current operator tier (<b>{userRole}</b>) is read-only. Modifying or provisioning cameras requires Admin credentials.
            </span>
          </div>
        )}

        {/* Cameras List Grid/Table */}
        <div className="glass-panel rounded-xl border border-zinc-800 overflow-hidden shadow-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900/40 text-xs font-bold text-zinc-500 uppercase tracking-widest">
                  <th className="py-4 px-6">Camera Node</th>
                  <th className="py-4 px-6">Coordinates (Lat / Lon)</th>
                  <th className="py-4 px-6">Stream Connection</th>
                  <th className="py-4 px-6">System Status</th>
                  {isAuthorized && <th className="py-4 px-6 text-right">Actions</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-900 text-sm">
                {cameras.map((camera) => (
                  <tr key={camera._id} className="hover:bg-zinc-900/20 transition duration-150">
                    <td className="py-4 px-6">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-400">
                          <Video className="h-4 w-4" />
                        </div>
                        <div>
                          <p className="font-extrabold text-zinc-200 tracking-wider uppercase">
                            {camera.camera_name}
                          </p>
                          <p className="text-xs text-zinc-500 font-semibold">{camera.camera_id}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-6 font-mono text-xs text-zinc-400">
                      {camera.latitude.toFixed(6)} / {camera.longitude.toFixed(6)}
                    </td>
                    <td className="py-4 px-6 font-mono text-xs text-zinc-500 max-w-[200px] truncate" title={camera.stream_url}>
                      {camera.stream_url}
                    </td>
                    <td className="py-4 px-6">
                      <span className={`text-[10px] font-bold px-2.5 py-1 rounded border uppercase tracking-wider ${
                        (camera.status?.toLowerCase() === 'online') ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/40' :
                        (camera.status?.toLowerCase() === 'no video input' || camera.status?.toLowerCase() === 'no_video_input') ? 'bg-amber-950/60 text-amber-400 border-amber-800/40' :
                        (camera.status?.toLowerCase() === 'tampered') ? 'bg-purple-950/60 text-purple-400 border-purple-800/40' :
                        (camera.status?.toLowerCase() === 'frozen') ? 'bg-yellow-950/60 text-yellow-400 border-yellow-800/40' :
                        camera.status === 'Warning' ? 'bg-amber-950/60 text-amber-400 border-amber-800/40' :
                        camera.status === 'Critical Alert' ? 'bg-red-950/60 text-red-400 border-red-800/40' :
                        'bg-zinc-800 text-zinc-400 border-zinc-700/40'
                      }`}>
                        {camera.status}
                      </span>
                    </td>
                    {isAuthorized && (
                      <td className="py-4 px-6 text-right">
                        <div className="flex items-center justify-end space-x-2.5">
                          <button
                            onClick={() => openEditModal(camera)}
                            className="p-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 rounded transition"
                            title="Edit Node"
                          >
                            <Edit2 className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteCamera(camera._id || camera.camera_id)}
                            className="p-1.5 bg-zinc-900 hover:bg-red-950/40 border border-zinc-800 hover:border-red-900/60 text-zinc-400 hover:text-red-400 rounded transition"
                            title="Decommission Node"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Modal Overlay Dialog */}
        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setIsModalOpen(false)} />
            <div className="relative w-full max-w-lg glass-panel p-8 rounded-2xl border border-zinc-800 shadow-2xl z-10">
              <div className="mb-6 flex items-center gap-3">
                <Radio className="h-6 w-6 text-blue-500" />
                <h3 className="text-lg font-black uppercase tracking-wider text-zinc-200">
                  {modalMode === 'create' ? 'Provision Surveillance Node' : 'Edit Surveillance Node'}
                </h3>
              </div>

              {formError && (
                <div className="p-3 rounded-lg border border-red-900/50 bg-red-950/20 text-red-400 text-xs mb-5 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>{formError}</span>
                </div>
              )}

              <form onSubmit={handleFormSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                      Camera ID code
                    </label>
                    <input
                      type="text"
                      required
                      disabled={modalMode === 'edit'}
                      value={cameraIdField}
                      onChange={(e) => setCameraIdField(e.target.value)}
                      placeholder="e.g. CAM-099"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500 disabled:opacity-50"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                      Operational Status
                    </label>
                    <select
                      value={statusField}
                      onChange={(e) => setStatusField(e.target.value as Camera['status'])}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500"
                    >
                      <option value="Online">Online</option>
                      <option value="No Video Input">No Video Input</option>
                      <option value="Warning">Warning</option>
                      <option value="Critical Alert">Critical Alert</option>
                      <option value="Offline">Offline</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                    Camera Node Name
                  </label>
                  <input
                    type="text"
                    required
                    value={cameraNameField}
                    onChange={(e) => setCameraNameField(e.target.value)}
                    placeholder="e.g. Broadway Crossing"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                      Latitude Coordinates
                    </label>
                    <input
                      type="number"
                      step="any"
                      required
                      value={latField}
                      onChange={(e) => setLatField(Number(e.target.value))}
                      placeholder="16.8302"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                      Longitude Coordinates
                    </label>
                    <input
                      type="number"
                      step="any"
                      required
                      value={lonField}
                      onChange={(e) => setLonField(Number(e.target.value))}
                      placeholder="75.7100"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[10px] font-bold uppercase tracking-wider text-zinc-450 mb-1.5">
                    Stream Source URL / File Path / Device Index
                  </label>
                  <input
                    type="text"
                    required
                    value={streamUrlField}
                    onChange={(e) => setStreamUrlField(e.target.value)}
                    placeholder="rtsp://192.168.1.100:554/live or 0 for OpenCV Local Camera"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 text-xs focus:outline-none focus:border-blue-500 font-mono"
                  />
                </div>

                <div className="pt-4 flex items-center justify-end space-x-3">
                  <button
                    type="button"
                    onClick={() => setIsModalOpen(false)}
                    className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-850 px-4 py-2 rounded-lg text-xs font-bold text-zinc-400 tracking-wider uppercase transition"
                  >
                    CANCEL
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-5 rounded-lg text-xs tracking-wider uppercase disabled:opacity-50 transition"
                  >
                    {submitting ? 'PROCESSING...' : 'APPLY CONFIGS'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </CommandCenterLayout>
  );
};
export default CameraManagement;
