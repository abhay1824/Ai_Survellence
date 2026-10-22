import React, { useEffect, useRef } from 'react';
import type { ThreatLevel } from '../../types/surveillance';
import { SIREN_TRIGGER_LEVELS } from '../../constants/alertLevels';

interface SirenAudioPlayerProps {
  threatLevel: ThreatLevel;
  muted: boolean;
}

export const SirenAudioPlayer: React.FC<SirenAudioPlayerProps> = ({ threatLevel, muted }) => {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscRef = useRef<OscillatorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const intervalRef = useRef<any>(null);

  useEffect(() => {
    if (SIREN_TRIGGER_LEVELS.includes(threatLevel) && !muted) {
      startSiren();
    } else {
      stopSiren();
    }

    return () => {
      stopSiren();
    };
  }, [threatLevel, muted]);

  const startSiren = () => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      if (oscRef.current) return; // already running

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sawtooth';
      gain.gain.setValueAtTime(0.08, ctx.currentTime); // Low volume for safety

      osc.connect(gain);
      gain.connect(ctx.destination);

      let high = false;
      osc.frequency.setValueAtTime(600, ctx.currentTime);
      osc.start();

      intervalRef.current = setInterval(() => {
        if (!oscRef.current) return;
        high = !high;
        const targetFreq = high ? 950 : 550;
        osc.frequency.exponentialRampToValueAtTime(targetFreq, ctx.currentTime + 0.25);
      }, 350);

      oscRef.current = osc;
      gainRef.current = gain;
    } catch (e) {
      console.warn('Audio siren context error:', e);
    }
  };

  const stopSiren = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (oscRef.current) {
      try {
        oscRef.current.stop();
        oscRef.current.disconnect();
      } catch (e) {}
      oscRef.current = null;
    }
    if (gainRef.current) {
      try {
        gainRef.current.disconnect();
      } catch (e) {}
      gainRef.current = null;
    }
  };

  return null;
};
