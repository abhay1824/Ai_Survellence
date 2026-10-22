/**
 * Threat levels that trigger audio siren playback and the red emergency alert modal.
 * Default is ['CRITICAL'].
 * Setting ['CRITICAL', 'HIGH'] allows confirmed HIGH events to sound the siren and open modal.
 */
export const SIREN_TRIGGER_LEVELS: string[] = ['CRITICAL', 'HIGH'];
