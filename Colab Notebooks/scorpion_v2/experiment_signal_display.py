"""Display-only filter alternatives and phase intervals; never accept annotations."""
import numpy as np
from scipy import signal
from scorpion_qc import PARAMS, MODES, MODE_NAMES, mains_notch, zero_phase


def ecg_variants(x, fs):
    """Two independently switchable, whole-record, zero-phase filter stages."""
    sos = signal.butter(PARAMS['ecg_band_order'], PARAMS['ecg_display_band_hz'],
                        btype='bandpass', fs=fs, output='sos')
    notch = mains_notch(x, fs, PARAMS['ecg_notch_hz'])
    return {'00': np.asarray(x), '10': notch,
            '01': zero_phase(sos, x, fs, PARAMS['ecg_padding_s']),
            '11': zero_phase(sos, notch, fs, PARAMS['ecg_padding_s'])}


def spectrum(x, fs):
    """Welch PSD: full recording, Hann 8 s windows, 50% overlap, mean removal."""
    return signal.welch(x, fs=fs, window='hann', nperseg=min(len(x), int(8*fs)),
                        detrend='constant', scaling='density')


def phase_intervals(breathing, fs, duration):
    """Retain the source's candidate boundaries and reported four-mode order."""
    boundaries = np.asarray(breathing.get('boundaries_samples', []), dtype=float)/fs
    if np.any(np.diff(boundaries) <= 0) or np.any(boundaries <= 0) or np.any(boundaries >= duration):
        raise ValueError('Invalid candidate breathing boundaries')
    if breathing.get('single_mode') in MODES and not len(boundaries):
        modes = [breathing['single_mode']]
    elif len(boundaries) == len(MODES)-1:
        modes = MODES
    else:
        return []  # Missing reconstruction must not become invented phase labels.
    edges = np.r_[0., boundaries, duration]
    return [dict(start=float(edges[i]), end=float(edges[i+1]), mode=m,
                 label=MODE_NAMES[MODES.index(m)]) for i, m in enumerate(modes)]
