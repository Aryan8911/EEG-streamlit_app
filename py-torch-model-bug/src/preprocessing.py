"""
Preprocessing Module for EEG Motor Imagery Application.

This module contains functions for preprocessing EEG signals including
filtering, CAR, epoching, and signal normalization.

CRITICAL: The preprocessing pipeline must match the training notebook exactly.
"""

import mne
import numpy as np
from scipy import signal
from typing import Optional, Dict, Tuple, Any

# Suppress MNE output
mne.set_log_level('WARNING')

# Motor imagery channels (must match training notebook)
MOTOR_CHS = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']

# Event IDs for motor imagery (PhysioNet convention)
EVENT_ID = {'T1': 2, 'T2': 3}


def apply_filter(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    """
    Band-pass FIR Hamming filter. Returns a copy.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
    l_freq : float, default=8.0
        Low cutoff frequency for bandpass filter (Hz).
    h_freq : float, default=30.0
        High cutoff frequency for bandpass filter (Hz).
        
    Returns
    -------
    mne.io.Raw
        Filtered MNE Raw object (copy).
    """
    try:
        # Create a copy to avoid modifying original
        raw_filtered = raw.copy()
        
        # Apply FIR bandpass filter with Hamming window, zero-phase
        raw_filtered.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method='fir',
            fir_window='hamming',
            phase='zero',
            verbose=False
        )
        
        return raw_filtered
        
    except Exception as e:
        print(f"Error applying filter: {e}")
        return raw.copy()


def apply_car(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Common Average Reference. Returns a copy.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
        
    Returns
    -------
    mne.io.Raw
        CAR-referenced MNE Raw object (copy).
    """
    try:
        raw_car = raw.copy()
        raw_car.set_eeg_reference('average', projection=False, verbose=False)
        return raw_car
    except Exception as e:
        print(f"Error applying CAR: {e}")
        return raw.copy()


def per_subject_normalise(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Channel-wise z-score over entire recording. Returns a copy.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
        
    Returns
    -------
    mne.io.Raw
        Z-score normalised MNE Raw object (copy).
    """
    try:
        raw_norm = raw.copy()
        data = raw_norm.get_data()
        
        # Z-score normalisation over time axis per channel
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        data_norm = (data - mean) / (std + 1e-8)
        
        # Update the raw object's data
        raw_norm._data = data_norm
        
        return raw_norm
        
    except Exception as e:
        print(f"Error normalising: {e}")
        return raw.copy()


def select_motor_channels(raw: mne.io.Raw, channels: list = None) -> mne.io.Raw:
    """
    Pick available motor channels. Warn if any missing. Returns a copy.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
    channels : list, optional
        List of channel names to select. Defaults to MOTOR_CHS.
        
    Returns
    -------
    mne.io.Raw
        MNE Raw object with only motor channels (copy).
    """
    if channels is None:
        channels = MOTOR_CHS
    
    try:
        raw_picked = raw.copy()
        
        # Get available channels (case-insensitive matching)
        available_upper = {ch.upper(): ch for ch in raw.ch_names}
        channels_to_pick = []
        missing_channels = []
        
        for ch in channels:
            ch_upper = ch.upper()
            if ch_upper in available_upper:
                channels_to_pick.append(available_upper[ch_upper])
            else:
                missing_channels.append(ch)
        
        if missing_channels:
            print(f"Warning: Missing motor channels: {missing_channels}")
        
        if not channels_to_pick:
            print("Warning: No motor channels found. Using all available channels.")
            return raw_picked
        
        raw_picked.pick_channels(channels_to_pick, ordered=True)
        return raw_picked
        
    except Exception as e:
        print(f"Error selecting channels: {e}")
        return raw.copy()


def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Full pipeline: filter -> CAR -> z-score -> channel select. Used for inference.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing raw EEG data.
        
    Returns
    -------
    mne.io.Raw
        Fully preprocessed MNE Raw object.
    """
    # Step 1: Band-pass filter 8-30 Hz
    raw_filt = apply_filter(raw, l_freq=8.0, h_freq=30.0)
    
    # Step 2: Common Average Reference
    raw_car = apply_car(raw_filt)
    
    # Step 3: Per-subject z-score normalisation
    raw_norm = per_subject_normalise(raw_car)
    
    # Step 4: Select motor channels
    raw_motor = select_motor_channels(raw_norm, MOTOR_CHS)
    
    return raw_motor


def create_epochs_from_annotations(
    raw: mne.io.Raw,
    tmin: float = 0.5,
    tmax: float = 3.5,
    baseline: Tuple[float, float] = (0.5, 1.0)
) -> Optional[mne.Epochs]:
    """
    Extract T1/T2 epochs. Return None if no events found.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data with annotations.
    tmin : float, default=0.5
        Start time of epoch relative to event onset (seconds).
    tmax : float, default=3.5
        End time of epoch relative to event onset (seconds).
    baseline : tuple, default=(0.5, 1.0)
        Time interval for baseline correction (start, end) in seconds.
        
    Returns
    -------
    mne.Epochs or None
        MNE Epochs object, or None if creation fails.
    """
    try:
        # Get events from annotations
        events, event_id_found = mne.events_from_annotations(raw, verbose=False)
        
        # Map to our event IDs (T1=Left, T2=Right)
        mapped_event_id = {}
        for name in ['T1', 'T2']:
            if name in event_id_found:
                mapped_event_id[name] = event_id_found[name]
        
        if not mapped_event_id:
            print("No T1/T2 events found in annotations")
            return None
        
        # Filter events to only include T1/T2
        valid_codes = list(mapped_event_id.values())
        mask = np.isin(events[:, 2], valid_codes)
        events_filtered = events[mask]
        
        if len(events_filtered) == 0:
            print("No T1/T2 events found after filtering")
            return None
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events_filtered,
            event_id=mapped_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False
        )
        
        return epochs
        
    except Exception as e:
        print(f"Error creating epochs: {e}")
        return None


def compute_psd(
    raw: mne.io.Raw,
    fmin: float = 1.0,
    fmax: float = 60.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (psd_array, freqs_array) using Welch method. 
    Shape: (n_channels, n_freqs).
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    fmin : float, default=1.0
        Minimum frequency for PSD computation.
    fmax : float, default=60.0
        Maximum frequency for PSD computation.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (psd, freqs) arrays.
    """
    try:
        spectrum = raw.compute_psd(
            method='welch',
            fmin=fmin,
            fmax=fmax,
            n_fft=int(raw.info['sfreq']),
            verbose=False
        )
        psd = spectrum.get_data()
        freqs = spectrum.freqs
        return psd, freqs
        
    except Exception as e:
        print(f"Error computing PSD: {e}")
        return np.array([]), np.array([])


def get_preprocessing_config() -> Dict[str, Any]:
    """
    Return dict of pipeline settings for display in the UI.
    
    Returns
    -------
    dict
        Dictionary containing preprocessing parameters.
    """
    return {
        'filter': {
            'type': 'FIR',
            'window': 'Hamming',
            'phase': 'zero-phase',
            'l_freq': 8.0,
            'h_freq': 30.0
        },
        'reference': {
            'type': 'Common Average Reference (CAR)'
        },
        'normalisation': {
            'type': 'Per-subject z-score',
            'axis': 'time (per channel)'
        },
        'channels': {
            'selected': MOTOR_CHS,
            'count': len(MOTOR_CHS)
        },
        'epoching': {
            'tmin': 0.5,
            'tmax': 3.5,
            'baseline': (0.5, 1.0),
            'events': ['T1 (Left)', 'T2 (Right)']
        }
    }
