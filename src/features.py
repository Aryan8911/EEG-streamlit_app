"""
Feature Extraction Module for EEG Motor Imagery Application.

This module contains functions for extracting features from EEG epochs
for motor imagery classification.

CRITICAL: The feature vector must be exactly 38 elements in the specified order
to match the training notebook.
"""

import mne
import numpy as np
from scipy import signal
from scipy.integrate import simpson
from typing import Dict, List, Tuple, Any

# Suppress MNE output
mne.set_log_level('WARNING')

# Frequency bands (must match training notebook)
MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Channel order (must match training notebook)
FEATURE_CHANNELS = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']


def band_power(signal_data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> float:
    """
    Welch PSD integrated over band. Signal in V, multiply by 1e6 internally.
    
    Parameters
    ----------
    signal_data : np.ndarray
        1D EEG signal (in Volts).
    sfreq : float
        Sampling frequency in Hz.
    band : tuple
        Frequency band (low, high) in Hz.
        
    Returns
    -------
    float
        Band power integrated using Simpson's rule.
    """
    # Convert V to uV for numerical stability
    signal_uv = signal_data * 1e6
    
    # Compute Welch PSD
    nperseg = min(int(sfreq), len(signal_uv))
    freqs, psd = signal.welch(signal_uv, fs=sfreq, nperseg=nperseg)
    
    # Find frequency indices within band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    if not np.any(idx_band):
        return 0.0
    
    # Integrate using Simpson's rule
    bp = simpson(psd[idx_band], x=freqs[idx_band])
    
    return bp


def total_power(signal_data: np.ndarray, sfreq: float, fmin: float = 1.0, fmax: float = 45.0) -> float:
    """
    Total power for relative feature computation.
    
    Parameters
    ----------
    signal_data : np.ndarray
        1D EEG signal (in Volts).
    sfreq : float
        Sampling frequency in Hz.
    fmin : float, default=1.0
        Minimum frequency.
    fmax : float, default=45.0
        Maximum frequency.
        
    Returns
    -------
    float
        Total power in the specified frequency range.
    """
    # Convert V to uV
    signal_uv = signal_data * 1e6
    
    # Compute Welch PSD
    nperseg = min(int(sfreq), len(signal_uv))
    freqs, psd = signal.welch(signal_uv, fs=sfreq, nperseg=nperseg)
    
    # Find frequency indices within range
    idx_range = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    if not np.any(idx_range):
        return 1e-8  # Avoid division by zero
    
    # Integrate using Simpson's rule
    tp = simpson(psd[idx_range], x=freqs[idx_range])
    
    return tp if tp > 0 else 1e-8


def epoch_features(epoch_data: np.ndarray, sfreq: float, channel_names: List[str]) -> np.ndarray:
    """
    Extract features from a single epoch.
    
    epoch_data shape: (n_channels, n_times)
    Returns 1D array of length 38:
      [absolute band powers x 18] + [relative band powers x 18] + [lateralisation x 2]
    
    MUST match training notebook feature order exactly.
    
    Parameters
    ----------
    epoch_data : np.ndarray
        Single epoch data of shape (n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    channel_names : list
        List of channel names corresponding to epoch_data rows.
        
    Returns
    -------
    np.ndarray
        1D feature array of length 38.
    """
    # Map channel names to indices (case-insensitive)
    ch_name_upper = {ch.upper(): i for i, ch in enumerate(channel_names)}
    
    # Prepare feature arrays
    abs_powers = []  # 18 features: 9 channels x 2 bands (mu, beta)
    rel_powers = []  # 18 features: 9 channels x 2 bands (mu, beta)
    
    # Storage for lateralisation calculation
    c3_mu, c3_beta = None, None
    c4_mu, c4_beta = None, None
    
    # Extract features for each channel in order
    for ch_name in FEATURE_CHANNELS:
        ch_upper = ch_name.upper()
        
        if ch_upper in ch_name_upper:
            ch_idx = ch_name_upper[ch_upper]
            ch_signal = epoch_data[ch_idx, :]
            
            # Compute absolute band powers
            mu_power = band_power(ch_signal, sfreq, MU_BAND)
            beta_power = band_power(ch_signal, sfreq, BETA_BAND)
            
            # Compute total power for relative features
            tp = total_power(ch_signal, sfreq, fmin=1.0, fmax=45.0)
            
            # Compute relative band powers
            mu_rel = mu_power / tp
            beta_rel = beta_power / tp
            
            # Store absolute powers
            abs_powers.append(mu_power)
            abs_powers.append(beta_power)
            
            # Store relative powers
            rel_powers.append(mu_rel)
            rel_powers.append(beta_rel)
            
            # Store C3/C4 for lateralisation
            if ch_upper == 'C3':
                c3_mu, c3_beta = mu_power, beta_power
            elif ch_upper == 'C4':
                c4_mu, c4_beta = mu_power, beta_power
        else:
            # Channel not found - use zeros
            abs_powers.extend([0.0, 0.0])
            rel_powers.extend([0.0, 0.0])
    
    # Compute lateralisation indices (only if both C3 and C4 present)
    if c3_mu is not None and c4_mu is not None:
        lat_mu = (c4_mu - c3_mu) / (c4_mu + c3_mu + 1e-8)
        lat_beta = (c4_beta - c3_beta) / (c4_beta + c3_beta + 1e-8)
    else:
        lat_mu, lat_beta = 0.0, 0.0
    
    # Combine all features: absolute (18) + relative (18) + lateralisation (2) = 38
    features = np.array(abs_powers + rel_powers + [lat_mu, lat_beta], dtype=np.float32)
    
    return features


def extract_bandpower_features(epochs: mne.Epochs) -> np.ndarray:
    """
    Process all epochs. Returns X of shape (n_epochs, n_features).
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object containing segmented EEG data.
        
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_epochs, 38).
    """
    try:
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        sfreq = epochs.info['sfreq']
        ch_names = epochs.info['ch_names']
        
        n_epochs = data.shape[0]
        features_list = []
        
        for epoch_idx in range(n_epochs):
            epoch_data = data[epoch_idx]
            feats = epoch_features(epoch_data, sfreq, ch_names)
            features_list.append(feats)
        
        X = np.array(features_list)
        return X
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.array([])


def get_feature_config() -> Dict[str, Any]:
    """
    Return dict describing feature settings for display.
    
    Returns
    -------
    dict
        Dictionary containing feature extraction parameters.
    """
    return {
        'n_features': 38,
        'feature_breakdown': {
            'absolute_band_powers': 18,
            'relative_band_powers': 18,
            'lateralisation_indices': 2
        },
        'frequency_bands': {
            'mu': MU_BAND,
            'beta': BETA_BAND
        },
        'channels': FEATURE_CHANNELS,
        'psd_method': 'Welch',
        'integration_method': "Simpson's rule",
        'feature_order': [
            'Absolute mu/beta power for each of 9 channels (18 features)',
            'Relative mu/beta power for each of 9 channels (18 features)',
            'Lateralisation indices: lat_mu, lat_beta (2 features)'
        ]
    }
