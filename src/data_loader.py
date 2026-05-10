"""
Data Loader Module for EEG Motor Imagery Application.

This module handles loading and parsing EDF files containing EEG recordings
from the PhysioNet Motor Movement/Imagery Dataset.
"""

import mne
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any

# Suppress MNE output
mne.set_log_level('WARNING')


def load_edf_file(file_path: str) -> Optional[mne.io.Raw]:
    """
    Load EDF file with MNE. Standardise channel names to 10-20 system.
    Return None on failure - never crash.
    
    Parameters
    ----------
    file_path : str
        Path to the EDF file to be loaded.
        
    Returns
    -------
    mne.io.Raw or None
        MNE Raw object containing the EEG data, or None if loading fails.
    """
    try:
        # Load the EDF file using MNE
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Use MNE's built-in standardize function for PhysioNet EEG BCI dataset
        try:
            mne.datasets.eegbci.standardize(raw)
        except Exception:
            # Fallback: strip trailing dots manually (safe for non-PhysioNet files)
            rename = {ch: ch.rstrip('.').upper() for ch in raw.ch_names
                      if ch.rstrip('.') != ch}
            if rename:
                raw.rename_channels(rename)
        
        return raw
        
    except Exception as e:
        print(f"Error loading EDF file: {e}")
        return None


def extract_annotations(raw: mne.io.Raw) -> pd.DataFrame:
    """
    Return DataFrame with columns: onset, duration, description.
    Include T0, T1, T2.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data with annotations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing annotation information.
    """
    try:
        annotations = raw.annotations
        
        # Convert annotations to DataFrame
        annotations_df = pd.DataFrame({
            'onset': annotations.onset,
            'duration': annotations.duration,
            'description': annotations.description
        })
        
        return annotations_df
        
    except Exception as e:
        print(f"Error extracting annotations: {e}")
        return pd.DataFrame(columns=['onset', 'duration', 'description'])


def get_recording_metadata(raw: mne.io.Raw) -> Dict[str, Any]:
    """
    Return dict with keys:
    - sampling_frequency (float)
    - n_channels (int)
    - duration (float, seconds)
    - channel_names (list of str)
    - n_annotations (int)
    - annotation_counts (dict, e.g. {'T0':10, 'T1':15, 'T2':15})
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
        
    Returns
    -------
    dict
        Dictionary containing recording metadata.
    """
    try:
        info = raw.info
        annotations = raw.annotations
        
        # Count annotations by type
        annotation_counts = {}
        for desc in annotations.description:
            annotation_counts[desc] = annotation_counts.get(desc, 0) + 1
        
        metadata = {
            'sampling_frequency': float(info['sfreq']),
            'n_channels': int(info['nchan']),
            'duration': float(raw.times[-1]) if len(raw.times) > 0 else 0.0,
            'channel_names': list(info['ch_names']),
            'n_annotations': len(annotations),
            'annotation_counts': annotation_counts
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {
            'sampling_frequency': 0.0,
            'n_channels': 0,
            'duration': 0.0,
            'channel_names': [],
            'n_annotations': 0,
            'annotation_counts': {}
        }


def validate_edf_file(raw: mne.io.Raw) -> Dict[str, Any]:
    """
    Return dict with:
    - is_valid (bool)
    - has_motor_channels (bool)
    - has_t1_t2_annotations (bool)
    - warnings (list of str)
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object to validate.
        
    Returns
    -------
    dict
        Dictionary with validation results.
    """
    from src.utils import MI_CHANNELS
    
    validation = {
        'is_valid': False,
        'has_motor_channels': False,
        'has_t1_t2_annotations': False,
        'warnings': []
    }
    
    try:
        ch_names_upper = [ch.upper() for ch in raw.info['ch_names']]
        
        # Check for motor channels
        motor_channels_present = []
        for ch in MI_CHANNELS:
            if ch.upper() in ch_names_upper:
                motor_channels_present.append(ch)
        
        validation['has_motor_channels'] = len(motor_channels_present) > 0
        
        if not validation['has_motor_channels']:
            validation['warnings'].append(
                f"No motor imagery channels found. Expected: {MI_CHANNELS}"
            )
        elif len(motor_channels_present) < len(MI_CHANNELS):
            missing = set(MI_CHANNELS) - set(motor_channels_present)
            validation['warnings'].append(
                f"Missing some motor channels: {list(missing)}. "
                f"Found: {motor_channels_present}"
            )
        
        # Check for T1/T2 annotations
        descriptions = list(raw.annotations.description)
        has_t1 = 'T1' in descriptions
        has_t2 = 'T2' in descriptions
        validation['has_t1_t2_annotations'] = has_t1 and has_t2
        
        if not has_t1:
            validation['warnings'].append("No T1 (Left imagery) events found")
        if not has_t2:
            validation['warnings'].append("No T2 (Right imagery) events found")
        
        # Check sampling rate
        if raw.info['sfreq'] < 100:
            validation['warnings'].append(
                f"Low sampling rate: {raw.info['sfreq']} Hz. "
                "Expected >= 100 Hz for motor imagery analysis."
            )
        
        # Determine overall validity
        validation['is_valid'] = (
            validation['has_motor_channels'] and 
            validation['has_t1_t2_annotations'] and
            raw.info['sfreq'] >= 100
        )
        
    except Exception as e:
        validation['warnings'].append(f"Validation error: {e}")
        
    return validation
