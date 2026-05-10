"""
Utility Module for EEG Motor Imagery Application.

This module contains helper functions, constants, and utilities
used across the application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# Application constants
APP_CONFIG = {
    'title': 'EEG Motor Imagery Prediction',
    'version': '1.0.0',
    'dataset': 'PhysioNet EEG Motor Movement/Imagery',
    'runs': [4, 8, 12],
    'filter_fmin': 8.0,
    'filter_fmax': 30.0,
    'epoch_tmin': 0.5,
    'epoch_tmax': 3.5,
}

# PhysioNet Motor Imagery event codes
EVENT_CODES = {
    'T0': {'name': 'Rest', 'color': '#8D99AE'},
    'T1': {'name': 'Left Fist (Imagined)', 'color': '#4A90D9'},
    'T2': {'name': 'Right Fist (Imagined)', 'color': '#E84040'},
}

# Motor imagery relevant channels (must match training notebook)
MI_CHANNELS = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def create_session_state_defaults():
    """
    Initialise all required st.session_state keys if not present:
    file_uploaded, raw_data, filtered_data, metadata, annotations,
    epochs, features, predictions, processing_complete, models_loaded,
    selected_channels, time_range, show_filtered, show_psd, show_events,
    selected_model, current_file, available_channels, recording_duration
    """
    defaults = {
        'file_uploaded': False,
        'raw_data': None,
        'filtered_data': None,
        'metadata': None,
        'annotations': None,
        'epochs': None,
        'features': None,
        'predictions': None,
        'processing_complete': False,
        'models_loaded': False,
        'selected_channels': None,
        'time_range': (0.0, 10.0),
        'show_filtered': False,
        'show_psd': False,
        'show_events': True,
        'selected_model': None,
        'current_file': None,
        'available_channels': [],
        'recording_duration': 0.0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def load_offline_results() -> pd.DataFrame:
    """
    Load results/offline_model_results.csv.
    Expected columns: model_name, accuracy, f1_score, precision, recall, roc_auc.
    Return empty DataFrame (not crash) if file missing.
    Normalise column names flexibly (handle 'model', 'f1', 'auc' variants).
    """
    filepath = get_project_root() / 'results' / 'offline_model_results.csv'
    
    try:
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Normalise column names
            column_mapping = {
                'model': 'model_name',
                'f1': 'f1_score',
                'auc': 'roc_auc'
            }
            
            for old, new in column_mapping.items():
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})
            
            return df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'model_name', 'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'type'
            ])
            
    except Exception as e:
        print(f"Error loading offline results: {e}")
        return pd.DataFrame(columns=[
            'model_name', 'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'type'
        ])


def get_annotation_table() -> pd.DataFrame:
    """Return a static reference DataFrame explaining T0/T1/T2 with descriptions."""
    return pd.DataFrame([
        {
            'Annotation': 'T0',
            'Event': 'Rest',
            'Description': 'Eyes open, resting state baseline period',
            'Label': 'N/A (not classified)'
        },
        {
            'Annotation': 'T1',
            'Event': 'Left Fist Imagery',
            'Description': 'Imagined left fist opening/closing movement',
            'Label': '0 (Left)'
        },
        {
            'Annotation': 'T2',
            'Event': 'Right Fist Imagery',
            'Description': 'Imagined right fist opening/closing movement',
            'Label': '1 (Right)'
        }
    ])


def get_pipeline_steps() -> List[Dict[str, Any]]:
    """
    Return list of dicts describing pipeline steps for display.
    Each dict: {step: int, icon: str, name: str, description: str}
    """
    return [
        {
            'step': 1,
            'icon': '📁',
            'name': 'Upload EDF',
            'description': 'Load EDF file and extract raw EEG data with annotations'
        },
        {
            'step': 2,
            'icon': '🔧',
            'name': 'Band-pass Filter',
            'description': 'Apply 8-30 Hz FIR Hamming filter (zero-phase)'
        },
        {
            'step': 3,
            'icon': '⚡',
            'name': 'Common Average Reference',
            'description': 'Apply CAR to reduce common noise'
        },
        {
            'step': 4,
            'icon': '📊',
            'name': 'Z-Score Normalisation',
            'description': 'Per-channel z-score over entire recording'
        },
        {
            'step': 5,
            'icon': '🎯',
            'name': 'Channel Selection',
            'description': f'Select motor channels: {", ".join(MI_CHANNELS)}'
        },
        {
            'step': 6,
            'icon': '✂️',
            'name': 'Epoch Extraction',
            'description': 'Extract T1/T2 epochs (0.5-3.5s from cue onset)'
        },
        {
            'step': 7,
            'icon': '📈',
            'name': 'Feature Extraction',
            'description': 'Compute 38-feature vector (band power + lateralisation)'
        },
        {
            'step': 8,
            'icon': '🤖',
            'name': 'Model Inference',
            'description': 'Predict motor imagery class (Left vs Right)'
        }
    ]


def generate_sample_predictions(n_epochs: int = 20) -> pd.DataFrame:
    """
    Generate synthetic prediction data for demo when no model is loaded.
    Returns DataFrame matching create_results_dataframe output schema.
    """
    np.random.seed(42)
    
    labels = ['Left', 'Right']
    ground_truth = np.random.choice(labels, n_epochs)
    
    # Simulate ~75% accuracy
    predictions = ground_truth.copy()
    n_errors = int(n_epochs * 0.25)
    error_indices = np.random.choice(n_epochs, n_errors, replace=False)
    for idx in error_indices:
        predictions[idx] = 'Right' if predictions[idx] == 'Left' else 'Left'
    
    return pd.DataFrame({
        'epoch': list(range(n_epochs)),
        'start_time': [i * 4.0 + 0.5 for i in range(n_epochs)],
        'end_time': [i * 4.0 + 3.5 for i in range(n_epochs)],
        'ground_truth': ground_truth,
        'prediction': predictions,
        'correct': ground_truth == predictions,
        'confidence': np.random.uniform(0.55, 0.95, n_epochs)
    })


def format_duration(seconds: float) -> str:
    """Format e.g. 125.4 -> '2m 5s'"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_metric(value: float, as_percent: bool = True) -> str:
    """Format e.g. 0.7834 -> '78.3%' or '0.783'"""
    if as_percent:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def validate_channels(requested: List[str], available: List[str]) -> List[str]:
    """
    Validate and filter channel names.
    
    Returns list of valid channel names that exist in both lists.
    """
    available_upper = {ch.upper(): ch for ch in available}
    valid = []
    
    for ch in requested:
        ch_upper = ch.upper()
        if ch_upper in available_upper:
            valid.append(available_upper[ch_upper])
    
    return valid


def create_metric_card_html(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "#00B4D8"
) -> str:
    """
    Generate HTML for a styled metric card.
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(27,38,59,0.9) 0%, rgba(13,27,42,0.9) 100%);
        border: 1px solid rgba(0,180,216,0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <p style="margin: 0; color: #8D99AE; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
            {title}
        </p>
        <p style="margin: 10px 0 5px 0; color: {color}; font-size: 36px; font-weight: bold;">
            {value}
        </p>
        <p style="margin: 0; color: #E0E1DD; font-size: 12px;">
            {subtitle}
        </p>
    </div>
    """


def clear_session_state():
    """Clear all session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    create_session_state_defaults()


def save_results(results_df: pd.DataFrame, filepath: str = None) -> bool:
    """
    Save prediction results to CSV file.
    """
    if filepath is None:
        filepath = get_project_root() / 'results' / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False
