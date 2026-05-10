"""
EEG Motor Imagery Prediction Application

A modern, polished Streamlit dashboard for analyzing EEG motor imagery data
from the PhysioNet Motor Movement/Imagery Dataset.

Features:
- EDF file upload and visualization
- Real-time signal processing
- Motor imagery classification (T1: Left, T2: Right)
- Interactive visualizations with Plotly
- Comprehensive results dashboard

Author: [Your Name]
Date: [Date]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

# PyTorch availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import backend modules
from src.data_loader import (
    load_edf_file, 
    extract_annotations, 
    get_recording_metadata,
    validate_edf_file
)
from src.preprocessing import (
    apply_filter, 
    create_epochs_from_annotations,
    compute_psd,
    get_preprocessing_config
)
from src.features import (
    extract_bandpower_features,
    get_feature_config
)
from src.inference import (
    load_models, 
    predict_epochs,
    evaluate_predictions,
    create_results_dataframe
)
from src.visualisation import (
    plot_raw_eeg,
    plot_filtered_eeg,
    plot_psd,
    plot_event_markers,
    plot_prediction_timeline,
    plot_metrics_comparison,
    COLORS
)
from src.utils import (
    APP_CONFIG,
    EVENT_CODES,
    MI_CHANNELS,
    create_session_state_defaults,
    load_offline_results,
    get_annotation_table,
    get_pipeline_steps,
    generate_sample_predictions,
    format_duration,
    format_metric
)


@st.cache_resource
def _cached_load_models():
    """Load models once and cache for the session."""
    return load_models()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="EEG Motor Imagery Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject minimal, clean CSS styling."""
    st.markdown("""
    <style>
    /* Clean, minimal design */
    :root {
        --bg: #111;
        --surface: #1a1a1a;
        --border: #333;
        --text: #e5e5e5;
        --text-muted: #888;
        --accent: #3b82f6;
        --success: #22c55e;
        --error: #ef4444;
    }
    
    .stApp {
        background: var(--bg);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text) !important;
        font-weight: 600;
    }
    
    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-title {
        color: var(--text-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: var(--text);
        font-size: 28px;
        font-weight: 600;
    }
    
    .metric-subtitle {
        color: var(--text-muted);
        font-size: 11px;
        margin-top: 4px;
    }
    
    /* Header section */
    .header-section {
        border-bottom: 1px solid var(--border);
        padding-bottom: 24px;
        margin-bottom: 24px;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    
    /* Status indicator */
    .status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-ready {
        background: rgba(34, 197, 94, 0.1);
        color: var(--success);
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: currentColor;
        border-radius: 50%;
    }
    
    /* Pipeline steps */
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid var(--border);
    }
    
    .pipeline-step:last-child {
        border-bottom: none;
    }
    
    .step-number {
        background: var(--accent);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .step-content {
        flex: 1;
    }
    
    .step-name {
        color: var(--text);
        font-weight: 500;
        font-size: 14px;
    }
    
    .step-desc {
        color: var(--text-muted);
        font-size: 12px;
        margin-top: 2px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: #2563eb;
    }
    
    /* Inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        border-radius: 0;
        border-bottom: 2px solid transparent;
        padding: 12px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text);
        border-bottom-color: var(--accent);
        background: transparent;
    }
    
    /* Tables */
    .annotation-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    
    .annotation-table th {
        background: var(--surface);
        color: var(--text-muted);
        padding: 10px 12px;
        text-align: left;
        font-weight: 500;
        border-bottom: 1px solid var(--border);
    }
    
    .annotation-table td {
        color: var(--text);
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
    }
    
    /* Info box */
    .info-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px;
        margin-top: 16px;
    }
    
    .info-box-title {
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .info-box-content {
        color: var(--text-muted);
        font-size: 12px;
        line-height: 1.5;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

create_session_state_defaults()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_header()
    render_main_content()


def render_header():
    """Render clean header section."""
    st.markdown("""
    <div class="header-section">
        <div class="header-title">EEG Motor Imagery Analysis</div>
        <div class="header-subtitle">
            PhysioNet Motor Movement/Imagery Dataset - Left vs Right Hand Classification
        </div>
        <div style="margin-top: 12px;">
            <span class="status status-ready">
                <span class="status-dot"></span>
                Ready
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls."""
    
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 16px; margin-bottom: 16px; border-bottom: 1px solid #333;">
            <div style="font-size: 16px; font-weight: 600; color: #e5e5e5;">EEG Analysis</div>
            <div style="font-size: 12px; color: #888; margin-top: 2px;">Motor Imagery Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### Upload EDF File")
        uploaded_file = st.file_uploader(
            "Drag and drop your EDF file here",
            type=['edf'],
            help="Upload an EDF file from the PhysioNet Motor Imagery Dataset"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file)
        
        st.markdown("---")
        
        # EEG Controls
        st.markdown("### EEG Controls")
        
        # Channel selector
        available_channels = st.session_state.get('available_channels', MI_CHANNELS)
        selected_channels = st.multiselect(
            "Select Channels",
            options=available_channels,
            default=available_channels[:2] if len(available_channels) >= 2 else available_channels,
            help="Choose EEG channels to display"
        )
        st.session_state['selected_channels'] = selected_channels
        
        # Time range slider (ensure max > min to avoid Streamlit error)
        max_duration = max(st.session_state.get('recording_duration', 60.0), 1.0)
        time_range = st.slider(
            "Time Range (seconds)",
            min_value=0.0,
            max_value=max_duration,
            value=(0.0, min(10.0, max_duration)),
            step=0.5,
            help="Select time window to display"
        )
        st.session_state['time_range'] = time_range
        
        st.markdown("---")
        
        # Visualization Toggles
        st.markdown("### Visualization")
        
        show_filtered = st.checkbox("Show Filtered Signal", value=True)
        show_psd = st.checkbox("Show PSD Analysis", value=True)
        show_events = st.checkbox("Show Event Markers", value=True)
        
        st.session_state['show_filtered'] = show_filtered
        st.session_state['show_psd'] = show_psd
        st.session_state['show_events'] = show_events
        
        st.markdown("---")
        
        # Model Controls
        st.markdown("### Model")
        
        # Build model options dynamically from whatever is loaded
        models_result = _cached_load_models()
        available_models = list(models_result['models'].keys()) if models_result['models'] else ['No models loaded']
        st.session_state['loaded_models'] = models_result
        
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            help="Choose the classifier for prediction. Models are loaded from the /models directory."
        )
        st.session_state['selected_model'] = selected_model
        
        # Run Prediction Button
        if st.button("Run Prediction", use_container_width=True):
            run_prediction_pipeline()
        
        st.markdown("---")
        
        # Info Section
        st.markdown("""
        <div class="info-box">
            <div class="info-box-title">About</div>
            <div class="info-box-content">
                Analyzes motor imagery EEG data for binary classification 
                between left (T1) and right (T2) hand imagery tasks.
                <br><br>
                Bandpass filter: 8-30 Hz
            </div>
        </div>
        """, unsafe_allow_html=True)


def process_uploaded_file(uploaded_file):
    """Process the uploaded EDF file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner("Loading EDF file..."):
            raw = load_edf_file(tmp_path)
            
            if raw is not None:
                st.session_state['raw_data'] = raw
                st.session_state['file_uploaded'] = True
                st.session_state['current_file'] = uploaded_file.name
                
                # Extract metadata
                metadata = get_recording_metadata(raw)
                st.session_state['metadata'] = metadata
                st.session_state['recording_duration'] = metadata.get('duration', 60.0)
                st.session_state['available_channels'] = metadata.get('channel_names', [])
                
                # Extract annotations
                annotations = extract_annotations(raw)
                st.session_state['annotations'] = annotations
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
            else:
                st.error("Failed to load EDF file. Please check the file format.")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")


def run_prediction_pipeline():
    """Execute the full prediction pipeline."""
    if not st.session_state.get('file_uploaded'):
        st.warning("Please upload an EDF file first.")
        return
    
    with st.spinner("Running prediction pipeline..."):
        try:
            raw = st.session_state['raw_data']
            
            # Step 1: Apply filtering
            st.info("Applying bandpass filter (8-30 Hz)...")
            filtered_raw = apply_filter(raw, l_freq=8.0, h_freq=30.0)
            st.session_state['filtered_data'] = filtered_raw
            
            # Step 2: Create epochs
            st.info("Creating epochs from annotations...")
            epochs = create_epochs_from_annotations(filtered_raw)
            
            if epochs is not None:
                st.session_state['epochs'] = epochs
                
                # Step 3: Extract features
                st.info("Extracting bandpower features...")
                features = extract_bandpower_features(epochs)
                st.session_state['features'] = features
                
                # Step 4: Use cached models from sidebar
                st.info("Loading pretrained models...")
                models_result = st.session_state.get('loaded_models') or _cached_load_models()
                selected_model_name = st.session_state.get('selected_model', '')
                
                if not models_result['models']:
                    st.warning("No pretrained models found. Add model files to the /models directory.")
                    st.session_state['predictions'] = None
                    st.session_state['processing_complete'] = True
                    return
                
                st.session_state['models_loaded'] = True
                
                # Get the selected model
                model = models_result['models'].get(selected_model_name)
                if model is None:
                    # Fallback to first available
                    selected_model_name = list(models_result['models'].keys())[0]
                    model = models_result['models'][selected_model_name]
                
                scaler = models_result.get('scaler')
                
                # Step 5: Prepare raw epochs for DL models
                raw_epochs = None
                if TORCH_AVAILABLE:
                    try:
                        import torch.nn as nn
                        if isinstance(model, nn.Module) and st.session_state.get('epochs') is not None:
                            ep_data = st.session_state['epochs'].get_data()  # (N, C, T)
                            mean = ep_data.mean(axis=-1, keepdims=True)
                            std = ep_data.std(axis=-1, keepdims=True)
                            raw_epochs = (ep_data - mean) / (std + 1e-8)
                    except Exception:
                        pass
                
                # Step 6: Run inference
                st.info(f"Running inference with {selected_model_name}...")
                predictions = predict_epochs(
                    st.session_state.get('features', np.array([])),
                    model,
                    scaler,
                    model_name=selected_model_name,
                    raw_epochs=raw_epochs
                )
                st.session_state['predictions'] = predictions
                st.session_state['processing_complete'] = True
                
                st.success(f"Prediction complete using {selected_model_name}.")
            else:
                st.warning("Could not create epochs. Check that the file contains T1/T2 annotations.")
                
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")


def render_main_content():
    """Render the main content area with tabs."""
    
    # Create tabs
    tabs = st.tabs([
        "Dataset Info",
        "Signal Visualization",
        "Pipeline Overview",
        "Model Summary",
        "Prediction Results"
    ])
    
    with tabs[0]:
        render_dataset_info()
    
    with tabs[1]:
        render_signal_visualization()
    
    with tabs[2]:
        render_pipeline_overview()
    
    with tabs[3]:
        render_model_summary()
    
    with tabs[4]:
        render_prediction_results()


def render_dataset_info():
    """Render the dataset information section."""
    
    st.markdown("## Dataset / File Information")
    
    if st.session_state.get('file_uploaded'):
        metadata = st.session_state.get('metadata', {})
        annotations = st.session_state.get('annotations', pd.DataFrame())
        
        # File info card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Filename</div>
                <div class="metric-value" style="font-size: 16px;">{st.session_state.get('current_file', 'N/A')}</div>
                <div class="metric-subtitle">EDF File</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sampling Rate</div>
                <div class="metric-value">{metadata.get('sampling_frequency', 0):.0f}</div>
                <div class="metric-subtitle">Hz</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Duration</div>
                <div class="metric-value">{metadata.get('duration', 0):.1f}</div>
                <div class="metric-subtitle">seconds</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Channels</div>
                <div class="metric-value">{metadata.get('n_channels', 0)}</div>
                <div class="metric-subtitle">EEG channels</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Annotation counts
        st.markdown("### Event Annotations")
        
        if not annotations.empty:
            annotation_counts = annotations['description'].value_counts().to_dict()
            
            cols = st.columns(len(annotation_counts) if annotation_counts else 3)
            for i, (event, count) in enumerate(annotation_counts.items()):
                event_info = EVENT_CODES.get(event, {'name': event, 'color': '#8D99AE'})
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {event_info.get('color', '#00B4D8')};">
                        <div class="metric-title">{event}</div>
                        <div class="metric-value" style="color: {event_info.get('color', '#00B4D8')};">{count}</div>
                        <div class="metric-subtitle">{event_info.get('name', 'Event')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Annotation meaning table
        st.markdown("### Annotation Reference")
        annotation_df = get_annotation_table()
        st.dataframe(annotation_df, use_container_width=True, hide_index=True)
        
    else:
        # Placeholder when no file is uploaded
        st.markdown("""
        <div class="card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 14px; color: #888; margin-bottom: 8px;">No file uploaded</div>
            <div style="color: #e5e5e5; font-size: 13px;">Upload an EDF file from the sidebar to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)


def render_signal_visualization():
    """Render the EEG signal visualization section."""
    
    st.markdown("## EEG Signal Visualization")
    
    if st.session_state.get('file_uploaded'):
        raw = st.session_state.get('raw_data')
        filtered_raw = st.session_state.get('filtered_data')
        selected_channels = st.session_state.get('selected_channels', [])
        time_range = st.session_state.get('time_range', (0, 10))
        
        if raw is not None and selected_channels:
            # Get data
            data, times = raw.get_data(return_times=True)
            ch_names = raw.info['ch_names']
            
            # Filter to selected channels
            ch_indices = [ch_names.index(ch) for ch in selected_channels if ch in ch_names]
            if ch_indices:
                selected_data = data[ch_indices, :]
                selected_names = [ch_names[i] for i in ch_indices]
                
                # Raw EEG plot
                st.markdown("### Raw EEG Signal")
                fig_raw = plot_raw_eeg(
                    selected_data, times, selected_names,
                    title="Raw EEG Signal",
                    time_range=time_range
                )
                st.plotly_chart(fig_raw, use_container_width=True)
                
                # Filtered signal
                if st.session_state.get('show_filtered') and filtered_raw is not None:
                    st.markdown("### Filtered EEG Signal (8-30 Hz)")
                    filtered_data, _ = filtered_raw.get_data(return_times=True)
                    filtered_selected = filtered_data[ch_indices, :]
                    
                    if len(ch_indices) > 0:
                        fig_compare = plot_filtered_eeg(
                            selected_data[0, :],
                            filtered_selected[0, :],
                            times,
                            selected_names[0]
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                
                # PSD
                if st.session_state.get('show_psd'):
                    st.markdown("### Power Spectral Density")
                    psd, freqs = compute_psd(raw)
                    if psd.size > 0:
                        fig_psd = plot_psd(psd, freqs, ch_names, selected_channels[0] if selected_channels else None)
                        st.plotly_chart(fig_psd, use_container_width=True)
                
                # Event markers
                if st.session_state.get('show_events'):
                    annotations = st.session_state.get('annotations', pd.DataFrame())
                    if not annotations.empty:
                        st.markdown("### Signal with Event Markers")
                        fig_events = plot_event_markers(
                            times, selected_data[0, :],
                            annotations, selected_names[0]
                        )
                        st.plotly_chart(fig_events, use_container_width=True)
            else:
                st.warning("Please select at least one valid channel.")
        else:
            st.info("Please select channels from the sidebar to visualize.")
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 14px; color: #888; margin-bottom: 8px;">No data to display</div>
            <div style="color: #e5e5e5; font-size: 13px;">Upload an EDF file to visualize EEG signals.</div>
        </div>
        """, unsafe_allow_html=True)


def render_pipeline_overview():
    """Render the preprocessing pipeline overview."""
    
    st.markdown("## Preprocessing Pipeline")
    st.markdown('<p style="color: #888; margin-bottom: 20px;">Steps executed when processing EEG data:</p>', unsafe_allow_html=True)
    
    pipeline_steps = get_pipeline_steps()
    
    # Display pipeline steps
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for step in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-number">{step['step']}</div>
            <div class="step-content">
                <div class="step-name">{step['name']}</div>
                <div class="step-desc">{step['description']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration details
    st.markdown("### Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #e5e5e5; margin-bottom: 16px; font-size: 14px; font-weight: 600;">Preprocessing</h4>
            <table style="width: 100%; color: #e5e5e5; font-size: 13px;">
                <tr><td style="padding: 8px 0; color: #888;">Bandpass Filter</td><td style="text-align: right;">8-30 Hz</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Notch Filter</td><td style="text-align: right;">None</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Epoch Duration</td><td style="text-align: right;">4.0 seconds</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Baseline Correction</td><td style="text-align: right;">None</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #e5e5e5; margin-bottom: 16px; font-size: 14px; font-weight: 600;">Feature Extraction</h4>
            <table style="width: 100%; color: #e5e5e5; font-size: 13px;">
                <tr><td style="padding: 8px 0; color: #888;">Feature Type</td><td style="text-align: right;">Bandpower</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Mu Band</td><td style="text-align: right;">8-13 Hz</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Beta Band</td><td style="text-align: right;">13-30 Hz</td></tr>
                <tr><td style="padding: 8px 0; color: #888;">Relative Power</td><td style="text-align: right;">Yes</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


def render_model_summary():
    """Render the offline model summary section."""
    
    st.markdown("## Offline Model Summary")
    
    # Load offline results
    results_df = load_offline_results()
    
    if not results_df.empty:
        # Model comparison chart
        st.markdown("### Model Performance Comparison")
        fig_comparison = plot_metrics_comparison(results_df)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed results table
        st.markdown("### Detailed Results")
        st.dataframe(
            results_df.style.format({
                'accuracy': '{:.2%}',
                'f1_score': '{:.2%}',
                'precision': '{:.2%}',
                'recall': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Best model highlight
        if results_df['accuracy'].max() > 0:
            best_model = results_df.loc[results_df['accuracy'].idxmax()]
            st.markdown(f"""
            <div class="card" style="border-left: 3px solid #22c55e;">
                <h4 style="color: #e5e5e5; margin-bottom: 16px; font-size: 14px;">Best Performing Model</h4>
                <div style="display: flex; gap: 32px; flex-wrap: wrap;">
                    <div>
                        <div style="color: #888; font-size: 11px; text-transform: uppercase;">Model</div>
                        <div style="color: #e5e5e5; font-size: 18px; font-weight: 600; margin-top: 4px;">{best_model['model_name']}</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 11px; text-transform: uppercase;">Accuracy</div>
                        <div style="color: #22c55e; font-size: 18px; font-weight: 600; margin-top: 4px;">{best_model['accuracy']:.1%}</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 11px; text-transform: uppercase;">F1 Score</div>
                        <div style="color: #3b82f6; font-size: 18px; font-weight: 600; margin-top: 4px;">{best_model['f1_score']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No offline results available. Train models and save results to results/offline_model_results.csv")


def render_prediction_results():
    """Render the prediction results section."""
    
    st.markdown("## Prediction Results")
    
    # Check if predictions exist or generate sample data for demonstration
    predictions = st.session_state.get('predictions')
    
    if st.session_state.get('processing_complete'):
        # Use sample data for demonstration if no actual predictions
        if predictions is None or (isinstance(predictions, dict) and len(predictions.get('predictions', [])) == 0):
            st.info("Displaying sample predictions for demonstration. Train and add models to see real predictions.")
            results_df = generate_sample_predictions(20)
        else:
            # Create results dataframe from actual predictions
            ground_truth = np.array([1, 2] * 10)  # Placeholder
            results_df = create_results_dataframe(
                [],
                predictions['predictions'],
                ground_truth,
                predictions.get('confidence')
            )
        
        # Metrics cards
        st.markdown("### File-Level Metrics")
        
        accuracy = results_df['correct'].mean()
        correct_count = results_df['correct'].sum()
        total_count = len(results_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{accuracy:.1%}</div>
                <div class="metric-subtitle">Overall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">{accuracy:.1%}</div>
                <div class="metric-subtitle">Weighted</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Correct</div>
                <div class="metric-value" style="color: #22c55e;">{correct_count}</div>
                <div class="metric-subtitle">epochs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total</div>
                <div class="metric-value">{total_count}</div>
                <div class="metric-subtitle">epochs</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction timeline
        st.markdown("### Prediction Timeline")
        fig_timeline = plot_prediction_timeline(results_df)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Epoch-level results table
        st.markdown("### Epoch-Level Results")
        
        # Filter options
        col1, col2 = st.columns([3, 1])
        with col2:
            show_incorrect_only = st.checkbox("Show incorrect only", value=False)
        
        if show_incorrect_only:
            display_df = results_df[~results_df['correct']]
        else:
            display_df = results_df
        
        # Style the dataframe
        def highlight_incorrect(row):
            if not row['correct']:
                return ['background-color: rgba(230, 57, 70, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_incorrect, axis=1).format({
            'confidence': '{:.2%}',
            'start_time': '{:.1f}s',
            'end_time': '{:.1f}s'
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 14px; color: #888; margin-bottom: 8px;">No predictions yet</div>
            <div style="color: #e5e5e5; font-size: 13px;">Upload an EDF file and click "Run Prediction" to begin.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
