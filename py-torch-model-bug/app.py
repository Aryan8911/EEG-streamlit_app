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
    """Inject custom CSS for modern, polished UI with fluid animations."""
    st.markdown("""
    <style>
    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #00B4D8;
        --secondary: #7B2CBF;
        --accent: #00F5D4;
        --background: #0D1B2A;
        --surface: #1B263B;
        --surface-light: #415A77;
        --text-primary: #E0E1DD;
        --text-secondary: #8D99AE;
        --success: #2EC4B6;
        --error: #E63946;
        --warning: #F4A261;
    }
    
    /* ===== KEYFRAME ANIMATIONS ===== */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-10px) rotate(1deg); }
        66% { transform: translateY(5px) rotate(-1deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 180, 216, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 180, 216, 0.6), 0 0 60px rgba(123, 44, 191, 0.3); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes brainWave {
        0% { transform: scaleY(1); }
        25% { transform: scaleY(1.2); }
        50% { transform: scaleY(0.8); }
        75% { transform: scaleY(1.1); }
        100% { transform: scaleY(1); }
    }
    
    @keyframes neuralPulse {
        0%, 100% { 
            filter: drop-shadow(0 0 8px rgba(0, 180, 216, 0.5));
            opacity: 0.8;
        }
        50% { 
            filter: drop-shadow(0 0 20px rgba(0, 245, 212, 0.8));
            opacity: 1;
        }
    }
    
    @keyframes dataStream {
        0% { transform: translateX(-100%); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -20px) scale(1.1); }
        50% { transform: translate(-20px, 20px) scale(0.9); }
        75% { transform: translate(10px, -10px) scale(1.05); }
    }
    
    /* ===== MAIN CONTAINER WITH ANIMATED BG ===== */
    .stApp {
        background: linear-gradient(-45deg, #0D1B2A, #1B263B, #0D1B2A, #162032);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        position: relative;
    }
    
    /* ===== FLOATING ORBS BACKGROUND ===== */
    .floating-orbs {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.4;
        animation: orbFloat 20s ease-in-out infinite;
    }
    
    .orb-1 {
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 180, 216, 0.4) 0%, transparent 70%);
        top: 10%;
        left: 10%;
        animation-delay: 0s;
    }
    
    .orb-2 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(123, 44, 191, 0.3) 0%, transparent 70%);
        top: 60%;
        right: 15%;
        animation-delay: -5s;
    }
    
    .orb-3 {
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(0, 245, 212, 0.3) 0%, transparent 70%);
        bottom: 10%;
        left: 30%;
        animation-delay: -10s;
    }
    
    /* ===== NEURAL NETWORK LINES ===== */
    .neural-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        opacity: 0.1;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(0, 180, 216, 0.3) 1px, transparent 1px),
            radial-gradient(circle at 80% 70%, rgba(123, 44, 191, 0.3) 1px, transparent 1px),
            radial-gradient(circle at 50% 50%, rgba(0, 245, 212, 0.2) 1px, transparent 1px);
        background-size: 100px 100px, 150px 150px, 80px 80px;
        animation: gradientShift 30s linear infinite;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(27, 38, 59, 0.95) 0%, rgba(13, 27, 42, 0.98) 100%);
        border-right: 1px solid rgba(0, 180, 216, 0.2);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00B4D8, #7B2CBF, #00F5D4);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #E0E1DD;
    }
    
    /* ===== HEADERS ===== */
    h1 {
        background: linear-gradient(90deg, #00B4D8 0%, #00F5D4 50%, #7B2CBF 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -1px;
        animation: shimmer 4s linear infinite;
    }
    
    h2, h3 {
        color: #E0E1DD !important;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, rgba(0, 180, 216, 0.5), transparent) 1;
        padding-bottom: 10px;
        position: relative;
    }
    
    h2::after, h3::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #00B4D8, #7B2CBF);
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(27, 38, 59, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 180, 216, 0.15);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(0, 180, 216, 0.1),
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 180, 216, 0.4);
        box-shadow: 
            0 12px 40px rgba(0, 180, 216, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-4px);
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(27, 38, 59, 0.8) 0%, rgba(13, 27, 42, 0.9) 100%);
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00B4D8, #7B2CBF);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--primary);
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 180, 216, 0.2);
    }
    
    .metric-card:hover::after {
        transform: scaleX(1);
    }
    
    .metric-title {
        color: #8D99AE;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-weight: 500;
    }
    
    .metric-value {
        color: #00B4D8;
        font-size: 36px;
        font-weight: 700;
        margin: 8px 0;
        text-shadow: 0 0 30px rgba(0, 180, 216, 0.3);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .metric-subtitle {
        color: #E0E1DD;
        font-size: 11px;
        opacity: 0.8;
    }
    
    /* ===== LIVE INDICATOR ===== */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(46, 196, 182, 0.15);
        border: 1px solid rgba(46, 196, 182, 0.4);
        border-radius: 20px;
        font-size: 12px;
        color: #2EC4B6;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #2EC4B6;
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
        box-shadow: 0 0 10px #2EC4B6;
    }
    
    /* ===== PIPELINE STEPS ===== */
    .pipeline-step {
        background: rgba(27, 38, 59, 0.5);
        border: 1px solid rgba(0, 245, 212, 0.15);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        gap: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .pipeline-step::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #00B4D8, #7B2CBF);
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    
    .pipeline-step:hover {
        border-color: var(--accent);
        background: rgba(0, 245, 212, 0.05);
        transform: translateX(8px);
    }
    
    .pipeline-step:hover::before {
        transform: scaleY(1);
    }
    
    .step-number {
        background: linear-gradient(135deg, #00B4D8, #7B2CBF);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        flex-shrink: 0;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
        animation: float 6s ease-in-out infinite;
    }
    
    .step-content {
        flex: 1;
    }
    
    .step-name {
        color: #E0E1DD;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .step-desc {
        color: #8D99AE;
        font-size: 12px;
        line-height: 1.4;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: rgba(27, 38, 59, 0.4);
        border: 2px dashed rgba(0, 180, 216, 0.3);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stFileUploader"]::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(45deg, 
            transparent 40%, 
            rgba(0, 180, 216, 0.1) 50%, 
            transparent 60%);
        background-size: 200% 200%;
        animation: shimmer 3s linear infinite;
        pointer-events: none;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: rgba(0, 180, 216, 0.08);
        transform: scale(1.01);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #00B4D8 0%, #7B2CBF 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 180, 216, 0.4);
        animation: glow 2s ease-in-out infinite;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* ===== SELECTBOX & INPUTS ===== */
    .stSelectbox > div > div {
        background: rgba(27, 38, 59, 0.7);
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 10px;
        color: #E0E1DD;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(0, 180, 216, 0.5);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00B4D8, #7B2CBF);
        border-radius: 10px;
    }
    
    .stMultiSelect > div > div {
        background: rgba(27, 38, 59, 0.7);
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stMultiSelect > div > div:hover {
        border-color: rgba(0, 180, 216, 0.5);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(27, 38, 59, 0.4);
        border-radius: 16px;
        padding: 6px;
        gap: 6px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8D99AE;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 216, 0.1);
        color: #E0E1DD;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.25), rgba(123, 44, 191, 0.25));
        color: #E0E1DD;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.2);
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        background: rgba(27, 38, 59, 0.4);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(0, 180, 216, 0.1);
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(27, 38, 59, 0.6);
        border-radius: 12px;
        color: #E0E1DD;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 180, 216, 0.1);
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-color: #00B4D8 transparent transparent transparent;
    }
    
    /* ===== HERO SECTION ===== */
    .hero-section {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.08) 0%, rgba(123, 44, 191, 0.08) 100%);
        border: 1px solid rgba(0, 180, 216, 0.15);
        border-radius: 24px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 180, 216, 0.1) 0%, transparent 50%);
        animation: orbFloat 15s ease-in-out infinite;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00B4D8, #00F5D4, #7B2CBF, #00B4D8);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        animation: shimmer 4s linear infinite;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: #8D99AE;
        font-size: 1.1rem;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    
    .hero-description {
        color: #E0E1DD;
        font-size: 0.95rem;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }
    
    /* ===== BRAIN WAVE ANIMATION ===== */
    .brain-wave-container {
        display: flex;
        justify-content: center;
        gap: 4px;
        margin: 20px 0;
    }
    
    .brain-wave-bar {
        width: 4px;
        height: 30px;
        background: linear-gradient(180deg, #00B4D8, #7B2CBF);
        border-radius: 2px;
        animation: brainWave 1s ease-in-out infinite;
    }
    
    .brain-wave-bar:nth-child(1) { animation-delay: 0s; }
    .brain-wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .brain-wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .brain-wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .brain-wave-bar:nth-child(5) { animation-delay: 0.4s; }
    .brain-wave-bar:nth-child(6) { animation-delay: 0.5s; }
    .brain-wave-bar:nth-child(7) { animation-delay: 0.4s; }
    .brain-wave-bar:nth-child(8) { animation-delay: 0.3s; }
    .brain-wave-bar:nth-child(9) { animation-delay: 0.2s; }
    .brain-wave-bar:nth-child(10) { animation-delay: 0.1s; }
    
    /* ===== STATUS BADGES ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-success {
        background: rgba(46, 196, 182, 0.15);
        color: #2EC4B6;
        border: 1px solid rgba(46, 196, 182, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-warning {
        background: rgba(244, 162, 97, 0.15);
        color: #F4A261;
        border: 1px solid rgba(244, 162, 97, 0.3);
    }
    
    .status-error {
        background: rgba(230, 57, 70, 0.15);
        color: #E63946;
        border: 1px solid rgba(230, 57, 70, 0.3);
    }
    
    /* ===== DATA STREAM EFFECT ===== */
    .data-stream {
        position: relative;
        overflow: hidden;
        padding: 20px;
        background: rgba(27, 38, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(0, 180, 216, 0.2);
    }
    
    .data-stream::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        width: 100px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00B4D8, #7B2CBF, transparent);
        animation: dataStream 3s linear infinite;
    }
    
    /* ===== ANNOTATION TABLE ===== */
    .annotation-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 16px 0;
    }
    
    .annotation-table th {
        background: rgba(0, 180, 216, 0.15);
        color: #E0E1DD;
        padding: 14px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid rgba(0, 180, 216, 0.2);
    }
    
    .annotation-table td {
        background: rgba(27, 38, 59, 0.4);
        color: #E0E1DD;
        padding: 14px;
        border-bottom: 1px solid rgba(0, 180, 216, 0.08);
        transition: all 0.3s ease;
    }
    
    .annotation-table tr:hover td {
        background: rgba(0, 180, 216, 0.1);
    }
    
    /* ===== NEURAL ICON ===== */
    .neural-icon {
        font-size: 3rem;
        animation: neuralPulse 2s ease-in-out infinite;
    }
    
    /* ===== PROGRESS RING ===== */
    .progress-ring {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #00B4D8 0%, #7B2CBF 70%, rgba(27, 38, 59, 0.5) 70%);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: float 4s ease-in-out infinite;
    }
    
    .progress-ring-inner {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #1B263B;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #E0E1DD;
        font-weight: 700;
        font-size: 14px;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1B263B;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00B4D8, #7B2CBF);
        border-radius: 5px;
        border: 2px solid #1B263B;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00F5D4, #00B4D8);
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 28px;
        }
        
        .orb {
            opacity: 0.2;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inject animated background elements
    st.markdown("""
    <div class="floating-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    <div class="neural-bg"></div>
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
    """Render the hero header section with animated elements."""
    st.markdown("""
    <div class="hero-section">
        <div class="neural-icon">🧠</div>
        <div class="brain-wave-container">
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
            <div class="brain-wave-bar"></div>
        </div>
        <div class="hero-title">EEG Motor Imagery Prediction</div>
        <div class="hero-subtitle">PhysioNet EEG Motor Movement/Imagery Dataset — Left vs Right Hand Classification</div>
        <div style="margin: 20px 0;">
            <span class="live-indicator">
                <span class="live-dot"></span>
                Ready for Analysis
            </span>
        </div>
        <div class="hero-description">
            An advanced biomedical AI dashboard for analyzing EEG signals and classifying motor imagery tasks.
            Upload your EDF files to visualize brain activity and predict imagined hand movements using state-of-the-art machine learning models.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls."""
    
    with st.sidebar:
        # Logo/Branding with animated elements
        st.markdown("""
        <div style="text-align: center; padding: 24px 0; border-bottom: 1px solid rgba(0,180,216,0.2); margin-bottom: 24px; position: relative;">
            <div class="neural-icon" style="font-size: 2.5rem; margin-bottom: 8px;">🧠</div>
            <div class="brain-wave-container" style="margin: 12px 0;">
                <div class="brain-wave-bar" style="height: 20px;"></div>
                <div class="brain-wave-bar" style="height: 20px;"></div>
                <div class="brain-wave-bar" style="height: 20px;"></div>
                <div class="brain-wave-bar" style="height: 20px;"></div>
                <div class="brain-wave-bar" style="height: 20px;"></div>
            </div>
            <div style="color: #00B4D8; font-weight: 700; font-size: 1.2rem; letter-spacing: -0.5px;">EEG Analysis</div>
            <div style="color: #8D99AE; font-size: 0.8rem; margin-top: 4px;">Motor Imagery Dashboard</div>
            <div style="margin-top: 12px;">
                <span class="live-indicator" style="font-size: 10px; padding: 4px 10px;">
                    <span class="live-dot" style="width: 6px; height: 6px;"></span>
                    System Active
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### 📁 Upload EDF File")
        uploaded_file = st.file_uploader(
            "Drag and drop your EDF file here",
            type=['edf'],
            help="Upload an EDF file from the PhysioNet Motor Imagery Dataset"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file)
        
        st.markdown("---")
        
        # EEG Controls
        st.markdown("### 🎛️ EEG Controls")
        
        # Channel selector
        available_channels = st.session_state.get('available_channels', MI_CHANNELS)
        selected_channels = st.multiselect(
            "Select Channels",
            options=available_channels,
            default=available_channels[:2] if len(available_channels) >= 2 else available_channels,
            help="Choose EEG channels to display"
        )
        st.session_state['selected_channels'] = selected_channels
        
        # Time range slider
        max_duration = st.session_state.get('recording_duration', 60.0)
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
        st.markdown("### 📊 Visualization Options")
        
        show_filtered = st.checkbox("Show Filtered Signal", value=True)
        show_psd = st.checkbox("Show PSD Analysis", value=True)
        show_events = st.checkbox("Show Event Markers", value=True)
        
        st.session_state['show_filtered'] = show_filtered
        st.session_state['show_psd'] = show_psd
        st.session_state['show_events'] = show_events
        
        st.markdown("---")
        
        # Model Controls
        st.markdown("### 🤖 Model Controls")
        
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
        if st.button("🚀 Run Prediction", use_container_width=True):
            run_prediction_pipeline()
        
        st.markdown("---")
        
        # Info Section with data stream effect
        st.markdown("""
        <div class="data-stream" style="margin-top: 20px;">
            <div style="color: #00B4D8; font-weight: 600; font-size: 12px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 8px; height: 8px; background: linear-gradient(135deg, #00B4D8, #7B2CBF); border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></span>
                ABOUT THIS TOOL
            </div>
            <div style="color: #8D99AE; font-size: 11px; line-height: 1.6;">
                This application analyzes motor imagery EEG data for binary classification between left (T1) and right (T2) hand imagery tasks using advanced signal processing and machine learning.
            </div>
            <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-badge status-success" style="font-size: 9px; padding: 3px 8px;">ML Ready</span>
                <span class="status-badge" style="font-size: 9px; padding: 3px 8px; background: rgba(123, 44, 191, 0.15); color: #7B2CBF; border: 1px solid rgba(123, 44, 191, 0.3);">8-30 Hz Filter</span>
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
        "📋 Dataset Info",
        "📈 Signal Visualization",
        "⚙️ Pipeline Overview",
        "📊 Model Summary",
        "🎯 Prediction Results"
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
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
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
        <div class="glass-card" style="text-align: center; padding: 80px 40px;">
            <div class="neural-icon" style="margin-bottom: 20px;">📁</div>
            <div class="brain-wave-container" style="margin: 20px auto; opacity: 0.5;">
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
            </div>
            <div style="color: #E0E1DD; font-size: 1.3rem; margin-bottom: 12px; font-weight: 600;">No File Uploaded</div>
            <div style="color: #8D99AE; max-width: 400px; margin: 0 auto; line-height: 1.6;">Upload an EDF file using the sidebar to view dataset information and begin analysis.</div>
            <div style="margin-top: 20px;">
                <span class="status-badge status-warning">Awaiting Data</span>
            </div>
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
        <div class="glass-card" style="text-align: center; padding: 80px 40px;">
            <div class="neural-icon" style="margin-bottom: 20px;">📈</div>
            <div class="brain-wave-container" style="margin: 20px auto; opacity: 0.5;">
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
            </div>
            <div style="color: #E0E1DD; font-size: 1.3rem; margin-bottom: 12px; font-weight: 600;">No Data to Display</div>
            <div style="color: #8D99AE; max-width: 400px; margin: 0 auto; line-height: 1.6;">Upload an EDF file to visualize EEG signals with interactive charts and event markers.</div>
            <div style="margin-top: 20px;">
                <span class="status-badge status-warning">Awaiting Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_pipeline_overview():
    """Render the preprocessing pipeline overview."""
    
    st.markdown("## Preprocessing Pipeline")
    st.markdown("""
    <div style="color: #8D99AE; margin-bottom: 24px; display: flex; align-items: center; gap: 12px;">
        <span class="live-indicator">
            <span class="live-dot"></span>
            Pipeline Ready
        </span>
        <span style="font-size: 14px;">The following steps are executed when processing EEG data:</span>
    </div>
    """, unsafe_allow_html=True)
    
    pipeline_steps = get_pipeline_steps()
    
    # Display pipeline as connected cards
    cols = st.columns(4)
    
    for i, step in enumerate(pipeline_steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-number">{step['step']}</div>
                <div class="step-content">
                    <div class="step-name">{step['icon']} {step['name']}</div>
                    <div class="step-desc">{step['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Configuration details
    st.markdown("### Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: radial-gradient(circle, rgba(0, 180, 216, 0.1) 0%, transparent 70%); pointer-events: none;"></div>
            <h4 style="color: #00B4D8; margin-bottom: 18px; display: flex; align-items: center; gap: 10px;">
                <span style="display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: linear-gradient(135deg, rgba(0, 180, 216, 0.2), rgba(0, 180, 216, 0.1)); border-radius: 8px;">⚡</span>
                Preprocessing Settings
            </h4>
            <table style="width: 100%; color: #E0E1DD;">
                <tr><td style="padding: 10px 0; color: #8D99AE;">Bandpass Filter</td><td style="text-align: right; font-weight: 500;">8-30 Hz</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Notch Filter</td><td style="text-align: right; font-weight: 500;">None</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Epoch Duration</td><td style="text-align: right; font-weight: 500;">4.0 seconds</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Baseline Correction</td><td style="text-align: right; font-weight: 500;">None</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: radial-gradient(circle, rgba(123, 44, 191, 0.1) 0%, transparent 70%); pointer-events: none;"></div>
            <h4 style="color: #7B2CBF; margin-bottom: 18px; display: flex; align-items: center; gap: 10px;">
                <span style="display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: linear-gradient(135deg, rgba(123, 44, 191, 0.2), rgba(123, 44, 191, 0.1)); border-radius: 8px;">🔬</span>
                Feature Extraction
            </h4>
            <table style="width: 100%; color: #E0E1DD;">
                <tr><td style="padding: 10px 0; color: #8D99AE;">Feature Type</td><td style="text-align: right; font-weight: 500;">Bandpower</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Mu Band</td><td style="text-align: right; font-weight: 500;">8-13 Hz</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Beta Band</td><td style="text-align: right; font-weight: 500;">13-30 Hz</td></tr>
                <tr><td style="padding: 10px 0; color: #8D99AE;">Relative Power</td><td style="text-align: right; font-weight: 500;">Yes</td></tr>
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
            <div class="glass-card" style="border-color: rgba(46, 196, 182, 0.4); position: relative; overflow: hidden;">
                <div style="position: absolute; top: -20px; right: -20px; width: 150px; height: 150px; background: radial-gradient(circle, rgba(46, 196, 182, 0.15) 0%, transparent 70%); pointer-events: none;"></div>
                <h4 style="color: #2EC4B6; display: flex; align-items: center; gap: 10px;">
                    <span style="display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; background: linear-gradient(135deg, rgba(46, 196, 182, 0.3), rgba(46, 196, 182, 0.1)); border-radius: 10px; animation: pulse 2s ease-in-out infinite;">🏆</span>
                    Best Performing Model
                </h4>
                <div style="display: flex; justify-content: space-between; margin-top: 20px; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1; min-width: 120px;">
                        <div style="color: #8D99AE; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Model</div>
                        <div style="color: #E0E1DD; font-size: 20px; font-weight: 600; margin-top: 4px;">{best_model['model_name']}</div>
                    </div>
                    <div style="flex: 1; min-width: 120px;">
                        <div style="color: #8D99AE; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Accuracy</div>
                        <div style="color: #2EC4B6; font-size: 20px; font-weight: 600; margin-top: 4px; text-shadow: 0 0 20px rgba(46, 196, 182, 0.3);">{best_model['accuracy']:.1%}</div>
                    </div>
                    <div style="flex: 1; min-width: 120px;">
                        <div style="color: #8D99AE; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">F1 Score</div>
                        <div style="color: #00B4D8; font-size: 20px; font-weight: 600; margin-top: 4px; text-shadow: 0 0 20px rgba(0, 180, 216, 0.3);">{best_model['f1_score']:.1%}</div>
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
            <div class="metric-card" style="border-color: #7B2CBF;">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value" style="color: #7B2CBF;">{accuracy:.1%}</div>
                <div class="metric-subtitle">Weighted</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2EC4B6;">
                <div class="metric-title">Correct</div>
                <div class="metric-value" style="color: #2EC4B6;">{correct_count}</div>
                <div class="metric-subtitle">epochs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #E63946;">
                <div class="metric-title">Total</div>
                <div class="metric-value" style="color: #E0E1DD;">{total_count}</div>
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
        <div class="glass-card" style="text-align: center; padding: 80px 40px;">
            <div class="neural-icon" style="margin-bottom: 20px;">🎯</div>
            <div class="brain-wave-container" style="margin: 20px auto; opacity: 0.5;">
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
                <div class="brain-wave-bar" style="height: 15px; background: linear-gradient(180deg, #8D99AE, #415A77);"></div>
            </div>
            <div style="color: #E0E1DD; font-size: 1.3rem; margin-bottom: 12px; font-weight: 600;">No Predictions Yet</div>
            <div style="color: #8D99AE; max-width: 400px; margin: 0 auto; line-height: 1.6;">Upload an EDF file and click "Run Prediction" to classify motor imagery tasks with machine learning.</div>
            <div style="margin-top: 20px;">
                <span class="status-badge status-warning">Awaiting Prediction</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
