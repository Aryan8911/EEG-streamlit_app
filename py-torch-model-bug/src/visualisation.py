"""
Visualization Module for EEG Motor Imagery Application.

This module contains Plotly-based visualization functions for
displaying EEG signals, spectral analysis, and prediction results.

All plots use dark theme matching the app CSS.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any


# Color palette for consistent styling (matching app CSS)
COLORS = {
    'primary': '#00B4D8',      # Cyan
    'secondary': '#7B2CBF',    # Deep purple
    'accent': '#00F5D4',       # Teal
    'background': '#0D1B2A',   # Dark blue
    'surface': '#1B263B',      # Lighter dark
    'left': '#4A90D9',         # Left class - steelblue
    'right': '#E84040',        # Right class - tomato
    'correct': '#2EC4B6',      # Correct prediction - teal
    'incorrect': '#E63946',    # Incorrect prediction - red
    'text': '#E0E1DD',         # Light gray text
    'muted': '#8D99AE',        # Muted gray
    'grid': 'rgba(255,255,255,0.1)'
}


def plot_raw_eeg(
    data: np.ndarray,
    times: np.ndarray,
    channel_names: List[str],
    title: str = "Raw EEG Signal",
    time_range: Tuple[float, float] = None
) -> go.Figure:
    """
    Multi-channel EEG time-series. One trace per channel, offset vertically.
    Add vertical dashed lines for events if annotations passed.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_times).
    times : np.ndarray
        Time array in seconds.
    channel_names : List[str]
        List of channel names.
    title : str
        Plot title.
    time_range : Tuple[float, float], optional
        Time range to display (start, end) in seconds.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Filter time range
    if time_range is not None:
        mask = (times >= time_range[0]) & (times <= time_range[1])
        times = times[mask]
        data = data[:, mask]
    
    # Plot each channel with offset
    n_channels = len(channel_names)
    offset_scale = np.std(data) * 4 if data.size > 0 else 1
    
    for i, (ch_name, ch_data) in enumerate(zip(channel_names, data)):
        offset = (n_channels - 1 - i) * offset_scale
        fig.add_trace(go.Scatter(
            x=times,
            y=ch_data + offset,
            mode='lines',
            name=ch_name,
            line=dict(width=1, color=COLORS['primary']),
            hovertemplate=f'{ch_name}<br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.2f}}uV<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS['text'])),
        xaxis_title="Time (s)",
        yaxis_title="Channels",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(
            gridcolor=COLORS['grid'], 
            showgrid=False,
            tickmode='array',
            tickvals=[(n_channels - 1 - i) * offset_scale for i in range(n_channels)],
            ticktext=channel_names
        ),
        height=400 + n_channels * 30
    )
    
    return fig


def plot_filtered_eeg(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    times: np.ndarray,
    channel_name: str
) -> go.Figure:
    """
    Overlay raw (grey) and filtered (primary blue) for one channel.
    
    Parameters
    ----------
    raw_signal : np.ndarray
        Raw EEG signal (1D array).
    filtered_signal : np.ndarray
        Filtered EEG signal (1D array).
    times : np.ndarray
        Time array in seconds.
    channel_name : str
        Name of the channel being displayed.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Raw signal (grey, behind)
    fig.add_trace(go.Scatter(
        x=times, 
        y=raw_signal,
        mode='lines',
        name='Raw',
        line=dict(color=COLORS['muted'], width=1),
        opacity=0.5
    ))
    
    # Filtered signal (primary blue, on top)
    fig.add_trace(go.Scatter(
        x=times, 
        y=filtered_signal,
        mode='lines',
        name='Filtered (8-30 Hz)',
        line=dict(color=COLORS['primary'], width=1.5)
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{channel_name} - Raw vs Filtered', 
            font=dict(size=18, color=COLORS['text'])
        ),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (uV)",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        height=350
    )
    
    return fig


def plot_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
    ch_names: List[str],
    selected_channel: str = None
) -> go.Figure:
    """
    Log-scale PSD. Shade mu (8-13 Hz) and beta (13-30 Hz) bands.
    Show selected channel highlighted, others muted.
    
    Parameters
    ----------
    psd : np.ndarray
        PSD data of shape (n_channels, n_freqs).
    freqs : np.ndarray
        Frequency array.
    ch_names : List[str]
        List of channel names.
    selected_channel : str, optional
        Channel to highlight.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Get index of selected channel
    selected_idx = None
    if selected_channel is not None and selected_channel in ch_names:
        selected_idx = ch_names.index(selected_channel)
    
    # Add shaded regions for frequency bands
    # Mu band (8-13 Hz)
    mu_mask = (freqs >= 8) & (freqs <= 13)
    if np.any(mu_mask):
        y_max = np.max(10 * np.log10(psd + 1e-12))
        y_min = np.min(10 * np.log10(psd + 1e-12))
        fig.add_vrect(
            x0=8, x1=13,
            fillcolor='rgba(0, 180, 216, 0.2)',
            line_width=0,
            annotation_text="Mu",
            annotation_position="top left",
            annotation_font_color=COLORS['primary']
        )
    
    # Beta band (13-30 Hz)
    fig.add_vrect(
        x0=13, x1=30,
        fillcolor='rgba(123, 44, 191, 0.2)',
        line_width=0,
        annotation_text="Beta",
        annotation_position="top left",
        annotation_font_color=COLORS['secondary']
    )
    
    # Plot each channel's PSD
    for i, (ch_name, ch_psd) in enumerate(zip(ch_names, psd)):
        psd_db = 10 * np.log10(ch_psd + 1e-12)
        
        if selected_idx is not None:
            if i == selected_idx:
                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=psd_db,
                    mode='lines',
                    name=ch_name,
                    line=dict(color=COLORS['accent'], width=2.5)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=psd_db,
                    mode='lines',
                    name=ch_name,
                    line=dict(color=COLORS['muted'], width=1),
                    opacity=0.3
                ))
        else:
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd_db,
                mode='lines',
                name=ch_name,
                line=dict(width=1.5)
            ))
    
    fig.update_layout(
        title=dict(text="Power Spectral Density", font=dict(size=18, color=COLORS['text'])),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(gridcolor=COLORS['grid'], range=[0, 50]),
        yaxis=dict(gridcolor=COLORS['grid']),
        height=400
    )
    
    return fig


def plot_event_markers(
    times: np.ndarray,
    signal: np.ndarray,
    annotations_df: pd.DataFrame,
    channel_name: str
) -> go.Figure:
    """
    EEG trace with vertical coloured lines for T0/T1/T2 events.
    T0=grey, T1=steelblue, T2=tomato. Add legend.
    
    Parameters
    ----------
    times : np.ndarray
        Time array in seconds.
    signal : np.ndarray
        EEG signal (1D array).
    annotations_df : pd.DataFrame
        DataFrame with onset, duration, description columns.
    channel_name : str
        Name of the channel.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Plot EEG signal
    fig.add_trace(go.Scatter(
        x=times,
        y=signal,
        mode='lines',
        name=channel_name,
        line=dict(color=COLORS['text'], width=1)
    ))
    
    # Color mapping for events
    event_colors = {
        'T0': COLORS['muted'],
        'T1': COLORS['left'],
        'T2': COLORS['right']
    }
    
    # Track which event types have been added to legend
    legend_added = set()
    
    # Add event markers
    for _, row in annotations_df.iterrows():
        onset = row['onset']
        description = str(row['description'])
        
        color = event_colors.get(description, COLORS['primary'])
        show_legend = description not in legend_added
        legend_added.add(description)
        
        # Add vertical line at event onset
        fig.add_vline(
            x=onset,
            line_dash="dash",
            line_color=color,
            line_width=1.5
        )
        
        # Add a trace for legend
        if show_legend:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=f'{description}',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=True
            ))
    
    fig.update_layout(
        title=dict(
            text=f'{channel_name} with Event Markers', 
            font=dict(size=18, color=COLORS['text'])
        ),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (uV)",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        height=400
    )
    
    return fig


def plot_prediction_timeline(results_df: pd.DataFrame) -> go.Figure:
    """
    Colour-coded prediction timeline.
    Row 1 (y=2): Ground Truth — Left=blue, Right=red
    Row 2 (y=1): Prediction   — correct=teal border, incorrect=red border
    X-axis = epoch number.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with epoch, start_time, end_time, ground_truth,
        prediction, and correct columns.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    label_colors = {'Left': COLORS['left'], 'Right': COLORS['right']}
    
    for _, row in results_df.iterrows():
        ep = int(row['epoch'])
        gt = str(row['ground_truth'])
        pred = str(row['prediction'])
        correct = bool(row['correct'])
        confidence = float(row.get('confidence', 1.0))
        
        gt_col = label_colors.get(gt, COLORS['primary'])
        pred_col = label_colors.get(pred, COLORS['primary'])
        border = COLORS['correct'] if correct else COLORS['incorrect']
        
        # Ground truth rectangle (y band 1.6 – 2.4)
        fig.add_shape(
            type='rect',
            x0=ep - 0.45, x1=ep + 0.45, y0=1.6, y1=2.4,
            fillcolor=gt_col, opacity=0.85,
            line=dict(width=0)
        )
        
        # Prediction rectangle (y band 0.6 – 1.4) with correctness border
        fig.add_shape(
            type='rect',
            x0=ep - 0.45, x1=ep + 0.45, y0=0.6, y1=1.4,
            fillcolor=pred_col, opacity=0.85,
            line=dict(color=border, width=2.5)
        )
        
        # Invisible scatter for hover
        fig.add_trace(go.Scatter(
            x=[ep, ep], y=[2.0, 1.0],
            mode='markers',
            marker=dict(size=1, opacity=0),
            hovertemplate=(
                f'<b>Epoch {ep}</b><br>'
                f'Ground Truth: {gt}<br>'
                f'Prediction: {pred}<br>'
                f'Correct: {correct}<br>'
                f'Confidence: {confidence:.1%}<extra></extra>'
            ),
            showlegend=False
        ))
    
    # Legend via dummy traces
    for label, color in [('Left', COLORS['left']), ('Right', COLORS['right'])]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=label
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color=COLORS['correct'], symbol='square'),
        name='Correct border'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color=COLORS['incorrect'], symbol='square'),
        name='Incorrect border'
    ))
    
    fig.update_layout(
        title=dict(text='Prediction Timeline', font=dict(size=18, color=COLORS['text'])),
        xaxis=dict(
            title='Epoch', 
            tickmode='linear', 
            dtick=1,
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[1.0, 2.0],
            ticktext=['Prediction', 'Ground Truth'],
            range=[0.2, 2.8],
            gridcolor=COLORS['grid']
        ),
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation='h', y=1.12),
        height=350,
        showlegend=True
    )
    
    return fig


def plot_metrics_comparison(results_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: one group per model, bars for accuracy/f1/auc.
    Horizontal reference line at 0.5 (chance level).
    Sort by F1 descending.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with model names and metric columns.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Handle different column name variations
    df = results_df.copy()
    
    # Normalise column names
    column_mapping = {
        'model': 'model_name',
        'f1': 'f1_score',
        'auc': 'roc_auc'
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Sort by F1 score descending
    if 'f1_score' in df.columns:
        df = df.sort_values('f1_score', ascending=False)
    
    fig = go.Figure()
    
    metrics = [
        ('accuracy', 'Accuracy', COLORS['primary']),
        ('f1_score', 'F1 Score', COLORS['secondary']),
        ('roc_auc', 'ROC AUC', COLORS['accent'])
    ]
    
    x_labels = df['model_name'].tolist() if 'model_name' in df.columns else df.index.tolist()
    
    for metric, label, color in metrics:
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=x_labels,
                y=df[metric],
                marker_color=color,
                text=[f'{v:.1%}' for v in df[metric]],
                textposition='outside'
            ))
    
    # Add chance level line
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color=COLORS['muted'],
        annotation_text="Chance Level",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text="Model Performance Comparison", font=dict(size=18, color=COLORS['text'])),
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        legend=dict(orientation="h", y=1.1),
        yaxis=dict(gridcolor=COLORS['grid'], range=[0, 1.1]),
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-45),
        height=500
    )
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str] = None
) -> go.Figure:
    """
    Create an interactive confusion matrix heatmap.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix array.
    labels : List[str], optional
        Class labels.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if labels is None:
        labels = ['Left', 'Right']
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale=[[0, COLORS['surface']], [1, COLORS['primary']]],
        showscale=True,
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 24, "color": COLORS['text']},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=18, color=COLORS['text'])),
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        height=400,
        width=450,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_epoch_heatmap(
    epoch_data: np.ndarray,
    times: np.ndarray,
    channel_names: List[str],
    title: str = "Epoch Activity Heatmap"
) -> go.Figure:
    """
    Create a heatmap visualization of epoch activity across channels.
    
    Parameters
    ----------
    epoch_data : np.ndarray
        Epoch data of shape (n_channels, n_times).
    times : np.ndarray
        Time array.
    channel_names : List[str]
        Channel names.
    title : str
        Plot title.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure(data=go.Heatmap(
        z=epoch_data,
        x=times,
        y=channel_names,
        colorscale='Viridis',
        colorbar=dict(title='uV')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS['text'])),
        xaxis_title="Time (s)",
        yaxis_title="Channel",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        height=400
    )
    
    return fig
