"""
Inference Module for EEG Motor Imagery Application.

This module handles loading pretrained models and running inference
on preprocessed EEG features for motor imagery classification.

Includes all PyTorch model architectures (EEGNet, ShallowConvNet, DeepConvNet,
BiLSTM, CNN-Transformer) so that .pth weights can be loaded without external imports.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")

# Default paths for model files
MODELS_DIR = Path(__file__).parent.parent / "models"

# Label mapping
LABEL_MAP = {0: 'Left', 1: 'Right'}

# Default DL config (used if dl_model_config.json is missing)
DEFAULT_DL_CONFIG = {
    "n_channels": 9,
    "n_times": 481,
    "n_classes": 2,
    "eegnet": {"F1": 16, "D": 2, "F2": 32, "drop": 0.5},
    "shallow": {"n_filters": 40, "drop": 0.5},
    "deep": {"drop": 0.5},
    "bilstm": {"hidden": 128, "n_layers": 2, "drop": 0.4},
    "cnn_tf": {"d_model": 64, "n_heads": 4, "n_layers": 2, "drop": 0.2}
}


# =============================================================================
# PyTorch Model Architectures
# =============================================================================

if TORCH_AVAILABLE:

    class EEGNet(nn.Module):
        """
        EEGNet architecture for EEG classification.
        Input shape: (batch, 1, n_channels, n_times)
        
        Uses nn.Sequential blocks with names (b1, b2, b3) to match training notebook state_dict keys.
        """
        def __init__(self, n_channels=9, n_times=481, n_classes=2,
                     F1=16, D=2, F2=32, drop=0.5):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, n_times // 2), padding=(0, n_times // 4), bias=False),
                nn.BatchNorm2d(F1))
            self.b2 = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.b3 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            with torch.no_grad():
                flat = self.b3(self.b2(self.b1(
                    torch.zeros(1, 1, n_channels, n_times)))).view(1, -1).shape[1]
            self.fc = nn.Linear(flat, n_classes)

        def forward(self, x):
            return self.fc(self.b3(self.b2(self.b1(x))).view(x.size(0), -1))


    class ShallowConvNet(nn.Module):
        """
        Shallow ConvNet architecture for EEG classification.
        Input shape: (batch, 1, n_channels, n_times)
        
        Uses attribute names (temporal, spatial, bn, drop, pool, fc) to match training notebook.
        """
        def __init__(self, n_channels=9, n_times=481, n_classes=2,
                     n_filters=40, drop=0.5):
            super().__init__()
            self.temporal = nn.Conv2d(1, n_filters, (1, 25), bias=False)
            self.spatial  = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
            self.bn       = nn.BatchNorm2d(n_filters)
            self.drop     = nn.Dropout(drop)
            self.pool     = nn.AvgPool2d((1, 75), stride=(1, 15))
            with torch.no_grad():
                d = self.pool(self.bn(self.spatial(self.temporal(
                    torch.zeros(1, 1, n_channels, n_times)))) ** 2)
                flat = d.view(1, -1).shape[1]
            self.fc = nn.Linear(flat, n_classes)

        def forward(self, x):
            x = self.temporal(x)
            x = self.spatial(x)
            x = self.bn(x) ** 2
            x = torch.log(self.pool(x).clamp(min=1e-6))
            x = self.drop(x)
            return self.fc(x.view(x.size(0), -1))


    class DeepConvNet(nn.Module):
        """
        Deep ConvNet architecture for EEG classification.
        Input shape: (batch, 1, n_channels, n_times)
        
        Uses nn.Sequential blocks (b0, b1, b2, b3) to match training notebook state_dict keys.
        """
        def __init__(self, n_channels=9, n_times=481, n_classes=2, drop=0.5):
            super().__init__()
            def block(in_f, out_f, k=10, pool=3):
                return nn.Sequential(
                    nn.Conv2d(in_f, out_f, (1, k), bias=False),
                    nn.BatchNorm2d(out_f), nn.ELU(),
                    nn.MaxPool2d((1, pool), stride=(1, 3)), nn.Dropout(drop))
            self.b0 = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), bias=False),
                nn.Conv2d(25, 25, (n_channels, 1), bias=False),
                nn.BatchNorm2d(25), nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(drop))
            self.b1 = block(25, 50)
            self.b2 = block(50, 100)
            self.b3 = block(100, 200)
            with torch.no_grad():
                flat = self.b3(self.b2(self.b1(self.b0(
                    torch.zeros(1, 1, n_channels, n_times))))).view(1, -1).shape[1]
            self.fc = nn.Linear(flat, n_classes)

        def forward(self, x):
            return self.fc(self.b3(self.b2(self.b1(self.b0(x)))).view(x.size(0), -1))


    class BiLSTM(nn.Module):
        """
        Bidirectional LSTM with LayerNorm and self-attention for EEG classification.
        Input shape: (batch, n_times, n_channels)
        
        Uses (ln, lstm, attn, fc) to match training notebook state_dict keys.
        """
        def __init__(self, n_channels=9, n_times=481, n_classes=2,
                     hidden=128, n_layers=2, drop=0.4):
            super().__init__()
            self.ln   = nn.LayerNorm(n_channels)
            self.lstm = nn.LSTM(n_channels, hidden, n_layers, batch_first=True,
                                bidirectional=True,
                                dropout=drop if n_layers > 1 else 0.0)
            self.attn = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
            self.fc   = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            x = self.ln(x)
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn(out), dim=1)
            return self.fc((out * w).sum(1))


    class CNNTransformer(nn.Module):
        """
        CNN + Transformer architecture for EEG classification.
        Input shape: (batch, n_channels, n_times)
        
        Uses (cnn, pos_emb, cls_token, tf, norm, fc) to match training notebook state_dict keys.
        """
        def __init__(self, n_channels=9, n_times=481, n_classes=2,
                     d_model=64, n_heads=4, n_layers=2, drop=0.2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_channels, d_model, 7, padding=3),
                nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(drop),
                nn.Conv1d(d_model, d_model, 5, padding=2),
                nn.BatchNorm1d(d_model), nn.GELU())
            self.pos_emb   = nn.Parameter(torch.randn(1, n_times, d_model) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=drop,
                activation='gelu', batch_first=True, norm_first=True)
            self.tf   = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.fc   = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, n_classes))

        def forward(self, x):
            x   = self.cnn(x).transpose(1, 2)
            x   = x + self.pos_emb[:, :x.size(1), :]
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x   = torch.cat([cls, x], dim=1)
            x   = self.tf(x)
            return self.fc(self.norm(x[:, 0]))


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_models(models_dir: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Scan models/ directory and load everything found.
    
    Returns:
    {
      'models': {
          'Random Forest': sklearn_model,
          'LDA': sklearn_model,
          'MLP': sklearn_model,
          'SVM': sklearn_model,
          'EEGNet': torch_model,
          'ShallowConvNet': torch_model,
          'DeepConvNet': torch_model,
          'BiLSTM': torch_model,
          'CNN-Transformer': torch_model,
      },
      'scaler': StandardScaler | None,
      'metadata': dict | None,
      'dl_config': dict | None,
    }
    
    Gracefully skip any file that fails to load (log warning, continue).
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    else:
        models_dir = Path(models_dir)
    
    result = {
        'models': {},
        'scaler': None,
        'metadata': None,
        'dl_config': DEFAULT_DL_CONFIG.copy(),
        'status': []
    }
    
    # Check if models directory exists
    if not models_dir.exists():
        result['status'].append(f"Models directory not found: {models_dir}")
        return result
    
    # Define model file mappings
    sklearn_models = {
        'rf_model.pkl': 'Random Forest',
        'lda_model.pkl': 'LDA',
        'mlp_model.pkl': 'MLP',
        'svm_model.pkl': 'SVM'
    }
    
    pytorch_models = {
        'eegnet_model.pth': 'EEGNet',
        'shallowconv_model.pth': 'ShallowConvNet',
        'deepconv_model.pth': 'DeepConvNet',
        'bilstm_model.pth': 'BiLSTM',
        'cnn_transformer_model.pth': 'CNN-Transformer'
    }
    
    # Load DL config first
    dl_config_path = models_dir / "dl_model_config.json"
    if dl_config_path.exists():
        try:
            with open(dl_config_path, 'r') as f:
                result['dl_config'] = json.load(f)
            result['status'].append("Loaded DL config successfully")
        except Exception as e:
            result['status'].append(f"Error loading DL config: {e}. Using defaults.")
    
    # Load sklearn models
    for filename, model_name in sklearn_models.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                result['models'][model_name] = model
                result['status'].append(f"Loaded {model_name} successfully")
            except Exception as e:
                result['status'].append(f"Error loading {model_name}: {e}")
    
    # Load scaler
    scaler_path = models_dir / "scaler.pkl"
    if scaler_path.exists():
        try:
            result['scaler'] = joblib.load(scaler_path)
            result['status'].append("Loaded scaler successfully")
        except Exception as e:
            result['status'].append(f"Error loading scaler: {e}")
    
    # Load PyTorch models
    if TORCH_AVAILABLE:
        dl_config = result['dl_config']
        device = torch.device('cpu')
        
        for filename, model_name in pytorch_models.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    # Instantiate model architecture
                    n_channels = dl_config.get('n_channels', 9)
                    n_times = dl_config.get('n_times', 481)
                    n_classes = dl_config.get('n_classes', 2)
                    
                    if model_name == 'EEGNet':
                        cfg = dl_config.get('eegnet', DEFAULT_DL_CONFIG['eegnet'])
                        model = EEGNet(
                            n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                            F1=cfg.get('F1', 16), D=cfg.get('D', 2),
                            F2=cfg.get('F2', 32), drop=cfg.get('drop', 0.5)
                        )
                    elif model_name == 'ShallowConvNet':
                        cfg = dl_config.get('shallow', DEFAULT_DL_CONFIG['shallow'])
                        model = ShallowConvNet(
                            n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                            n_filters=cfg.get('n_filters', 40), drop=cfg.get('drop', 0.5)
                        )
                    elif model_name == 'DeepConvNet':
                        cfg = dl_config.get('deep', DEFAULT_DL_CONFIG['deep'])
                        model = DeepConvNet(
                            n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                            drop=cfg.get('drop', 0.5)
                        )
                    elif model_name == 'BiLSTM':
                        cfg = dl_config.get('bilstm', DEFAULT_DL_CONFIG['bilstm'])
                        model = BiLSTM(
                            n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                            hidden=cfg.get('hidden', 128), n_layers=cfg.get('n_layers', 2),
                            drop=cfg.get('drop', 0.4)
                        )
                    elif model_name == 'CNN-Transformer':
                        cfg = dl_config.get('cnn_tf', DEFAULT_DL_CONFIG['cnn_tf'])
                        model = CNNTransformer(
                            n_channels=n_channels, n_times=n_times, n_classes=n_classes,
                            d_model=cfg.get('d_model', 64), n_heads=cfg.get('n_heads', 4),
                            n_layers=cfg.get('n_layers', 2), drop=cfg.get('drop', 0.2)
                        )
                    else:
                        continue
                    
                    # Load state dict
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    model.eval()
                    model.to(device)
                    
                    result['models'][model_name] = model
                    result['status'].append(f"Loaded {model_name} successfully")
                    
                except Exception as e:
                    result['status'].append(f"Error loading {model_name}: {e}")
    
    # Load metadata
    metadata_path = models_dir / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                result['metadata'] = json.load(f)
            result['status'].append("Loaded metadata successfully")
        except Exception as e:
            result['status'].append(f"Error loading metadata: {e}")
    
    return result


def predict_epochs(
    features: np.ndarray,
    model: Any,
    scaler: Any = None,
    model_name: str = '',
    raw_epochs: np.ndarray = None
) -> Dict[str, Any]:
    """
    Unified inference for both sklearn and PyTorch models.
    
    Parameters:
    - features: (n_epochs, 38) band-power features — used for sklearn models
    - raw_epochs: (n_epochs, n_channels, n_times) raw array — used for DL models
                  If None and model is DL, returns empty result gracefully.
    - model_name: Name of the model for routing and scaling decisions
    
    Returns:
    {
      'predictions': np.ndarray (int, 0=Left, 1=Right),
      'confidence':  np.ndarray (float, 0-1),
      'label_names': list of str ('Left' or 'Right'),
    }
    """
    result = {
        'predictions': np.array([]),
        'confidence': np.array([]),
        'label_names': []
    }
    
    # Check if this is a PyTorch model
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        # Deep learning model path
        if raw_epochs is None or len(raw_epochs) == 0:
            return result
        
        preds, probs = predict_raw_epochs_dl(raw_epochs, model, model_name, torch.device('cpu'))
        if len(preds) == 0:
            return result
        
        result['predictions'] = preds.astype(int)
        result['confidence'] = probs.max(axis=1) if probs.ndim == 2 else probs
        result['label_names'] = [LABEL_MAP.get(int(p), str(p)) for p in preds]
        return result
    
    # Sklearn model path
    if features is None or len(features) == 0:
        return result
    
    try:
        X = features.copy()
        
        # Apply scaling for models that need it (LDA, MLP, SVM)
        needs_scaling = model_name in ['LDA', 'MLP', 'SVM']
        if needs_scaling and scaler is not None:
            X = scaler.transform(X)
        
        # Get predictions (sklearn models)
        predictions = model.predict(X)
        result['predictions'] = predictions.astype(int)
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            # Confidence is the probability of the predicted class
            result['confidence'] = np.max(probabilities, axis=1)
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(X)
            # Convert decision function to pseudo-probability
            result['confidence'] = 1 / (1 + np.exp(-np.abs(decision)))
        else:
            result['confidence'] = np.ones(len(predictions))
        
        # Map to label names
        result['label_names'] = [LABEL_MAP.get(p, str(p)) for p in result['predictions']]
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return result


def predict_raw_epochs_dl(
    raw_epochs: np.ndarray,
    model: Any,
    model_name: str,
    device: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dedicated DL inference.
    raw_epochs shape: (n_epochs, n_channels, n_times) - already normalised.
    
    Returns (predictions_array, probabilities_array).
    """
    if not TORCH_AVAILABLE:
        return np.array([]), np.array([])
    
    if device is None:
        device = torch.device('cpu')
    
    try:
        model.eval()
        
        # Apply per-epoch normalisation
        # z-score over time axis per channel
        mean = raw_epochs.mean(axis=-1, keepdims=True)
        std = raw_epochs.std(axis=-1, keepdims=True)
        X_norm = (raw_epochs - mean) / (std + 1e-8)
        
        # Reshape based on model type
        if model_name in ['EEGNet', 'ShallowConvNet', 'DeepConvNet']:
            # Shape: (batch, 1, n_channels, n_times)
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(1).to(device)
        elif model_name == 'BiLSTM':
            # Shape: (batch, n_times, n_channels) - transpose axes 1 and 2
            X_tensor = torch.FloatTensor(X_norm).permute(0, 2, 1).to(device)
        elif model_name == 'CNN-Transformer':
            # Shape: (batch, n_channels, n_times) - standard
            X_tensor = torch.FloatTensor(X_norm).to(device)
        else:
            X_tensor = torch.FloatTensor(X_norm).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Error during DL prediction: {e}")
        return np.array([]), np.array([])


def evaluate_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Returns dict: accuracy, f1, precision, recall, confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    )
    
    if len(predictions) == 0 or len(ground_truth) == 0:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }
    
    try:
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'f1': f1_score(ground_truth, predictions, average='weighted', zero_division=0),
            'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(ground_truth, predictions)
        }
        return metrics
        
    except Exception as e:
        print(f"Error evaluating predictions: {e}")
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }


def create_results_dataframe(
    epoch_onsets: List[float],
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence: np.ndarray = None
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    epoch, start_time, end_time, ground_truth, prediction, correct, confidence
    """
    n_epochs = len(predictions)
    
    # Generate default onsets if not provided
    if epoch_onsets is None or len(epoch_onsets) == 0:
        epoch_onsets = [i * 4.0 for i in range(n_epochs)]
    
    # Generate default confidence if not provided
    if confidence is None or len(confidence) == 0:
        confidence = np.ones(n_epochs)
    
    # Ensure arrays are the right length
    epoch_onsets = list(epoch_onsets)[:n_epochs]
    while len(epoch_onsets) < n_epochs:
        epoch_onsets.append(epoch_onsets[-1] + 4.0 if epoch_onsets else 0.0)
    
    # Create DataFrame
    results = pd.DataFrame({
        'epoch': list(range(n_epochs)),
        'start_time': epoch_onsets,
        'end_time': [t + 3.0 for t in epoch_onsets],  # tmin=0.5, tmax=3.5 -> 3s duration
        'ground_truth': [LABEL_MAP.get(gt, str(gt)) for gt in ground_truth],
        'prediction': [LABEL_MAP.get(p, str(p)) for p in predictions],
        'correct': predictions == ground_truth,
        'confidence': confidence[:n_epochs]
    })
    
    return results
