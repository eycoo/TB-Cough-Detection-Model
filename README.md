# TB Cough Detection System

> **Gold Medal — Software Development Division, Gemastik XVIII (2025)**

Binary classification system for Tuberculosis detection from cough audio recordings. Extracts MFCC features from cough sounds and classifies them as TB or Non-TB using a stacked LSTM architecture.

---

## Model Performance

Evaluated on a held-out test set (10% of 2244 samples):

| Metric | Score |
|---|---|
| Accuracy | 83.91% |
| Precision | 81.27% |
| Recall | 84.79% |
| F1 Score | 83.00% |

**Dataset**: 2244 samples (1205 Non-TB, 1039 TB)
**Split**: 80% train / 10% validation / 10% test

---

## Architecture

```
Raw Audio (16 kHz WAV)
    ↓
Cough Segmentation (spectral power threshold)
    ↓
MFCC Extraction (13 coefficients + delta + delta-delta = 39 features)
    ↓
Normalization & Padding/Truncation → 63 frames
    ↓
LSTM Layer 1 (512 hidden units) + BatchNorm
    ↓
LSTM Layer 2 (512 hidden units) + BatchNorm
    ↓
Dropout (0.1) → Fully Connected
    ↓
Output: TB / Non-TB
```

---

## Files

| File | Description |
|---|---|
| `train.ipynb` | Full pipeline: data loading, MFCC extraction, training, evaluation, inference |
| `tb_lstm2.py` | Standalone training script |

---

## Requirements

```bash
pip install torch torchaudio librosa numpy pandas scikit-learn soundfile tqdm matplotlib seaborn scipy
```

Python 3.10+ recommended.

---

## Usage

### Training

Configure dataset paths in the `Config` class, then run:

```bash
python tb_lstm2.py
```

Default training parameters:

| Parameter | Value |
|---|---|
| Epochs | 23 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | 10 epochs |
| Random Seed | 42 |

### Inference

```python
from tb_lstm2 import Config, LSTMAudioClassifierMFCC, CombinedDataset, load_checkpoint
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()

model = LSTMAudioClassifierMFCC(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    output_size=config.output_size,
    dropout=config.dropout
).to(device)
model = load_checkpoint(config.model_path, model)
model.eval()

sample_mfcc = CombinedDataset(config).preprocess_audio_mfcc("path/to/audio.wav", target_length=63)
sample_mfcc = sample_mfcc.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_mfcc)
    probs = torch.softmax(output, dim=1).cpu().numpy()
    predicted = np.argmax(probs)
    confidence = probs[0][predicted]

print(f"Prediction: {'TB' if predicted == 1 else 'Non-TB'}")
print(f"Confidence: {confidence:.2f}")
```

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| `sampling_rate` | 16000 Hz | Audio sample rate |
| `desired_length` | 1.0 s | Target audio length |
| `n_mfcc` | 13 | Base MFCC coefficients |
| `n_mels` | 40 | Mel filterbank count |
| `fmin / fmax` | 60 / 6000 Hz | MFCC frequency range |
| `filter_length` | 512 | FFT window size |
| `hop_length` | 256 | STFT hop length |
| `hidden_size` | 512 | LSTM hidden units |
| `dropout` | 0.1 | Dropout rate |

---

## Dataset

Two audio sources used for training:

1. **CIDRZ Dataset** — clinical cough recordings with ground truth TB labels
2. **Forced Coughs Dataset** — labeled TB/Non-TB forced cough recordings

Dataset not included in this repository due to medical licensing. Contact the data owners for access.

---

## Reproducibility

Random seed fixed at `42` across `random`, `numpy`, `torch`, CUDA, and `PYTHONHASHSEED`. Deterministic CUDA ops enabled where available.

---

## Tech Stack

Python · PyTorch · Librosa · scikit-learn · NumPy
