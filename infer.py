"""
infer.py

Usage:
    python infer.py /path/to/test_audio

Loads pretrained gender and age models, processes each audio file in numeric order,
extracts features, makes predictions, and writes two output files:
  - results.txt: one label (0-3) per line
  - time.txt: per-file processing time (in seconds)

Model files expected in the same directory:
  - stacking_gender_model.joblib
  - male_KNN_baseline_male.joblib
  - female_KNN_pca_female.joblib

"""
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

import os
import sys
import time
import argparse
import joblib
import torch
import torchaudio
import librosa
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_formants(y, sr, n_formants=3, order=16):
    if len(y) < order + 1:
        return np.zeros(n_formants, dtype=float)
    try:
        a = librosa.lpc(y, order=order)
        if not np.isfinite(a).all():
            raise ValueError
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        freqs = np.angle(roots) * (sr / (2 * np.pi))
        freqs = np.sort(freqs)
        if len(freqs) < n_formants:
            freqs = np.pad(freqs, (0, n_formants - len(freqs)), constant_values=0.0)
        return freqs[:n_formants]
    except Exception:
        return np.zeros(n_formants, dtype=float)

def collapse(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0).mean(dim=1)

def extract_gender_feats(path, sr=16000, n_mfcc=13, n_fft=512, hop_length=256, n_mels=32, device=DEVICE):
    wav, file_sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)

    np_wave, _ = librosa.load(path, sr=sr, mono=True)

    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)

    wav = torch.where(wav.abs() < 1e-4, torch.zeros_like(wav), wav)
    feats = []

    freqs = torchaudio.functional.detect_pitch_frequency(
        wav, sample_rate=sr, frame_time=0.01, freq_low=50.0, freq_high=300.0
    )
    freqs = freqs[freqs>0]
    feats.append(float(freqs.mean()) if freqs.numel()>0 else 0.0)
    feats.append(float(freqs.std())  if freqs.numel()>0 else 0.0)

    feats.extend(compute_formants(np_wave, sr).tolist())

    mfcc_tf  = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={'n_fft':n_fft,'hop_length':hop_length,'n_mels':n_mels}
    ).to(device)
    delta_tf = torchaudio.transforms.ComputeDeltas().to(device)
    mfcc   = mfcc_tf(wav)
    mfcc_d = delta_tf(mfcc)
    feats.extend(collapse(mfcc).cpu().tolist())
    feats.extend(collapse(mfcc_d).cpu().tolist())

    sc = librosa.feature.spectral_contrast(y=np_wave, sr=sr)
    ch = librosa.feature.chroma_stft(y=np_wave, sr=sr)
    feats.extend(sc.mean(axis=1).tolist())
    feats.extend(ch.mean(axis=1).tolist())
    return np.array(feats, dtype=np.float32)

N_MFCC         = 40
N_MELS         = 64
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
n_fft      = 512
hop_length = 256
n_mels     = 64

melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
).to(DEVICE)

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=N_MFCC,
    melkwargs={
        'n_fft':       n_fft,
        'hop_length':  hop_length,
        'n_mels':      n_mels,
    }
).to(DEVICE)

def extract_age_feats(path, sr=16000, device=DEVICE):
    wav, file_sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    wav = torch.where(wav.abs()<1e-4, torch.zeros_like(wav), wav)
    mfcc = mfcc_transform(wav)

    melspec = melspec_transform(wav)

    mfcc_mean = mfcc.mean(dim=2).squeeze(0)
    mfcc_var  = mfcc.var(dim=2).squeeze(0)
    spec_mean = melspec.mean(dim=2).squeeze(0)
    spec_var  = melspec.var(dim=2).squeeze(0)

    feat = torch.cat([mfcc_mean, mfcc_var, spec_mean, spec_var], dim=0)

    return feat.cpu().numpy()

def main(data_dir):
    stack_gender    = joblib.load("stacking_gender_model.joblib")
    male_age_model  = joblib.load("male_KNN_baseline_male.joblib")
    female_age_model= joblib.load("female_KNN_pca_female.joblib")

    files = [f for f in os.listdir(data_dir)
            if f.lower().endswith((".wav",".mp3"))]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    results = []
    times   = []

    for fn in files:
        path = os.path.join(data_dir, fn)
        t0   = time.perf_counter()

        g_feats = extract_gender_feats(path)
        a_feats = extract_age_feats(path)

        g = int(stack_gender.predict([g_feats])[0])

        if g == 0:
            a = int(female_age_model.predict([a_feats])[0])
        else:
            a = int(male_age_model.predict([a_feats])[0])

        # combine → 0:Male-20s,1:Female-20s,2:Male-50s,3:Female-50s
        # Female: 0 Male: 1
        # 50s: 0 20s: 1
        if a == 1 and g == 1:
            lbl = 0
        elif a == 1 and g == 0:
            lbl = 1
        elif a == 0 and g == 1:
            lbl = 2
        else:
            lbl = 3
        results.append(str(lbl))

        t1 = time.perf_counter()
        times.append(f"{(t1-t0):.3f}")

    with open("results.txt","w") as f:
        f.write("\n".join(results))
    with open("time.txt","w") as f:
        f.write("\n".join(times))
    print(f"Wrote {len(results)} lines.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run inference on a folder of audio.")
    p.add_argument('data_dir', help='path to test_audio folder')
    args = p.parse_args()
    main(args.data_dir)