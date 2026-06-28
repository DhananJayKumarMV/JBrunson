"""Local MusicGen inference using HuggingFace transformers + Apple MPS.

Runs facebook/musicgen-small locally via the transformers library.
Model (~450 MB) is downloaded once to ~/Documents/MusicGen on first use.

Requires:
  pip install transformers accelerate scipy torch torchaudio
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import threading

import numpy as np

MUSICGEN_HOME = str(Path.home() / "Documents" / "MusicGen")
MODEL_ID = "facebook/musicgen-small"

_model = None
_processor = None
_sample_rate = 32000
_load_lock = threading.Lock()


def _get_model():
    global _model, _processor, _sample_rate
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is None:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            os.environ.setdefault("HF_HOME", MUSICGEN_HOME)

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"[local_musicgen] Loading {MODEL_ID} on {device}...", flush=True)
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
            _sample_rate = _model.config.audio_encoder.sampling_rate
            print(f"[local_musicgen] Ready (sr={_sample_rate}).", flush=True)
    return _model, _processor


def generate(prompt: str, duration: int = 30) -> bytes:
    """Generate background music from a text prompt. Returns WAV bytes (stereo)."""
    import scipy.io.wavfile
    import torch

    model, processor = _get_model()

    # musicgen-small generates ~50 tokens/sec; cap at 30 s
    max_tokens = min(duration, 30) * 50

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

    # audio_values: (batch, channels, samples) — take first item
    wav = audio_values[0].cpu().numpy()  # (channels, samples) or (samples,)

    if wav.ndim == 1:
        wav = np.stack([wav, wav])  # mono → stereo

    # Normalise and convert to int16
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak
    wav_int16 = (wav.clip(-1, 1) * 32767).astype(np.int16).T  # (samples, channels)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, _sample_rate, wav_int16)
    return buf.getvalue()
