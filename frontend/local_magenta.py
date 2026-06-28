"""Local Magenta RT2 inference using Apple MLX (Apple Silicon only).

Uses MagentaRT2SystemMlxfn which loads the pre-exported .mlxfn files
already present at ~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/.

Requires:
  pip install 'magenta-rt[mlx]'
"""

from __future__ import annotations

import io
import wave

import numpy as np

MODEL_SIZE = "mrt2_small"
SAMPLE_RATE = 48000

_system = None
_sessions: dict = {}  # session_id -> StyleEmbedding


def _get_system():
    global _system
    if _system is None:
        from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
        print(f"[local_magenta] Loading {MODEL_SIZE} via MLX...", flush=True)
        _system = MagentaRT2SystemMlxfn(size=MODEL_SIZE)
        print("[local_magenta] Ready.", flush=True)
    return _system


def _pcm_to_wav(samples: np.ndarray) -> bytes:
    """Convert float32 (N, 2) stereo array to WAV bytes.

    Applies a short fade-in and fade-out so the browser loop point is smooth:
    without this the abrupt onset at loop-restart and any non-zero amplitude
    at loop-end create an audible click or pop.
    """
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak

    n = len(samples)
    fade_in_samples  = min(int(SAMPLE_RATE * 0.05),  n // 8)  # 50 ms
    fade_out_samples = min(int(SAMPLE_RATE * 0.15), n // 8)   # 150 ms
    if fade_in_samples > 0:
        ramp = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        samples[:fade_in_samples] *= ramp[:, np.newaxis]
    if fade_out_samples > 0:
        ramp = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        samples[-fade_out_samples:] *= ramp[:, np.newaxis]

    int16 = (samples.clip(-1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def begin_session(session_id: str, prompt: str, **kwargs) -> None:
    """Embed the style prompt and cache it for this session."""
    system = _get_system()
    _sessions[session_id] = system.embed_style(prompt)


def end_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def render_melody(session_id: str, segments: list) -> bytes:
    """Render a melody segment list and return WAV bytes."""
    system = _get_system()
    style = _sessions.get(session_id)  # None = unconditional if session missing
    state = None
    chunks = []

    for seg in segments:
        wf, state = system.generate(
            style=style,
            notes=seg["notes"],   # 128-int conditioning vector
            drums=None,           # -1 (masked) = no drums
            frames=seg.get("frames", 1),
            state=state,
        )
        chunks.append(wf.samples)  # (N, 2) float32

    if not chunks:
        return b""

    return _pcm_to_wav(np.concatenate(chunks, axis=0))
