"""Real-Time Speech Denoiser: Streamlit demo.

A causal 2-layer GRU predicts a magnitude mask for each 16 ms STFT frame.
The model runs frame by frame in ONNX Runtime, exactly as it would in a live stream.
"""
import io, json, time
from math import gcd
from pathlib import Path

import numpy as np, soundfile as sf, streamlit as st, onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

SR, N_FFT, HOP, MAX_SEC = 16000, 512, 256, 60
ROOT = Path(__file__).parent
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann = torch
SAMPLES = {"Clip 1": 1, "Clip 2": 2, "Clip 3": 3}


@st.cache_resource
def load_model():
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ROOT / "models" / "gru_step.onnx"), so, providers=["CPUExecutionProvider"])
    try: target_rms = json.load(open(ROOT / "models" / "norm.json"))["target_rms"]
    except Exception: target_rms = 0.1
    return sess, target_rms


def load_audio(data):
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(1)                                            # stereo -> mono
    if sr != SR:                                             # any rate -> 16 kHz
        g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x[: MAX_SEC * SR], sr


def stream_enhance(x, sess, target_rms):
    gain = np.clip(target_rms / (np.sqrt(np.mean(x ** 2)) + 1e-8), 0.1, 100.0)
    x = x * gain                                             # match training loudness
    h = np.zeros((2, 1, 128), np.float32); buf = np.zeros(N_FFT, np.float32)
    xp = np.concatenate([x, np.zeros(2 * HOP, np.float32)])
    y = np.zeros(len(xp) + N_FFT, np.float32); lat = []
    for k in range(len(x) // HOP + 1):                       # 16 ms frames, past audio only
        buf = np.concatenate([buf[HOP:], xp[k * HOP:(k + 1) * HOP]])
        t0 = time.perf_counter()
        X = np.fft.rfft(buf * WIN)
        mask, h = sess.run(None, {"mag": np.abs(X).astype(np.float32)[None, None], "h_in": h})
        y[k * HOP:k * HOP + N_FFT] += np.fft.irfft(X * mask[0, 0], N_FFT).astype(np.float32)
        lat.append(time.perf_counter() - t0)
    return np.clip(y[HOP:HOP + len(x)] / gain, -1, 1), x / gain, np.array(lat) * 1e3


@st.cache_data(show_spinner=False)
def process(data):
    sess, target_rms = load_model()
    x, orig_sr = load_audio(data)
    t0 = time.perf_counter(); y, x, lat = stream_enhance(x, sess, target_rms)
    return x, y, lat, time.perf_counter() - t0, orig_sr


def wav_bytes(x):
    b = io.BytesIO(); sf.write(b, x, SR, format="WAV", subtype="PCM_16"); return b.getvalue()


def spec_fig(x, title):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.specgram(x + 1e-9, NFFT=N_FFT, Fs=SR, noverlap=N_FFT - HOP, cmap="magma")
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz"); fig.tight_layout()
    return fig


# ---------------- UI ----------------
st.set_page_config(page_title="Real-Time Speech Denoiser", page_icon="🎧", layout="wide")
st.title("🎧 Real-Time Speech Denoiser")
st.caption("Causal GRU mask model (264k parameters) running in ONNX Runtime on one CPU thread, "
           "frame by frame in true streaming mode (16 ms frames).")

with st.sidebar:
    st.header("About")
    st.markdown("- **Model:** 2-layer GRU predicting STFT magnitude masks\n"
                "- **Training data:** VoiceBank-DEMAND (16 kHz)\n"
                "- **Test results (200 clips):** PESQ 2.16 → 2.65, SI-SDR +8 dB\n"
                "- **Latency:** 32 ms algorithmic, well under 1 ms compute per frame")
    st.markdown("**Test-set results**")
    st.markdown((ROOT / "results.md").read_text())
    st.info("Trained on ~1.5 h of VoiceBank-DEMAND only. Real recordings with unseen noise types "
            "or reverb may be denoised less well.")
    st.markdown("Built by [Janvi Gabani](https://janvikgabani.github.io) · "
                "[Source code](https://github.com/JanviKGabani/speech-denoiser)")

tab_sample, tab_upload = st.tabs(["Try a test clip", "Upload your own audio"])
with tab_sample:
    pick = st.radio("Noisy clip from the VoiceBank-DEMAND test set", list(SAMPLES), horizontal=True)
with tab_upload:
    up = st.file_uploader("Noisy speech (wav, flac, ogg or mp3; the first 60 s are used)",
                          type=["wav", "flac", "ogg", "mp3"])

if up is not None:
    data, name, idx = up.getvalue(), up.name, None
else:
    idx = SAMPLES[pick]
    path = ROOT / "samples" / f"sample{idx}_1_noisy.wav"
    data, name = path.read_bytes(), path.name

with st.spinner("Denoising…"):
    try: x, y, lat, secs, orig_sr = process(data)
    except Exception as e: st.error(f"Could not read this file. Try a WAV or FLAC file. ({e})"); st.stop()

dur = len(x) / SR
c = st.columns(4)
c[0].metric("Audio length", f"{dur:.1f} s")
c[1].metric("Processing time", f"{secs:.2f} s")
c[2].metric("Real-time factor", f"{secs / dur:.3f}")
c[3].metric("Per-frame compute", f"{lat.mean():.2f} ms", f"p99 {np.percentile(lat, 99):.2f} ms", delta_color="off")
if orig_sr != SR: st.caption(f"Resampled from {orig_sr} Hz to 16 kHz mono.")

left, right = st.columns(2)
with left:
    st.subheader("Noisy input"); st.audio(wav_bytes(x), format="audio/wav")
    st.pyplot(spec_fig(x, "Noisy"))
with right:
    st.subheader("Denoised output (GRU)"); st.audio(wav_bytes(y), format="audio/wav")
    st.pyplot(spec_fig(y, "Denoised"))
    st.download_button("⬇️ Download denoised audio", wav_bytes(y),
                       file_name=Path(name).stem + "_denoised.wav", mime="audio/wav")

if idx is not None:
    st.divider()
    st.subheader("Compare with the classical baseline and the clean reference")
    a, b = st.columns(2)
    with a:
        st.markdown("**Wiener filter (baseline)**")
        st.audio(str(ROOT / "samples" / f"sample{idx}_2_wiener.wav"))
    with b:
        st.markdown("**Clean reference**")
        st.audio(str(ROOT / "samples" / f"sample{idx}_4_clean.wav"))
