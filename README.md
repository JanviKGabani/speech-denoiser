# Real-Time Speech Denoiser

A small causal speech-enhancement model that removes background noise from speech, frame by frame, fast enough for live audio.

**Live demo:** [janvi-speech-denoiser.streamlit.app](https://janvi-speech-denoiser.streamlit.app) · **Portfolio:** [janvikgabani.github.io](https://janvikgabani.github.io)

## How it works

- **Model:** a 2-layer GRU (264k parameters) that predicts a 0–1 mask for each STFT magnitude frame (`model.py`).
- **Streaming:** 512-sample Hann window, 256-sample hop (32 ms window, 16 ms frames). The model only sees past audio, and its hidden state carries across frames.
- **Deployment:** exported to ONNX as a single-frame step (`models/gru_step.onnx`) and run in ONNX Runtime on one CPU thread.
- **Baseline:** a decision-directed Wiener filter (`se_utils.py`).
- **Training data:** [VoiceBank-DEMAND 16 kHz](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k), one training shard (~1.5 h).

## Results (200 test clips)

|                      | SI-SDR (dB) | STOI  | PESQ-WB |
|:---------------------|------------:|------:|--------:|
| Noisy                |       8.469 | 0.919 |   2.162 |
| Wiener               |      15.072 | 0.905 |   2.406 |
| GRU (ours)           |      16.064 | 0.926 |   2.676 |
| GRU streaming (ONNX) |      16.423 | 0.925 |   2.653 |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| Path | What it is |
|---|---|
| `app.py` | Streamlit demo: upload audio or pick a test clip, hear and compare the output |
| `model.py` | GRU mask model (PyTorch) |
| `se_utils.py` | STFT helpers, Wiener baseline, SI-SDR / STOI / PESQ metrics |
| `models/gru_best.pt` | Trained PyTorch weights |
| `models/gru_step.onnx` | Streaming ONNX export used by the app |
| `samples/` | Three test clips: noisy, Wiener, GRU and clean versions |

The model was trained on clean studio speech mixed with DEMAND noise, so recordings with very different noise or strong reverb may be cleaned less well.
