import numpy as np, torch, time
from pesq import pesq
from pystoi import stoi

SR, N_FFT, HOP = 16000, 512, 256          # 32 ms window, 16 ms hop, 257 freq bins
WIN = torch.hann_window(N_FFT)

def stft(x):                               # np [L] -> complex tensor [F, T]
    return torch.stft(torch.from_numpy(x).float(), N_FFT, HOP, window=WIN, return_complex=True)

def istft(X, length):                      # complex [F, T] -> np [L]
    return torch.istft(X, N_FFT, HOP, window=WIN, length=length).numpy()

# ---------- Wiener filter (decision-directed, Ephraim-Malah style) ----------
def wiener(x, noise_frames=8, alpha=0.98, gmin=0.1):
    X = stft(x); P = X.abs() ** 2                          # noisy power [F, T]
    noise = P[:, :noise_frames].mean(1) + 1e-10            # noise PSD from leading frames
    G = torch.empty_like(P); S_prev = torch.zeros_like(noise)
    for t in range(P.shape[1]):
        gamma = P[:, t] / noise                            # a posteriori SNR
        xi = alpha * S_prev / noise + (1 - alpha) * torch.clamp(gamma - 1, min=0)  # a priori SNR
        g = torch.clamp(xi / (1 + xi), min=gmin)           # Wiener gain + floor
        S_prev = g ** 2 * P[:, t]
        G[:, t] = g
    return istft(X * G, len(x))

# ---------- Metrics ----------
def si_sdr(ref, est, eps=1e-8):
    ref, est = ref - ref.mean(), est - est.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref   # project est onto ref
    return 10 * np.log10((s @ s + eps) / ((est - s) @ (est - s) + eps))

def metrics(ref, est):
    try:    p = pesq(SR, ref, est, "wb")
    except Exception: p = np.nan
    return si_sdr(ref, est), stoi(ref, est, SR, extended=False), p

def evaluate(name, fn, data):
    t0 = time.time()
    rows = np.array([metrics(c, fn(n)) for n, c in data])
    m = np.nanmean(rows, 0)
    print(f"{name:8s} | SI-SDR {m[0]:6.2f} dB | STOI {m[1]:.3f} | PESQ {m[2]:.3f}   ({time.time()-t0:.0f}s)")
    return m
