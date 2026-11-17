import torch, torch.nn as nn

class GRUMask(nn.Module):
    """Causal frame-by-frame mask estimator: |noisy STFT| -> mask in [0,1]."""
    def __init__(self, n_freq=257, hidden=128, layers=2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_freq))   # global feature normalization
        self.register_buffer("sd", torch.ones(n_freq))    # (fixed stats, so it stays causal)
        self.inp = nn.Sequential(nn.Linear(n_freq, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, n_freq), nn.Sigmoid())

    def forward(self, mag, h=None):                  # mag: [B, T, F]
        x = (torch.log(mag + 1e-6) - self.mu) / self.sd
        x, h = self.gru(self.inp(x), h)
        return self.out(x), h                        # mask [B, T, F], hidden state
