"""
Acoustic models trained with CTC loss
bidirectional LSTM -> FC -> log_softmax) - differing front end:

CTCAcousticModel -- features in directly (LSTM+CTC baseline, no conv front end)
CTCConv1DModel -- raw waveform -> 1D-conv front end -> LSTM+CTC  (train1.py)
CTCConv2DModel -- log-mel spectr -> 2D-conv front end -> LSTM+CTC  (train2.py)

Command-word clips are ~1s, so each front end ends up emitting ~100 frames
to the LSTM -- the effective context window is short by construction, which
is what keeps inference latency low (no chunking, no long-range attention).
"""
import torch.nn as nn
import torch.nn.functional as F


class CTCAcousticModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, num_layers=2, vocab_size=27):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: (batch, time, input_dim) -> (batch, time, vocab_size) log-probs
        out, _ = self.lstm(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


class CTCConv1DModel(nn.Module):
    """
    Raw waveform -> stack of strided 1D convs 
    16kHz samples -> ~100 frames/sec, 
    """
    # (out_channels, kernel_size, stride, padding) per conv layer.
    CONV_CFG = [
        (32,  10, 5, 5),
        (64,   8, 4, 2),
        (128,  4, 2, 1),
        (128,  4, 2, 1),
        (128,  4, 2, 1),
    ]

    def __init__(self, hidden_dim=128, num_layers=2, vocab_size=27):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch, k, s, p in self.CONV_CFG:
            layers += [nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            in_ch, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: (batch, samples) raw waveform
        x = x.unsqueeze(1)             # (B, 1, samples)
        x = self.conv(x)               # (B, C, T')
        x = x.transpose(1, 2)          # (B, T', C)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

    def output_length(self, input_length):
        """Map raw sample counts to downsampled frame counts (for CTC input_lengths)."""
        L = input_length
        for _, k, s, p in self.CONV_CFG:
            L = (L + 2 * p - k) // s + 1
        return L


class CTCConv2DModel(nn.Module):
    """
    Log-mel spectrogram (treated as a 1-channel 80 x T "image") -> stack of
    strided 2D convs 
    """
    # (out_channels, kernel_size, stride, padding) per conv layer.
    CONV_CFG = [
        (32, (11, 41), (2, 2), (5, 20)),
        (32, (11, 21), (1, 2), (5, 10)),
    ]

    def __init__(self, n_mels=80, hidden_dim=128, num_layers=2, vocab_size=27):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch, k, s, p in self.CONV_CFG:
            layers += [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        freq = n_mels
        for _, k, s, p in self.CONV_CFG:
            freq = (freq + 2 * p[0] - k[0]) // s[0] + 1
        self.lstm = nn.LSTM(
            in_ch * freq, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: (batch, n_mels, time)
        x = x.unsqueeze(1)                    
        x = self.conv(x)                            
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  
        out, _ = self.lstm(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

    def output_length(self, input_length):
        """Map spectrogram frame counts to post-conv frame counts (for CTC input_lengths)."""
        L = input_length
        for _, k, s, p in self.CONV_CFG:
            L = (L + 2 * p[1] - k[1]) // s[1] + 1
        return L
