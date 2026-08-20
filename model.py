"""
Acoustic model trained with CTC loss:
log-mel spectrogram -> 2D-conv front end -> BiLSTM -> FC -> log_softmax.

Command-word clips are ~1s, so the front end ends up emitting ~100 frames
to the LSTM -- the effective context window is short by construction, which
is what keeps inference latency low (no chunking, no long-range attention).
"""
import torch.nn as nn
import torch.nn.functional as F


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
