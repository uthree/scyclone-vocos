import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm 
import torchaudio

from preprocess import log_mel_scale


class ResBlock(nn.Module):
    def __init__(self, channels, use_spectral_norm=False, alpha=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.c2 = nn.Conv1d(channels, channels, 5, 1, 2)
        if use_spectral_norm:
            self.c1 = spectral_norm(self.c1)
            self.c2 = spectral_norm(self.c2)
        self.alpha = alpha

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = F.leaky_relu(x, self.alpha)
        x = self.c2(x)
        return x + res


class Generator(nn.Module):
    def __init__(self, input_channels=513, internal_channels=256, num_layers=7):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.Sequential(*[ResBlock(internal_channels, False, 0.01) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, input_channels, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x


class LogMelDiscriminator(nn.Module):
    def __init__(self, n_mels=100, internal_channels=256, num_layers=6):
        super().__init__()
        self.input_layer = nn.Conv1d(n_mels, internal_channels, 5, 1, 2)
        self.mid_layers = nn.Sequential(*[ResBlock(internal_channels, True, 0.2) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, 1, 5, 1, 2)

    def forward(self, x):
        x = log_mel_scale(x)
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x

    def feat(self, x):
        feats = []
        x = log_mel_scale(x)
        x = self.input_layer(x)
        for l in self.mid_layers:
            x = l(x)
            feats.append(x)
        return feats


class LinearDiscriminator(nn.Module):
    def __init__(self, input_channels=513, internal_channels=256, num_layers=6):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 5, 1, 2)
        self.mid_layers = nn.Sequential(*[ResBlock(internal_channels, True, 0.2) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, 1, 5, 1, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x

    def feat(self, x):
        feats = []
        x = self.input_layer(x)
        for l in self.mid_layers:
            x = l(x)
            feats.append(x)
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_mel_d = LogMelDiscriminator()
        self.linear_d = LinearDiscriminator()

    def forward(self, x):
        return [self.log_mel_d(x), self.linear_d(x)]

    def feature_matching_loss(self, fake, real):
        fake_feats = self.log_mel_d.feat(fake) + self.linear_d.feat(fake)
        real_feats = self.log_mel_d.feat(real) + self.linear_d.feat(real)
        loss = 0
        for f, r in zip(fake_feats, real_feats):
            loss += (f - r).abs().mean()
        return loss
