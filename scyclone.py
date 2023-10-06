import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels=256, spectral_norm=True, alpha=0.01, dropout=0, groups=1):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2, groups=groups)
        self.act = nn.LeakyReLU(alpha)
        self.c2 = nn.Conv1d(channels, channels, 5, 1, 2, groups=groups)
        self.dropout_rate = dropout
        if spectral_norm:
            self.c1 = torch.nn.utils.spectral_norm(self.c1)
            self.c2 = torch.nn.utils.spectral_norm(self.c2)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.c2(x)
        x = F.dropout(x, p=self.dropout_rate)
        return x + res


class Generator(nn.Module):
    def __init__(self, input_channels=513, internal_channels=256, num_layers=7, groups=1):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.Sequential(
                *[ResBlock(internal_channels, False, groups=groups) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, input_channels, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        F.leaky_relu(x, 0.01)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=513, internal_channels=256, num_layers=6, groups=1):
        super().__init__()
        self.input_layer = nn.utils.spectral_norm(
                nn.Conv1d(input_channels, internal_channels, 5, 1, 0))
        self.mid_layers = nn.Sequential(
                *[ResBlock(internal_channels, True, groups=groups, dropout=0.0, alpha=0.2) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, 1, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.mid_layers:
            x = l(x)
        return [self.output_layer(x)]

    def feature_matching_loss(self, x, y):
        x = x = torch.randn_like(x) * 0.005
        x = self.input_layer(x)
        out = 0
        with torch.no_grad():
            y = self.input_layer(y)
        for l in self.mid_layers:
            x = l(x)
            with torch.no_grad():
                y = l(y)
            out += (x-y).abs().mean()
        return out
