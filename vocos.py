import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class ConvNeXt1d(nn.Module):
    def __init__(self, channels=512, hidden_channels=1024, kernel_size=7, scale=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, padding='same', groups=channels)
        self.norm = ChannelNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res


class Vocos(nn.Module):
    def __init__(self,
                 input_channels=513,
                 n_fft=1024,
                 internal_channels=512,
                 hidden_channels=1536,
                 num_layers=8):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.mid_layers.append(ConvNeXt1d(internal_channels, hidden_channels, scale=1/num_layers))
        self.last_norm = ChannelNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, n_fft+2, 1, 1, 0)
        self.n_fft = n_fft

    def forward(self, x):
        x = self.pad(x)
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_norm(x)
        x = self.output_layer(x)
        m, p = torch.chunk(x, 2, dim=1)
        m = torch.clamp_max(m, 6.0)
        m = torch.exp(m)
        s = m * (torch.cos(p) + 1j * torch.sin(p))
        return torch.istft(s, n_fft=self.n_fft, center=True, hop_length=256, onesided=True)


class PeriodicDiscriminator(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 dropout_rate=0.2,
                 groups = [],
                 max_channels=256
                 ):
        super().__init__()
        self.input_layer = weight_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), 0))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = min(channels * (4 ** i), max_channels)
            c_next = min(channels * (4 ** (i+1)), max_channels)
            if i == (num_stages - 1):
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i])))
            else:
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i])))
                self.layers.append(
                        nn.Dropout(dropout_rate))
                self.layers.append(
                        nn.LeakyReLU(LRELU_SLOPE))
        c = min(channels * (4 ** (num_stages-1)), max_channels)
        self.final_conv = weight_norm(
                nn.Conv2d(c, c, (5, 1), 1, 0)
                )
        self.final_relu = nn.LeakyReLU(LRELU_SLOPE)
        self.output_layer = weight_norm(
                nn.Conv2d(c, 1, (3, 1), 1, 0))
        self.period = period

    def forward(self, x, logit=True):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        if logit:
            x = self.output_layer(x)
        return x

    def feat(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 groups=[1, 2, 4, 4, 4],
                 channels=32,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])

        for p in periods:
            self.sub_discriminators.append(
                    PeriodicDiscriminator(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits
    
    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats



class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft, channels=64):
        super().__init__()
        self.n_fft = n_fft
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        return x

    def feat(self, x):
        feats = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feats.append(x)
        return feats
        
    def spectrogram(self, x):
        x = torch.stft(x, self.n_fft, self.n_fft // 4, return_complex=True, center=True, window=None).abs()
        return x


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=[512, 1024, 2048], channels=64):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            ResolutionDiscriminator(r, channels)
            for r in resolutions
            ])

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits

    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats += sd.feat(x)
        return feats


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.AvgPool1d(2),
            weight_norm(nn.Conv1d(1, 64, 41, 4, 0)),
            weight_norm(nn.Conv1d(64, 64, 41, 4, 0)),
            weight_norm(nn.Conv1d(64, 64, 41, 4, 0)),
            weight_norm(nn.Conv1d(64, 64, 41, 4, 0)),
            weight_norm(nn.Conv1d(64, 1, 41, 4, 0)),
            ])

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        return x

    def feat(self, x):
        feats = []
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feats.append(x)
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MRD = MultiResolutionDiscriminator()
        self.SD = ScaleDiscriminator()
    
    def logits(self, x):
        return self.MPD(x) + [self.SD(x)]
    
    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.MPD.feat(real) + self.MRD.feat(real) + self.SD.feat(real) + [real]
        fake_feat = self.MPD.feat(fake) + self.MRD.feat(fake) + self.SD.feat(fake) + [fake]
        loss = 0
        for r, f in zip(real_feat, fake_feat):
            loss = loss + F.l1_loss(f, r)
        return loss
