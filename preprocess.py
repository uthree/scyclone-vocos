import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt


global mel_filters
mel_filters = dict()
to_specs = dict()

# Plot spectrogram
def plot_spectrogram(x, save_path="./spectrogram.png", log=True):
    if log:
        x = torch.log10(x ** 2 + 1e-4)
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)


def safe_log(x, clip_val = 1e-7):
    return torch.log(torch.clip(x, min=clip_val))

def mel_scale(x):
    if x.device.__str__() not in mel_filters:
        mel_filters[x.device.__str__()] = torchaudio.transforms.MelScale(
                100,
                24000,
                n_stft=513).to(x.device)

    return mel_filters[x.device.__str__()](x)

def log_mel_scale(x):
    return safe_log(mel_scale(x))

def spectrogram(x):
    if x.device.__str__() not in to_specs:
        to_specs[x.device.__str__()] = torchaudio.transforms.Spectrogram(
                1024, 1024, 256, 0, power=1).to(x.device)

    return to_specs[x.device.__str__()](x)[:, :, 1:]

    return s
