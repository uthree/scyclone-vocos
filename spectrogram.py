import torch
import matplotlib.pyplot as plt

def spectrogram(x):
    x = torch.stft(x, n_fft=1024, hop_length=256, center=True, return_complex=True).abs()
    return x[:, :, 1:]

def plot_spectrogram(x, save_path="./spectrogram.png", log=True):
    if log:
        x = torch.log10(x ** 2 + 1e-4)
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)

