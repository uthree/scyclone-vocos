import torch

def spectrogram(x):
    x = torch.stft(x, n_fft=1024, hop_length=256, center=True, return_complex=True).abs()
    return x[:, :, 1:]
