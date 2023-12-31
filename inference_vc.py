import argparse
import sys
import json
import torchaudio
import os
import glob
from spectrogram import spectrogram, plot_spectrogram
from scyclone import Generator as Convertor
from vocos import Vocos as Vocoder
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-mp', '--model-path', default='./g_a2b.pt')
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-ps', '--pitch-shift', default=0, type=int)

args = parser.parse_args()

device = torch.device(args.device)

# Load Vocoder
vocoder = Vocoder()
vocoder.load_state_dict(torch.load("vocoder_g.pt", map_location=device))

# Load convertor
convertor = Convertor()
convertor.load_state_dict(torch.load(args.model_path, map_location=device))
convertor = convertor.to(device)

if args.pitch_shift != 0:
    ps = torchaudio.transforms.PitchShift(22050, args.pitch_shift).to(device)
else:
    ps = torch.nn.Identity().to(device)


if not os.path.exists("./outputs/"):
    os.mkdir("./outputs")

paths = glob.glob("./inputs/*.wav")
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 22050) * args.input_gain
    with torch.no_grad():
        print(f"converting {path}")
        wf = ps(wf)
        lin_spec = spectrogram(wf)
        plot_spectrogram(lin_spec.detach().cpu()[0], os.path.join("./outputs/", f"{i}_input.png"))
        lin_spec = convertor(lin_spec)
        plot_spectrogram(lin_spec.detach().cpu()[0], os.path.join("./outputs/", f"{i}_output.png"))
        wf = vocoder(lin_spec)
    wf = torchaudio.functional.resample(wf, 22050, sr) * args.gain
    wf = wf.cpu().detach()
    torchaudio.save(filepath=os.path.join("./outputs/", f"{i}.wav"), src=wf, sample_rate=sr)
