import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

from spectrogram import spectrogram
from vocos import Vocos

parser = argparse.ArgumentParser(description="Inference vocoder")

parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-gp', '--generator-path', default='./vocoder_g.pt')

args = parser.parse_args()

device = torch.device(args.device)

vocoder = Vocos().to(device)
vocoder.load_state_dict(torch.load(args.generator_path, map_location=device))

if not os.path.exists(args.output):
    os.mkdir(args.output)

for i, fname in enumerate(os.listdir(args.input)):
    print(f"Inferencing {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 22050)
        wf = wf.to(device)
        
        spec = spectrogram(wf)
        wf_out = vocoder(spec)

        wf_out = resample(wf_out, 22050, sr)
        wf_out = wf_out.to(torch.device('cpu'))
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf_out, sample_rate=sr)
