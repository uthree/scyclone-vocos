import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from dataset import WaveFileDirectory
from spectrogram import spectrogram
from vocos import Vocos, Discriminator

parser = argparse.ArgumentParser(description="train Vocoder")

parser.add_argument('dataset')
parser.add_argument('-gp', '--generator-path', default="vocoder_g.pt")
parser.add_argument('-dp', '--discriminator-path', default="vocoder_d.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    g = Vocos().to(device)
    d = Discriminator().to(device)
    if os.path.exists(args.generator_path):
        g.load_state_dict(torch.load(args.generator_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        d.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return g, d


def save_models(g, d):
    print("Saving Models...")
    torch.save(g.state_dict(), args.generator_path)
    torch.save(d.state_dict(), args.discriminator_path)
    print("complete!")


def write_preview(source_wave, file_path='./vocoder_preview.wav'):
    source_wave = source_wave.detach().to(torch.float32).cpu()
    torchaudio.save(src=source_wave, sample_rate=22050, filepath=file_path)


device = torch.device(args.device)
G, D = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptG = optim.AdamW(G.parameters(), lr=args.learning_rate)
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate)

mel = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=80).to(device)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device)
        spec = spectrogram(wave)
        
        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            fake_wave = G(spec)
            logits = D.logits(fake_wave)
            
            loss_mel = (mel(fake_wave) - mel(wave)).abs().mean()
            loss_feat = D.feat_loss(fake_wave, wave)
            loss_adv = 0
            for logit in logits:
                loss_adv += (logit ** 2).mean()
            
            loss_g = loss_mel * 45 + loss_feat * 2 + loss_adv
        scaler.scale(loss_g).backward()
        scaler.step(OptG)

        # Train D.
        OptD.zero_grad()
        fake_wave = fake_wave.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = D.logits(fake_wave)
            logits_real = D.logits(wave)
            loss_d = 0
            for logit in logits_real:
                loss_d += (logit ** 2).mean()
            for logit in logits_fake:
                loss_d += ((logit - 1) ** 2).mean()
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        
        tqdm.write(f"D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 100 == 0:
            save_models(G, D)
            write_preview(fake_wave[0].unsqueeze(0))

print("Training Complete!")
save_models(G, D)
