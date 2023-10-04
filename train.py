import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from model import Generator, Discriminator
from dataset import WaveFileDirectory
from preprocess import spectrogram


def load_or_init_models(device=torch.device('cpu')):
    paths = ["./g_a2b.pt", "./g_b2a.pt", "./d_a.pt", "./d_b.pt"]
    model_classes = [Generator, Generator, Discriminator, Discriminator]
    models = []
    for cls, p in zip(model_classes, paths):
        if os.path.exists(p):
            m = cls()
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device)
            models.append(m)
            print(f"Loaded model from {p}")
        else:
            models.append(cls().to(device))
            print(f"Initialized {p}")
    return models


def save_models(Gab, Gba, Da, Db):
    torch.save(Gab.state_dict(), "./g_a2b.pt")
    torch.save(Gba.state_dict(), "./g_b2a.pt")
    torch.save(Da.state_dict(), "./d_a.pt")
    torch.save(Db.state_dict(), "./d_b.pt")
    print("Saved models")


def cutmid(x):
    length = x.shape[2]
    s = length // 8
    e = length - (length // 8)
    return x[:, :, s:e]


parser = argparse.ArgumentParser(description="Train Cycle GAN")

parser.add_argument('dataset_path_a')
parser.add_argument('dataset_path_b')
parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch', default=16, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learningrate', default=2e-4, type=float)
parser.add_argument('-len', '--length', default=32768, type=int)
parser.add_argument('--consistency', default=5.0, type=float, help="weight of cycle-consistency loss")
parser.add_argument('--identity', default=1.0, type=float, help="weight of identity loss")
parser.add_argument('--feature-matching', default=5.0, type=float, help="weight of feature-matching loss")
parser.add_argument('-psa', '--pitch-shift-a', default=0, type=int)
parser.add_argument('-psb', '--pitch-shift-b', default=0, type=int)
parser.add_argument('-ga', '--gain-a', default=1, type=float)
parser.add_argument('-gb', '--gain-b', default=1, type=float)
parser.add_argument('-gacc', '--gradient-accumulation', type=int, default=1)

args = parser.parse_args()
device_name = args.device

grad_accm = args.gradient_accumulation

print(f"selected device: {device_name}")
if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

if torch.cuda.is_available() and device_name != "cuda":
    print(f"Warning: CUDA is available in this environment, but selected device is {device_name}. training process will may be slow.")

device = torch.device(device_name)

Gab, Gba, Da, Db = load_or_init_models(device)

ds_a = WaveFileDirectory(
        [args.dataset_path_a],
        length=args.length,
        max_files=args.maxdata)

ds_b = WaveFileDirectory(
        [args.dataset_path_b],
        length=args.length,
        max_files=args.maxdata)

dl_a = torch.utils.data.DataLoader(ds_a, batch_size=args.batch, shuffle=True)
dl_b = torch.utils.data.DataLoader(ds_b, batch_size=args.batch, shuffle=True)

OGab = optim.Adam(Gab.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
OGba = optim.Adam(Gba.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
ODa = optim.Adam(Da.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
ODb = optim.Adam(Db.parameters(), lr=args.learningrate, betas=(0.5, 0.999))

if args.pitch_shift_a != 0:
    Ta = torchaudio.transforms.PitchShift(22050, args.pitch_shift_a).to(device)
else:
    Ta = nn.Identity().to(device)

if args.pitch_shift_b != 0:
    Tb = torchaudio.transforms.PitchShift(22050, args.pitch_shift_b).to(device)
else:
    Tb = nn.Identity().to(device)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
L1 = nn.L1Loss()

Lcyc = args.consistency
Lid = args.identity
Lfeat = args.feature_matching

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=min(len(ds_a), len(ds_b)))
    for batch, (real_a, real_b) in enumerate(zip(dl_a, dl_b)):
        if real_a.shape[0] != real_b.shape[0]:
            continue
        N = real_a.shape[0]
        
        # Convert waveform to spectrogram
        rand_gain = torch.rand(N, 1,  device=device) * 0.75 + 0.25
        real_a = spectrogram(Ta(real_a.to(device) * args.gain_a * rand_gain)).detach()
        real_b = spectrogram(Tb(real_b.to(device) * args.gain_b * rand_gain)).detach()

        # Train G.
        if batch % grad_accm == 0:
            OGab.zero_grad()
            OGba.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            fake_b = Gab(real_a)
            fake_a = Gba(real_b)
            recon_a = Gba(fake_b)
            recon_b = Gab(fake_a)
            id_out_a = Gba(real_a)
            id_out_b = Gab(real_b)

            loss_G_cyc = L1(recon_b, real_b) + L1(recon_a, real_a)
            loss_G_id = Db.feature_matching_loss(id_out_b, real_b) + Da.feature_matching_loss(id_out_a, real_a) +\
                    L1(id_out_a, real_a) + L1(id_out_b, real_b)
            loss_G_feat = Da.feature_matching_loss(recon_a, real_a) +\
                Db.feature_matching_loss(recon_b, real_b)

            logits = Db(cutmid(fake_b)) +\
                Da(cutmid(fake_a)) +\
                Db(cutmid(recon_b)) +\
                Da(cutmid(recon_a))
            loss_G_adv = 0
            for logit in logits:
                loss_G_adv += F.relu(-logit).mean() / len(logits)
            loss_G = loss_G_adv + loss_G_id * Lid + loss_G_cyc * Lcyc + loss_G_feat * Lfeat

        scaler.scale(loss_G).backward()

        nn.utils.clip_grad_norm_(Gab.parameters(), max_norm=1.0, norm_type=2.0)
        nn.utils.clip_grad_norm_(Gba.parameters(), max_norm=1.0, norm_type=2.0)

        fake_a = fake_a.detach()
        fake_b = fake_b.detach()
        recon_a = recon_a.detach()
        recon_b = recon_b.detach()
        
        if batch % grad_accm == 0:
            scaler.step(OGab)
            scaler.step(OGba)
        
        if batch % grad_accm == 0:
            # Train D.
            ODa.zero_grad()
            ODb.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = Da(cutmid(fake_a)) + Db(cutmid(fake_b))
            logits_real = Da(cutmid(real_a)) + Db(cutmid(real_b))
            loss_D = 0
            for logit in logits_fake:
                loss_D += F.relu(0.5 + logit).mean() / len(logits_fake)
            for logit in logits_real:
                loss_D += F.relu(0.5 - logit).mean() / len(logits_real)

        scaler.scale(loss_D).backward()

        nn.utils.clip_grad_norm_(Da.parameters(), max_norm=1.0, norm_type=2.0)
        nn.utils.clip_grad_norm_(Db.parameters(), max_norm=1.0, norm_type=2.0)
        
        if batch % grad_accm == 0:
            scaler.step(ODa)
            scaler.step(ODb)
            scaler.update()
        
        tqdm.write(f"Id: {loss_G_id.item():.4f}, Adv.: {loss_G_adv.item():.4f}, Cyc.: {loss_G_cyc.item():.4f}, Feat.: {loss_G_feat.item():.4f}")
        bar.set_description(desc=f"G: {loss_G.item():.4f}, D: {loss_D.item():.4f}")
        bar.update(N)

        if loss_D.isnan().any() or loss_G.isnan().any():
            exit()

        if batch % 100 == 0:
            save_models(Gab, Gba, Da, Db)

save_models(Gab, Gba, Da, Db)
print("Training complete!")
