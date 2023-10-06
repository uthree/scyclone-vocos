import argparse
import pyaudio
import numpy as np
import torch
import torchaudio
from spectrogram import spectrogram
from scyclone import Generator as Convertor
from vocos import Vocos as Vocoder
import sys
import json

parser = argparse.ArgumentParser(description="Convert voice")

audio = pyaudio.PyAudio()
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Compute device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-thr', '--threshold', default=-40.0, type=float)
parser.add_argument('-v', '--vocoderpath', default='./vocoder_g.pt', type=str)
parser.add_argument('-m', '--modelpath', default='./g_a2b.pt')
parser.add_argument('-b', '--buffersize', default=8, type=int)
parser.add_argument('-c', '--chunk', default=2048, type=int)
parser.add_argument('-ic', '--inputchannels', default=1, type=int)
parser.add_argument('-oc', '--outputchannels', default=1, type=int)
parser.add_argument('-lc', '--loopbackchannels', default=1, type=int)
parser.add_argument('-ps', '--pitch-shift', default=0, type=int)
parser.add_argument('-compile', default=False, type=bool)
parser.add_argument('-fp16', default=False, type=bool)

args = parser.parse_args()
device_name = args.device

print(f"selected device: {device_name}")
if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

device = torch.device(device_name)
input_buff = []
chunk = args.chunk
buffer_size = args.buffersize

convertor = Convertor()
convertor.load_state_dict(torch.load(args.modelpath, map_location=device))
convertor = convertor.to(device)

# Load Vocoder
vocoder = Vocoder()
vocoder.load_state_dict(torch.load(args.vocoderpath, map_location=device))
vocoder = vocoder.to(device)

if args.compile:
    print("Compiling Models...")
    convertor = torch.compile(convertor)
    vocoder = torch.compile(vocoder)
    print("Complete!")

stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=args.inputchannels,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.outputchannels,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.loopbackchannels,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

if args.pitch_shift != 0:
    pitch_shift = torchaudio.transforms.PitchShift(22050, args.pitch_shift).to(device)
else:
    pitch_shift = torch.nn.Identity()

print("Converting Voice...")
while True:
    data = stream_input.read(chunk, exception_on_overflow=False)
    data = np.frombuffer(data, dtype=np.int16)
    input_buff.append(data)
    if len(input_buff) > buffer_size:
        del input_buff[0]
    else:
        continue
    if not data.max() > args.threshold:
        data = data * 0
    data = np.concatenate(input_buff, 0)
    data = data.astype(np.float32) / 32768 # convert -1 to 1
    data = torch.from_numpy(data).to(device)
    data = torch.unsqueeze(data, 0)
    data = data * args.input_gain
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            # Pitch shift
            data = pitch_shift(data)
            # Downsample
            data = torchaudio.functional.resample(data, 44100, 22050)
            # Calculate loudness
            loudness = torchaudio.functional.loudness(data, 22050)
            if loudness.item() > args.threshold:
                # to spectrogram
                spec = spectrogram(data)
                # convert voice
                spec = convertor(spec)
                # pass Vocoder
                data = vocoder(spec)
            # gain
            data = torchaudio.functional.gain(data, args.gain)
            # Upsample
            data = torchaudio.functional.resample(data, 22050, 44100)
            data = data[0]
    data = data.cpu().numpy()
    data = (data) * 32768
    data = data
    data = data.astype(np.int16)
    s = (chunk * buffer_size) // 2 - (chunk // 2)
    e = (chunk * buffer_size) - s
    data = data[s:e]
    data = data.tobytes()
    stream_output.write(data)
    if stream_loopback is not None:
        stream_loopback.write(data)
