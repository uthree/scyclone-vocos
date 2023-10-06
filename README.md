# Scyclone + Vocos : AI Based Voice Conversion System

this repository is voice conversion system.

successor of [VoiceChanger](https://github.com/uthree/voicechanger)

[Arch.](https://github.com/uthree/scyclone-vocos/blob/main/images/arch.png)

## Requirements

- NVIDIA GPU
- Audio devices

## Usage
1. Clone this repository and movve directory
```sh
git clone https://github.com/uthree/scyclone-vocos
cd scyclone-vocos
```

2. Install requirements
```sh
pip install -r requirements.txt
```

3. Train vocoder

Put the target speaker's audio files in one folder.
```sh
python3 train_vocoder.py <path to target speaker directory> -d cuda
```
If you using MacOS, change `-d` option to `-d mps`.

4. Check quality of vocoder (validation)
```sh
python3 inference_vocoder.py -i <path to target speaker directory>
```
or create `inputs` directory and run
```sh
python3 inference_vocoder.py
```

5. Train spectrogram convertor
put your voice files in one folder, put the target speaker's audio files in another one folder.
and run training script
```sh
python3 train_convertor.py <path to your voices directory> <path to target voices directory> -d cuda
```
If you using MacOS, change `-d` option to `-d mps`.

you can accelerate training via 16-bit floating point expression if GPU is newer than GTX10XX.

6. Check converting quality (validation)
```sh
python3 inference_vc.py -i <path to your voices directory>
```

## Realtime voice conversion
1. Check list of input device and output device.
```sh
python3 audio_device_list.py
```

2. run realtime inference.
```sh
python3 realtime_inference -i <input device ID> -o <output device ID> -d cuda
```

## References

 - [Scyclone](https://arxiv.org/abs/2005.03334)
 - [Vocos](https://arxiv.org/abs/2306.00814)