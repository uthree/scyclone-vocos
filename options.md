# Options

## train_vocoder.py

| Option name | Alias | Description |
|---| --- | ---|
|`--generator-path`| `-gp`| path to  generator |
|`--discriminator-path`| `-dp` | path to discriminator |
|`--device`| `-d` | set training device. you can use `cpu`, `cuda` or `mps` |
|`--epoch`|  `-e` | number of epochs |
|`--batch`| `-b`| batch size. default is `64`, decrase this if not enough memory |
|`--learning-rate`| `-lr` | learning rate |
|`-length`|`-len` | data length. default is `65535` |
|`-max-data`| `-m` | max number of data file. |
|  |`-fp16 True`| use 16-bit floating point (deprecated) |



## train_convertor.py
| Option name | Alias | Description |
|---| --- | ---|
|`--device`| `-d` | set training device. you can use `cpu`, `cuda` or `mps` |
|`--epoch`|  `-e` | number of epochs |
|`--batch`| `-b`| batch size. default is `64`, decrase this if not enough memory |
|`--learning-rate`| `-lr` | learning rate |
|`-length`|`-len` | data length. default is `65535` |
|`-max-data`| `-m` | max number of data file. |
|  |`-fp16 True`| use 16-bit floating point |
|`--consistency`| | weight of consistency loss. default is `5.0` |
|`--identity`| | weight of identity loss. default is `1.0` |
|`--feature-matching`| | weight of feature-matching loss. default is `5.0`|
|`--preview True`| | save preview in training | 

## realtime_inference.py
| Option name | Alias | Description |
|---| --- | ---|
|`-device`| `-d` | set inferencing device. you can use `cpu`, `cuda` or `mps` |
|`--output`| `-o` | output audio device ID |
|`--input`| `-i` | input audio device ID |
|`--loopback`| `-l` | loopback(second output) audio device ID |
|`--gain`| `-g` | output gain(dB) |
|`--threshold`| `-thr` | conversion threshold |
|`--chunk`| `-c` | chunk size. default is `3072` |
|`--buffersize`,| `-b` | buffer size. default is `8`|
|| `-fp16 True`| use 16-bit floatation point (deprecated)|
