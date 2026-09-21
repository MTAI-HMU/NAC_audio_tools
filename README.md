# NAC audio tools

Paper, code and weights for every model: [awesome-codec-architectures](https://github.com/kadirnar/awesome-codec-architectures).

Batched, plot-ready **encode / decode** for neural audio codecs (NACs) and audio VAEs, behind one interface.
Takes audio at any sample rate, returns every internal representation as a plottable `[B, D, T]` feature,
and decodes codes *or* latents back to audio aligned with the input.

```python
import librosa
from utils import encoder_models as em

waves, sr = librosa.load("song.wav", sr=None)   # mono [T] at the file's own sample rate

enc = em.DACEncoder()
out = enc.encode(waves, sr=sr)       # waves: audio as [T], [B, T] or [B, C, T]
out["codes"].plot()                  # any representation: .tensor, .shape, .fps, .plot()
audio = enc.decode(out["codes"])     # [B, C, T] at the input's sample rate and length
audio = enc.decode(out["quantized_z"])  # or the decoder's continuous input
```

See [example.py](example.py) for a tour and [batch_example.py](batch_example.py) for encoding a folder of audio files.

## Install

Python 3.10, torch 2.7.1 / CUDA 12.6. [requirements.txt](requirements.txt) is the full `pip freeze`.

```bash
conda create -n audio_codecs python=3.10 pip -y
conda activate audio_codecs
export PYTHONNOUSERSITE=1   # keep ~/.local packages out

python -m pip install "numpy==1.26.4" "protobuf==3.19.6" "stable-audio-tools==0.0.20" "descript-audio-codec==1.0.0"
python -m pip install pytorch_lightning snac music2latent diffusers "git+https://github.com/facebookresearch/dacvae"
```

- SAO and SAME-L need `pytorch_lightning`. EnCodec and εar-VAE need nothing extra.
- Weights download from Hugging Face on first use.

## Models

| Model | Load | Kind | Audio | Frame rate |
| --- | --- | --- | --- | --- |
| [DAC](#dac) | `em.DACEncoder()` | RVQ codec | 44.1 kHz mono | 86.1 |
| [EnCodec](#encodec) 32 kHz (default) | `em.EnCodecEncoder()` | RVQ codec | 32 kHz mono | 50 |
| [EnCodec](#encodec) 24 kHz | `em.EnCodecEncoder("facebook/encodec_24khz")` | RVQ codec | 24 kHz mono | 75 |
| [SNAC](#snac) 44 kHz (default) | `em.SNACEncoder()` | multi-scale RVQ codec | 44.1 kHz mono | 14.4–114.8 |
| [SNAC](#snac) 32 kHz | `em.SNACEncoder("hubertsiuzdak/snac_32khz")` | multi-scale RVQ codec | 32 kHz mono | 10.4–83.3 |
| [Stable Audio Open VAE](#sao) | `em.SAOEncoder()` | VAE | 44.1 kHz stereo | 21.5 |
| [SAME-L](#same-l) | `em.SAMLEncoder()` | autoencoder | 44.1 kHz stereo | 10.8 |
| [Music2Latent](#music2latent) | `em.Music2LatentEncoder()` | consistency autoencoder | 44.1 kHz mono | 10.8 |
| [DACVAE](#dacvae) | `em.DACVAEEncoder()` | VAE | 48 kHz mono | 25 |
| [ACE-Step 1.5 VAE](#ace-step-15-vae) | `em.ACEStep15Encoder()` | VAE | 48 kHz stereo | 25 |
| [ACE-Step v1 music DCAE](#ace-step-v1-music-dcae) | `em.ACEStepDCAEEncoder()` | mel autoencoder + vocoder | 44.1 kHz stereo | 10.8 |
| [εar-VAE](#ear-vae) v2 48 kHz (default) | `em.EARVAEEncoder()` | VAE | 48 kHz stereo | 50 |
| [εar-VAE](#ear-vae) 44.1 kHz | `em.EARVAEEncoder("ear_vae_44k")` | VAE | 44.1 kHz stereo | 43.1 |

Frame rate is frames per second of the returned features. Every constructor also takes `device` (default `"cuda:0"`)
and `batch_size` (default 8 for the codecs, 4 for the autoencoders, 1 for SAME-L).

### Shapes and decoding

| Model | Representation | Shape | Decodable |
| --- | --- | --- | --- |
| [DAC](#dac) | `encoder_z` | `[B, 1024, T]` | ✓ quantized first |
| | `latents` | `[B, 72, T]` (9 codebooks × 8) | ✓ quantized first |
| | `codes` | `[B, 9, T]` | **✓ default** |
| | `quantized_z` | `[B, 1024, T]` | ✓ |
| [EnCodec](#encodec) | `encoder_z` | `[B, 128, T]` | ✓ quantized first |
| | `codes` | `[B, 4, T]` at 32 kHz<br>`[B, 32, T]` at 24 kHz | **✓ default** |
| | `quantized_z` | `[B, 128, T]` | ✓ |
| [SNAC](#snac) | `encoder_z` | `[B, 1024, T]` | ✓ quantized first |
| | `codes` | `[B, 4, T]` (4 levels) | **✓ default** |
| | `quantized_z` | `[B, 1024, T]` | ✓ |
| [Stable Audio Open VAE](#sao) | `mu` | `[B, 64, T]` | **✓ default** |
| | `std` | `[B, 64, T]` | ✗ |
| | `sample` | `[B, 64, T]` | ✓ |
| [SAME-L](#same-l) | `pre_softnorm` | `[B, 256, T]` | ✓ normalized first |
| | `latent` | `[B, 256, T]` | **✓ default** |
| [Music2Latent](#music2latent) | `features` | `[B, 8192, T]` | ✗ |
| | `latent` | `[B, 64, T]` | **✓ default** |
| [DACVAE](#dacvae) | `mu` | `[B, 128, T]` | **✓ default** |
| | `std` | `[B, 128, T]` | ✗ |
| | `sample` | `[B, 128, T]` | ✓ |
| [ACE-Step 1.5 VAE](#ace-step-15-vae) | `mu` | `[B, 64, T]` | **✓ default** |
| | `std` | `[B, 64, T]` | ✗ |
| | `sample` | `[B, 64, T]` | ✓ |
| [ACE-Step v1 music DCAE](#ace-step-v1-music-dcae) | `latent` | `[B, 128, T]` (8 × 16, flattened) | **✓ default** |
| [εar-VAE](#ear-vae) | `mu` | `[B, 64, T]` | **✓ default** |
| | `std` | `[B, 64, T]` | ✗ |
| | `sample` | `[B, 64, T]` | ✓ |

**Shape** is `.tensor.shape`: B clips, D dimensions, T frames at the frame rate (for SNAC, its finest rate).
Both variants of a model share these shapes, except EnCodec's `codes`.

**Decodable** says what `enc.decode()` does with that representation:

- **✓** The model's own decoder takes it as it is. **✓ default** is the one used when you pass the whole `encode()`
  output.
- **✓ quantized first** The model can't decode it directly. `decode()` first snaps it to the codebook, the same step
  that produces `codes`, then decodes that. The audio is the same as decoding `codes`, not a more detailed version.
- **✓ normalized first** SAME-L's decoder takes `latent` only. `decode()` first runs `pre_softnorm` through the model's
  SoftNorm, which turns it into `latent`, so the audio is the same as decoding `latent`.
- **✗** Can't be decoded; `decode()` raises a `ValueError`.

## API

Every encoder has the same two methods:

| Call | Parameters | Returns |
| --- | --- | --- |
| `enc.encode(waves, sr, mono, seed)` | `waves`: audio as `[T]`, `[B, T]` or `[B, C, T]`<br>`sr=44_100`: its sample rate<br>`mono=False`: mix stereo down first<br>`seed=0`: item *i* draws its VAE `sample` from `seed + i` | A dict of `EncodedFeature`s, one per representation |
| `enc.decode(features, seed)` | `features`: one feature, e.g. `out["codes"]`, or the whole `encode()` output to decode `enc.decode_default`<br>`seed=0`: stochastic decoders decode item *i* with `seed + i` | `[B, C, T]` audio at the input's sample rate and length |

Each `EncodedFeature` has:

| Member | What it is |
| --- | --- |
| `.tensor` | `[B, D, T]` on the CPU: float32, or int64 for `codes` |
| `.shape` | `.tensor.shape` |
| `.fps` | Frames per second, from the actual output length |
| `.is_codes` | `True` for integer tokens |
| `.numpy()` | `.tensor` as a NumPy array |
| `.plot(mode="heatmap", index=0, ax=None)` | Plots batch item `index` as a `"heatmap"`, or its per-frame `"mean"` or `"norm"`. Pass `ax` to draw into a subplot. Returns `(fig, ax)` |

### Conventions

- **Input:** a single stereo clip is `[1, 2, T]`; `[2, T]` is two mono clips. Audio is resampled to the
  model's rate and right-padded to whole frames.
- **Channels:** mono models mix stereo down; stereo models get mono input as dual-mono.
- **Seeding:** item *i* uses `seed + i`, for VAE `sample`s and the stochastic decoders (SNAC, SAME-L,
  Music2Latent, DACVAE). Results don't depend on `batch_size`. To reproduce item 1 on its own, pass `seed=1`.
- **Determinism:** TF32 is off and cuDNN is deterministic (with TF32, DAC flipped RVQ codes on identical input).
  Codes are bit-identical across runs and batch sizes; floats agree to about 1e-5, except SAME-L in fp16.

## What the representations are

Every model turns audio into frames, one every 1 / frame rate seconds. They differ in what a frame holds.

**Tokens** (DAC, EnCodec, SNAC). Each frame becomes a few integers, each an index into a learned codebook. They're
chosen in stages: the first codebook approximates the frame and each next one encodes what's still missing.
Neighbouring indices aren't similar sounds. Tokens are what audio language models (e.g. MusicGen) predict.

| Name | What it is |
| --- | --- |
| `codes` | The tokens: `[B, codebooks, T]` integers |
| `encoder_z` | The encoder's output before quantization; the most detailed view |
| `quantized_z` | The vector rebuilt from the tokens (sum of the chosen codebook entries), which the decoder receives. Same information as `codes` |
| `latents` (DAC) | The 8-dimension per-codebook vectors used to look up the nearest codebook entry |

**Latents** (continuous autoencoders). Each frame becomes a vector of real numbers. Nothing is rounded, so latents
keep more detail, and similar sounds get nearby vectors.

- **VAEs** (SAO, DACVAE, ACE-Step 1.5, εar-VAE) describe each frame as a small cloud of likely points:

  | Name | What it is |
  | --- | --- |
  | `mu` | The cloud's centre, the best estimate. Deterministic: use it for analysis and decoding |
  | `std` | The cloud's width per dimension (the model's uncertainty). Usually small |
  | `sample` | `mu + std × noise`, one seeded point from the cloud. Diffusion models train on these. Sounds almost the same as `mu` |

- **Other autoencoders** (SAME-L, Music2Latent, ACE-Step v1 DCAE) output one plain `latent` per frame. Also exposed:
  `pre_softnorm` (SAME-L, before the final normalization) and `features` (Music2Latent, before the bottleneck).

Fewer frames per second or fewer dimensions means more compression and less detail.

## Model details

Each heading links to the model's entry in awesome-codec-architectures.

<a id="dac"></a>
### [DAC — Descript Audio Codec](https://github.com/kadirnar/awesome-codec-architectures#dac)

```python
em.DACEncoder()                                     # 44.1 kHz model (default)
em.DACEncoder(weights="path/to/weights_44khz.pth")  # a local 44.1 kHz checkpoint
```
- `encoder_z` [B, 1024, T] · `latents` [B, 72, T] (9 codebooks × 8) · `codes` [B, 9, T] · `quantized_z` [B, 1024, T]
- Decodes from `codes` or `quantized_z`. Default: `codes`.
- `encoder_z` and `latents` decode too, but are quantized first, so they give the same audio as `codes`.

<a id="encodec"></a>
### [EnCodec](https://github.com/kadirnar/awesome-codec-architectures#encodec)

```python
em.EnCodecEncoder()                                     # 32 kHz music model, MusicGen's tokenizer (default)
em.EnCodecEncoder(model_id="facebook/encodec_24khz")    # 24 kHz general-audio model
```
- `encoder_z` [B, 128, T] · `codes` [B, Q, T] · `quantized_z` [B, 128, T]
- Encodes at the highest bandwidth: Q = 4 codebooks at 32 kHz, 32 at 24 kHz.
- Decodes from `codes` or `quantized_z`. Default: `codes`.
- `encoder_z` decodes too, but is quantized first, so it gives the same audio as `codes`.
- The 48 kHz stereo model encodes in normalized 1 s chunks and is not supported.

<a id="snac"></a>
### [SNAC — Multi-Scale Neural Audio Codec](https://github.com/kadirnar/awesome-codec-architectures#snac)

```python
em.SNACEncoder()                                      # 44 kHz music model (default)
em.SNACEncoder(model_id="hubertsiuzdak/snac_32khz")   # 32 kHz music model
```
- `encoder_z` [B, 1024, T] · `codes` [B, 4, T] · `quantized_z` [B, 1024, T]
- `codes` stacks 4 levels, coarse to fine. Level *i* has 8, 4, 2, 1 times fewer frames and is repeated to the
  finest rate, so `codes[:, i, ::stride]` recovers it.
- Decodes from `codes` or `quantized_z`. Default: `codes`. The decoder adds noise, so it is seeded per item.
- `encoder_z` decodes too, but is quantized first, so it gives the same audio as `codes`.

<a id="sao"></a>
### [Stable Audio Open VAE](https://github.com/kadirnar/awesome-codec-architectures#stable-audio-autoencoder)

```python
em.SAOEncoder()
```
- `mu` · `std` · `sample` [B, 64, T]
- Decodes from `mu` or `sample`. Default: `mu`.
- Loads only the VAE, not the diffusion model ([utils/sao_vae.py](utils/sao_vae.py)).
- Gated on Hugging Face: request access, then log in with `hf auth login`.

<a id="same-l"></a>
### [SAME-L — Semantically-Aligned Music Autoencoder](https://github.com/kadirnar/awesome-codec-architectures#same)

```python
em.SAMLEncoder()                                  # fp16 on CUDA, one clip at a time (default)
em.SAMLEncoder(half=False, batch_size=4)          # fp32: batch-invariant, ~3.4 GB of weights
```
- `pre_softnorm` [B, 256, T] (before SoftNorm) · `latent` [B, 256, T]
- Decodes from `latent`. The decoder draws random numbers, so it is seeded per item.
- `pre_softnorm` decodes too, but goes through SoftNorm first, so it gives the same audio as `latent`.
- Encoding is deterministic: the train-time mask noise is off.
- fp16 results shift by about 1% with `batch_size`, hence the default of 1. Use `half=False` for larger batches.

<a id="music2latent"></a>
### [Music2Latent](https://github.com/kadirnar/awesome-codec-architectures#music2latent)

```python
em.Music2LatentEncoder()                          # one-step decoding (default)
em.Music2LatentEncoder(denoising_steps=3)         # more decoding steps
```
- `features` [B, 8192, T] (before the bottleneck) · `latent` [B, 64, T]
- Decodes from `latent`. Decoding starts from noise, so it is seeded per item.
- Runs in fp32: the package's fp16 autocast makes results batch-dependent and "can cause nans". This sets
  `music2latent.inference.mixed_precision = False` for the whole process.
- Weights download into the package's `site-packages` folder, not the Hugging Face cache.

<a id="dacvae"></a>
### [DACVAE](https://github.com/kadirnar/awesome-codec-architectures#dacvae)

```python
em.DACVAEEncoder()                                # facebook/dacvae-watermarked
```
- `mu` · `std` · `sample` [B, 128, T]
- Decodes from `mu` or `sample`. Default: `mu`.
- The decoder embeds an AudioSeal watermark with a random 16-bit message, so it is seeded per item.

<a id="ace-step-15-vae"></a>
### [ACE-Step 1.5 VAE](https://github.com/kadirnar/awesome-codec-architectures#acestep15-vae)

```python
em.ACEStep15Encoder()                             # the vae/ subfolder of ACE-Step/Ace-Step1.5
```
- `mu` · `std` · `sample` [B, 64, T]
- Decodes from `mu` or `sample`. Default: `mu`.
- Loads only the VAE, with `diffusers.AutoencoderOobleck`.

<a id="ace-step-v1-music-dcae"></a>
### [ACE-Step v1 music DCAE](https://github.com/kadirnar/awesome-codec-architectures#acestep-dcae)

```python
em.ACEStepDCAEEncoder()                           # music_dcae_f8c8 + music_vocoder of ACE-Step-v1-3.5B
```
- `latent` [B, 128, T] (the model's [B, 8, 16, T] latent, flattened)
- Decodes `latent` to a mel spectrogram, then to audio with ACE-Step's HiFi-GAN vocoder.
- Downloads only the DCAE and vocoder, not the 3.5B generator. The mel front end and vocoder are vendored in
  [third_party/ace_step](third_party/ace_step), because the `acestep` package pins versions
  (e.g. `transformers==4.50.0`) that clash with this environment.

<a id="ear-vae"></a>
### [εar-VAE](https://github.com/kadirnar/awesome-codec-architectures#ear-vae)

```python
em.EARVAEEncoder()                                # 48 kHz v2 model (default)
em.EARVAEEncoder(variant="ear_vae_44k")           # 44.1 kHz model
```
- `mu` · `std` · `sample` [B, 64, T]
- Decodes from `mu` or `sample`. Default: `mu`.
- The model code is imported from a pinned revision of the Hugging Face repo.
- The v2 config declares a decoder transformer that the checkpoint doesn't contain, so it is built only when its
  weights are present.

## Verification

Each wrapper was checked against the model's own reference pipeline on the same input (relative L2 difference):

| Model | Reference | Difference |
| --- | --- | --- |
| DAC | `DAC.encode` → `DAC.decode` | 3e-7 |
| EnCodec 32 kHz / 24 kHz | `EncodecModel(x, bandwidth=max)` | 0 / 0 |
| SNAC | `SNAC.decode(SNAC.encode(x))` | 0.9% = its own decode-to-decode noise |
| SAME-L | `model.decode(model.encode(x))` | latents 0; audio 0.33% = its own decode-to-decode noise |
| ACE-Step 1.5 VAE | `AutoencoderOobleck(x, sample_posterior=False)` | 0 |
| ACE-Step v1 DCAE | upstream `MusicDCAE.encode` / `.decode` | 0 / 4e-6 |
| εar-VAE v2 | `EAR_VAE.decode(EAR_VAE.encode(x, use_sample=False))` | 0 |

Every model was also round-tripped with 44.1 kHz stereo input, a different recording in each channel.
Output length and rate matched the input, and each output channel matched its own input channel.
