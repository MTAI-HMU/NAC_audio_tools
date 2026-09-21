"""Batched, plot-ready encode/decode for neural audio codecs and audio VAEs. See README.md for the models."""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio

SR = 44_100


@dataclass
class EncodedFeature:
    """
    One representation: [B, D, T] float features or [B, Q, T] integer codes.

    source_sr / source_length describe the encoded audio, so decode() returns audio that lines up with it.
    """
    tensor: torch.Tensor
    name: str
    fps: float
    source_sr: int
    source_length: int

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def is_codes(self) -> bool:
        return not self.tensor.is_floating_point()

    def numpy(self):
        return self.tensor.numpy()

    def plot(self, mode="heatmap", index=0, ax=None, figsize=(12, 4), title=None):
        """
        Plot batch item `index` into `ax` (e.g. one cell of a grid), or into a new figure.

        mode: "heatmap" (dimension x time), "mean" or "norm" (over dimensions, vs time)
        """
        x = self.tensor[index].float()
        duration = x.shape[-1] / self.fps
        time = torch.arange(x.shape[-1]) / self.fps

        new_figure = ax is None
        fig, ax = plt.subplots(figsize=figsize) if new_figure else (ax.figure, ax)
        if mode == "heatmap":
            im = ax.imshow(x, aspect="auto", origin="lower", interpolation="nearest",
                           extent=[0, duration, 0, x.shape[0]])
            ax.set_ylabel("Codebook" if self.is_codes else "Feature dimension")
            fig.colorbar(im, ax=ax, label="Code index" if self.is_codes else "Value")
        elif mode == "mean":
            ax.plot(time, x.mean(dim=0))
            ax.set_ylabel("Mean value")
        elif mode == "norm":
            ax.plot(time, x.norm(dim=0))
            ax.set_ylabel("L2 norm")
        else:
            raise ValueError(f"unknown mode {mode!r}; use 'heatmap', 'mean' or 'norm'")

        ax.set(xlabel="Time (s)", title=title or f"{self.name} (item {index})")
        if new_figure:
            fig.tight_layout()
        return fig, ax


def _deterministic():
    # TF32 and non-deterministic cuDNN kernels make results vary run to run (and flip DAC codes).
    return torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False)


def _frozen(model, device, dtype=None):
    model = model.to(device) if dtype is None else model.to(device, dtype)  # diffusers warns on any dtype cast
    return model.eval().requires_grad_(False)


class _Encoder:
    """
    Shared input handling, resampling, padding, batching and seeding.

    Subclasses set `self.model` and define:
        sample_rate         native sample rate
        channels            1 = mono model, 2 = stereo model
        pad_multiple        input is right-padded to a multiple of this many samples
        decode_default      feature decode() uses when given a whole encode() output
        stochastic_decoder  True if the decoder draws random noise internally
        _encode(x)          [B, channels, T] audio -> {name: tensor}; random draws go through self._randn_like
        _decode(name, z)    one encoded tensor -> [B, channels, T] audio at sample_rate

    Randomness is seeded per item: item i of an encode()/decode() call uses seed + i, so results are
    reproducible and don't depend on batch_size.
    """
    sample_rate: int
    channels: int
    pad_multiple: int
    decode_default: str
    stochastic_decoder = False
    dtype = torch.float32

    def __init__(self, device, batch_size):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        self.batch_size = batch_size
        self._item_seeds = range(0)  # seeds of the batch being encoded, set by encode()

    @torch.inference_mode()
    def encode(self, waves, sr: int = SR, mono: bool = False, seed: int = 0) -> dict[str, EncodedFeature]:
        """
        waves: audio at `sr` Hz as [T] (one mono clip), [B, T] (mono batch) or [B, C, T], C in {1, 2}.
        mono:  mix stereo down first. Mono models always mix down; stereo models get dual-mono.
        seed:  item i draws its random posterior `sample` (VAEs) from seed + i.
        """
        x = torch.as_tensor(waves, dtype=torch.float32)
        if x.ndim == 1:
            x = x[None]  # [T] -> [1, T]
        if x.ndim == 2:
            x = x[:, None]  # [B, T] -> [B, 1, T]
        if x.ndim != 3 or x.shape[1] not in (1, 2) or x.shape[-1] == 0:
            raise ValueError(f"expected [T], [B, T] or [B, C, T] with C in {{1, 2}}, got {tuple(x.shape)}")

        if mono or self.channels == 1:
            x = x.mean(dim=1, keepdim=True)
        x = x.expand(-1, self.channels, -1)

        outs = []
        with _deterministic():
            for start in range(0, len(x), self.batch_size):
                batch = torchaudio.functional.resample(x[start:start + self.batch_size].to(self.device), sr, self.sample_rate)
                batch = F.pad(batch, (0, -batch.shape[-1] % self.pad_multiple))  # right-pad to whole frames
                self._item_seeds = range(seed + start, seed + start + len(batch))
                out = self._encode(batch.to(self.dtype))
                outs.append({k: (v.float() if v.is_floating_point() else v).cpu() for k, v in out.items()})

        seconds = batch.shape[-1] / self.sample_rate  # padded duration, the same for every batch
        return {
            k: EncodedFeature(torch.cat([o[k] for o in outs]), k, outs[0][k].shape[-1] / seconds, sr, x.shape[-1])
            for k in outs[0]
        }

    @torch.inference_mode()
    def decode(self, features: EncodedFeature | dict[str, EncodedFeature], seed: int = 0) -> torch.Tensor:
        """
        Reconstruct audio [B, C, T] from codes or latents, at the sample rate and length that was encoded.

        features: one feature (e.g. out["codes"]) or a whole encode() output, which decodes `decode_default`.
                  Any number of frames works, e.g. features stitched together from several encode() calls.
        seed:     stochastic decoders (SNAC, SAME-L, Music2Latent, DACVAE) decode item i with seed + i.
        """
        f = features[self.decode_default] if isinstance(features, dict) else features
        # Decoders need whole blocks of pad_multiple samples (SNAC: 32 frames, others: 1). Fill the last
        # block by repeating the last frame; the extra audio is trimmed below.
        tensor, block = f.tensor, max(1, round(self.pad_multiple * f.fps / self.sample_rate))
        if tensor.shape[-1] % block:
            tensor = torch.cat([tensor, tensor[..., -1:].expand(*tensor.shape[:-1], block - tensor.shape[-1] % block)], -1)

        wavs = []
        with _deterministic():
            for start in range(0, len(tensor), self.batch_size):
                z = tensor[start:start + self.batch_size]
                z = z.to(self.device, self.dtype if z.is_floating_point() else z.dtype)
                if self.stochastic_decoder:  # its noise can't be drawn per item, so decode item by item
                    wav = torch.cat([self._seeded(seed + start + i, self._decode, f.name, z[i:i + 1])
                                     for i in range(len(z))])
                else:
                    wav = self._decode(f.name, z)
                wavs.append(torchaudio.functional.resample(wav.float(), self.sample_rate, f.source_sr).cpu())
        return torch.cat(wavs)[..., :f.source_length]

    def _encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _decode(self, name: str, z: torch.Tensor) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} cannot decode {name!r}")

    def _seeded(self, seed, fn, *args):
        """Call fn with torch's global RNG seeded, restoring the previous RNG state afterwards."""
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            return fn(*args)

    def _randn_like(self, t):
        """Standard normal noise like `t`, each item drawn from its own seed."""
        return torch.stack([
            torch.randn(t.shape[1:], generator=torch.Generator(t.device).manual_seed(s), device=t.device, dtype=t.dtype)
            for s in self._item_seeds
        ])

    def _gaussian(self, mean, scale, eps=1e-4):
        """Posterior of an Oobleck-style VAE bottleneck: std = softplus(scale) + eps."""
        std = F.softplus(scale) + eps
        return dict(mu=mean, std=std, sample=mean + self._randn_like(mean) * std)


# =====================================================================
# Discrete codecs
# =====================================================================

class DACEncoder(_Encoder):
    """
    Descript Audio Codec, 44.1 kHz, mono (stereo input is mixed down), ~86 fps.

        encoder_z    [B, 1024, T]  encoder output before RVQ
        latents      [B,   72, T]  continuous RVQ projections (9 codebooks x 8 dims)
        codes        [B,    9, T]  discrete RVQ indices
        quantized_z  [B, 1024, T]  quantized representation fed to the decoder

    Decodes from any of them.
    """
    sample_rate, channels, pad_multiple, decode_default = 44_100, 1, 512, "codes"

    def __init__(self, device="cuda:0", weights=None, batch_size=8):
        import dac

        super().__init__(device, batch_size)
        self.model = _frozen(dac.DAC.load(weights or dac.utils.download(model_type="44khz")), self.device)

    def _encode(self, x):
        encoder_z = self.model.encoder(x)
        quantized_z, codes, latents, _, _ = self.model.quantizer(encoder_z)
        return dict(encoder_z=encoder_z, latents=latents, codes=codes, quantized_z=quantized_z)

    def _decode(self, name, z):
        q = self.model.quantizer
        if name == "codes":
            z = q.from_codes(z)[0]
        elif name == "latents":
            z = q.from_latents(z)[0]
        elif name == "encoder_z":
            z = q(z)[0]
        elif name != "quantized_z":
            return super()._decode(name, z)
        return self.model.decode(z)


class EnCodecEncoder(_Encoder):
    """
    Meta EnCodec, mono: 32 kHz (MusicGen's music codec, 50 fps) or 24 kHz (general audio, 75 fps).

        encoder_z    [B, 128, T]  encoder output before RVQ
        codes        [B,   Q, T]  RVQ indices at the highest bandwidth (Q = 4 at 32 kHz, 32 at 24 kHz)
        quantized_z  [B, 128, T]  quantized representation fed to the decoder

    Decodes from any of them. The 48 kHz stereo model encodes normalized 1 s chunks and is not supported.
    """
    channels, decode_default = 1, "codes"

    def __init__(self, model_id="facebook/encodec_32khz", device="cuda:0", batch_size=8):
        from transformers import EncodecModel

        super().__init__(device, batch_size)
        self.model = _frozen(EncodecModel.from_pretrained(model_id), self.device)
        cfg = self.model.config
        if cfg.chunk_length_s is not None or cfg.audio_channels != 1:
            raise ValueError(f"{model_id}: only the unchunked mono EnCodec models (24 kHz, 32 kHz) are supported")
        self.sample_rate, self.pad_multiple = cfg.sampling_rate, math.prod(cfg.upsampling_ratios)
        self.bandwidth = max(cfg.target_bandwidths)

    def _encode(self, x):
        encoder_z = self.model.encoder(x)
        codes = self.model.quantizer.encode(encoder_z, self.bandwidth)  # [Q, B, T]
        return dict(encoder_z=encoder_z, codes=codes.transpose(0, 1), quantized_z=self.model.quantizer.decode(codes))

    def _decode(self, name, z):
        q = self.model.quantizer
        if name == "codes":
            z = q.decode(z.transpose(0, 1))
        elif name == "encoder_z":
            z = q.decode(q.encode(z, self.bandwidth))
        elif name != "quantized_z":
            return super()._decode(name, z)
        return self.model.decoder(z)


class SNACEncoder(_Encoder):
    """
    SNAC multi-scale codec, mono: 44 kHz (default) or 32 kHz music/SFX models.

        encoder_z    [B, 1024, T]  encoder output before RVQ (~115 fps at 44 kHz)
        codes        [B,    4, T]  RVQ indices per level, coarse to fine; level i has vq_strides[i]
                                   times fewer frames and is repeated to the finest rate
        quantized_z  [B, 1024, T]  quantized representation fed to the decoder

    Decodes from any of them.
    """
    channels, decode_default = 1, "codes"
    stochastic_decoder = True  # its decoder adds random noise (NoiseBlock)

    def __init__(self, model_id="hubertsiuzdak/snac_44khz", device="cuda:0", batch_size=8):
        from snac import SNAC

        super().__init__(device, batch_size)
        self.model = _frozen(SNAC.from_pretrained(model_id), self.device)
        self.sample_rate, self.strides = self.model.sampling_rate, self.model.vq_strides
        # What SNAC's own preprocess() pads to: the hop times the coarsest stride / attention window.
        self.pad_multiple = int(self.model.hop_length) * math.lcm(self.strides[0], self.model.attn_window_size or 1)

    def _encode(self, x):
        encoder_z = self.model.encoder(x)
        quantized_z, codes = self.model.quantizer(encoder_z)
        codes = torch.stack([c.repeat_interleave(s, dim=-1) for c, s in zip(codes, self.strides)], dim=1)
        return dict(encoder_z=encoder_z, codes=codes, quantized_z=quantized_z)

    def _decode(self, name, z):
        if name == "codes":
            return self.model.decode([z[:, i, ::s] for i, s in enumerate(self.strides)])
        if name == "encoder_z":
            z = self.model.quantizer(z)[0]
        elif name != "quantized_z":
            return super()._decode(name, z)
        return self.model.decoder(z)


# =====================================================================
# Continuous autoencoders
# =====================================================================

class SAOEncoder(_Encoder):
    """
    Stable Audio Open VAE, 44.1 kHz stereo, ~21.5 fps.

        mu      [B, 64, T]  posterior mean
        std     [B, 64, T]  posterior standard deviation
        sample  [B, 64, T]  random posterior sample, seeded per item

    Decodes from mu or sample.
    """
    sample_rate, channels, pad_multiple, decode_default = 44_100, 2, 2048, "mu"

    def __init__(self, device="cuda:0", batch_size=4):
        from .sao_vae import load_sao_v1_vae

        super().__init__(device, batch_size)
        self.model = load_sao_v1_vae(self.device)

    def _encode(self, x):
        return self._gaussian(*self.model.encoder(x).chunk(2, dim=1))

    def _decode(self, name, z):
        return self.model.decode(z) if name in ("mu", "sample") else super()._decode(name, z)


class SAMLEncoder(_Encoder):
    """
    Stability SAME-L, 44.1 kHz stereo, ~10.8 fps.

        pre_softnorm  [B, 256, T]  encoder output before SoftNorm
        latent        [B, 256, T]  final SoftNorm representation

    Decodes from either. Runs in fp16 on CUDA unless half=False (852M parameters: ~3.4 GB of fp32 weights).
    fp16 results shift by ~1% with batch_size, so keep batch_size=1 or use half=False for batch-invariant output.
    """
    sample_rate, channels, pad_multiple, decode_default = 44_100, 2, 4096, "latent"
    stochastic_decoder = True  # its decoder draws random numbers

    def __init__(self, device="cuda:0", batch_size=1, half=True):
        import stable_audio_tools.models.transformer as sat_transformer
        from stable_audio_tools import get_pretrained_model

        super().__init__(device, batch_size)
        self.dtype = torch.float16 if half and self.device.type == "cuda" else torch.float32

        if self.device.type == "cpu":
            # flash-attn has no CPU kernels; force stable-audio-tools onto its PyTorch attention path.
            for name in ("flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_varlen_func",
                         "index_first_axis", "pad_input", "unpad_input"):
                setattr(sat_transformer, name, None)

        model, self.config = get_pretrained_model("stabilityai/SAME-L")
        self.model = _frozen(model, self.device, self.dtype)

        # Turn off the train-time mask noise so encodings are deterministic.
        for module in self.model.modules():
            if hasattr(module, "mask_noise"):
                module.mask_noise = 0.0
            if hasattr(module, "checkpointing"):
                module.checkpointing = False

    def _encode(self, x):
        # SAME-L's own model.encode(), unrolled to also expose the pre-SoftNorm output.
        pre_softnorm = self.model.encoder(self.model.pretransform.encode(x))
        return dict(pre_softnorm=pre_softnorm, latent=self.model.bottleneck.encode(pre_softnorm))

    def _decode(self, name, z):
        if name == "pre_softnorm":
            z = self.model.bottleneck.encode(z)
        elif name != "latent":
            return super()._decode(name, z)
        return self.model.decode(z)


class Music2LatentEncoder(_Encoder):
    """
    Sony CSL Music2Latent consistency autoencoder, 44.1 kHz mono, ~10.8 fps. CC-BY-NC 4.0.

        features  [B, 8192, T]  encoder features before the bottleneck
        latent    [B,   64, T]  final latent

    Decodes from latent, with a one-step consistency sampler that starts from (seeded) noise.
    """
    sample_rate, channels, pad_multiple, decode_default = 44_100, 1, 4096, "latent"
    stochastic_decoder = True  # decoding starts from random noise

    def __init__(self, device="cuda:0", batch_size=4, denoising_steps=1):
        import music2latent.inference
        from music2latent import EncoderDecoder

        # Its fp16 autocast makes results depend on batch_size (and "can cause nans" per its own comment);
        # fp32 is batch-invariant at no measurable cost. This is a module-level switch in music2latent.
        music2latent.inference.mixed_precision = False
        super().__init__(device, batch_size)
        self.model = EncoderDecoder(device=self.device)
        self.denoising_steps = denoising_steps

    def _encode(self, x):
        x = F.pad(x[:, 0], (0, 3 * 512))  # its STFT front end needs 3 extra hops, and crops anything past them
        kw = dict(max_batch_size=len(x))
        return dict(features=self.model.encode(x, extract_features=True, **kw), latent=self.model.encode(x, **kw))

    def _decode(self, name, z):
        if name != "latent":
            return super()._decode(name, z)
        return self.model.decode(z, denoising_steps=self.denoising_steps, max_batch_size=len(z))[:, None]


class DACVAEEncoder(_Encoder):
    """
    Meta DACVAE (DAC with a VAE bottleneck), 48 kHz mono, 25 fps.

        mu, std, sample  [B, 128, T]  posterior mean, standard deviation and a random sample

    Decodes from mu or sample. This checkpoint's decoder adds an inaudible AudioSeal watermark.
    """
    channels, decode_default = 1, "mu"
    stochastic_decoder = True  # the watermark carries a random 16-bit message per item

    def __init__(self, device="cuda:0", batch_size=4):
        from dacvae import DACVAE

        super().__init__(device, batch_size)
        self.model = _frozen(DACVAE.load("facebook/dacvae-watermarked"), self.device)
        self.sample_rate, self.pad_multiple = self.model.sample_rate, self.model.hop_length

    def _encode(self, x):
        return self._gaussian(*self.model.quantizer.in_proj(self.model.encoder(x)).chunk(2, dim=1))

    def _decode(self, name, z):
        return self.model.decode(z) if name in ("mu", "sample") else super()._decode(name, z)


class ACEStep15Encoder(_Encoder):
    """
    ACE-Step 1.5 VAE (Oobleck, loaded with diffusers), 48 kHz stereo, 25 fps.

        mu, std, sample  [B, 64, T]  posterior mean, standard deviation and a random sample

    Decodes from mu or sample.
    """
    sample_rate, channels, pad_multiple, decode_default = 48_000, 2, 1920, "mu"

    def __init__(self, device="cuda:0", batch_size=4):
        from diffusers import AutoencoderOobleck

        super().__init__(device, batch_size)
        vae = AutoencoderOobleck.from_pretrained("ACE-Step/Ace-Step1.5", subfolder="vae")
        self.model = _frozen(vae, self.device)

    def _encode(self, x):
        return self._gaussian(*self.model.encoder(x).chunk(2, dim=1))

    def _decode(self, name, z):
        return self.model.decode(z).sample if name in ("mu", "sample") else super()._decode(name, z)


class ACEStepDCAEEncoder(_Encoder):
    """
    ACE-Step v1 music DCAE: a deep-compression autoencoder over stereo log-mel spectrograms,
    44.1 kHz stereo, ~10.8 fps. Decoding goes DCAE -> mel -> ACE-Step's HiFi-GAN vocoder.

        latent  [B, 128, T]  8 latent channels x 16 mel rows, flattened

    Decodes from latent.
    """
    sample_rate, channels, pad_multiple, decode_default = 44_100, 2, 8 * 512, "latent"
    repo = "ACE-Step/ACE-Step-v1-3.5B"
    # MusicDCAE constants: mel range for min-max normalization, and latent shift/scale.
    mel_min, mel_max, shift, scale = -11.0, 3.0, -1.9091, 0.1786

    def __init__(self, device="cuda:0", batch_size=4):
        from diffusers import AutoencoderDC

        from third_party.ace_step import ADaMoSHiFiGANV1

        super().__init__(device, batch_size)
        self.model = _frozen(AutoencoderDC.from_pretrained(self.repo, subfolder="music_dcae_f8c8"), self.device)
        self.vocoder = _frozen(ADaMoSHiFiGANV1.from_pretrained(self.repo, subfolder="music_vocoder"), self.device)

    def _encode(self, x):
        mel = self.vocoder.mel_transform(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # [B, 2, 128, frames]
        mel = (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1
        latent = (self.model.encoder(mel) - self.shift) * self.scale  # [B, 8, 16, T]
        return dict(latent=latent.flatten(1, 2))

    def _decode(self, name, z):
        if name != "latent":
            return super()._decode(name, z)
        mel = self.model.decoder(z.unflatten(1, (8, 16)) / self.scale + self.shift)
        mel = (mel + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min
        return torch.cat([self.vocoder.decode(mel[:, c]) for c in range(mel.shape[1])], dim=1)


class EARVAEEncoder(_Encoder):
    """
    εar-VAE, an Oobleck VAE trained for perceptual and phase fidelity, stereo:
    "ear_vae_v2_48k" (48 kHz, 50 fps; the upstream default) or "ear_vae_44k" (44.1 kHz, ~43 fps).

        mu, std, sample  [B, 64, T]  posterior mean, standard deviation and a random sample

    Decodes from mu or sample.
    """
    channels, decode_default = 2, "mu"
    repo, revision = "earlab/EAR_VAE", "5950873f1bca6d105fb873af8b68f43b05380093"  # pinned: its model code runs here

    def __init__(self, variant="ear_vae_v2_48k", device="cuda:0", batch_size=4):
        from huggingface_hub import snapshot_download

        super().__init__(device, batch_size)
        root = Path(snapshot_download(self.repo, revision=self.revision, allow_patterns=[
            "config.json", "config/*.json", "model/*.py", f"pretrained_weight/{variant}.pyt"]))
        spec = json.loads((root / "config.json").read_text())["variants"][variant]
        config = json.loads((root / spec["config"]).read_text())
        state = torch.load(root / spec["weights"], map_location="cpu")
        # The v2 config lists a decoder-side transformer that its checkpoint doesn't contain; build it only if present.
        if not any(k.startswith("transformers.") for k in state):
            config.pop("transformer", None)

        # The repo's model/ folder uses package-relative imports but has no __init__.py.
        package = types.ModuleType("ear_vae_hub")
        package.__path__ = [str(root / "model")]
        sys.modules.setdefault("ear_vae_hub", package)
        model = importlib.import_module("ear_vae_hub.ear_vae").EAR_VAE(config)
        model.load_state_dict(state)

        self.model = _frozen(model, self.device)
        self.sample_rate, self.pad_multiple = spec["sample_rate"], spec["downsampling_ratio"]

    def _encode(self, x):
        return self._gaussian(*self.model.encoder(x).chunk(2, dim=1), eps=0.0)  # εar-VAE: std = softplus(scale)

    def _decode(self, name, z):
        return self.model.decode(z) if name in ("mu", "sample") else super()._decode(name, z)
