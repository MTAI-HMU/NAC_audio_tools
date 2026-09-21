"""Minimal Stable Audio Open v1 VAE loader.

Why this module is needed
-------------------------
``stable_audio_tools`` provides the model components, but its normal pretrained
SAO entry point constructs the complete generative model. Feature extraction
needs only the much smaller pretrained audio VAE, so this module defines the
exact VAE architecture and loads its standalone ``vae_model.ckpt`` checkpoint.

The checkpoint is downloaded from Hugging Face on first use and then read from
the normal Hugging Face cache. Audio goes directly into the VAE; there is no
DAC/RVQ encoding or codec roundtrip.

The encoder predicts 64 posterior means and 64 posterior scale parameters. This
module only builds the model; ``SAOEncoder`` in ``encoder_models.py`` turns
them into ``mu`` (the deterministic posterior mean, the one to use for feature
caches), ``std``, and a ``sample`` drawn with a per-item seed so it is
reproducible too.

Input is 44.1-kHz stereo audio ``[B, 2, samples]``. The posterior mean has shape
``[B, 64, ceil(samples / 2048)]``: one frame per 2,048 samples, or
21.533203125 frames per second.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio as ta
from huggingface_hub import hf_hub_download

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

# --------------------------------------------------------------------------
# Constants from SAO-Instruct v1 / Stable Audio Open 1.0 VAE
# --------------------------------------------------------------------------
SAMPLE_RATE = 44_100
LATENT_CHANNELS = 64
DOWNSAMPLING_RATIO = 2048
SAO_RATE_HZ = SAMPLE_RATE / DOWNSAMPLING_RATIO  # 21.533203125

# Exact VAE architecture from SAO-Instruct's input_audio conditioner.
SAO_V1_VAE_CONFIG = {
    "model_type": "autoencoder",
    "sample_size": 2_097_152,
    "sample_rate": SAMPLE_RATE,
    "audio_channels": 2,
    "model": {
        "encoder": {
            "type": "oobleck",
            "requires_grad": False,
            "config": {
                "in_channels": 2,
                "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                # 64 posterior means + 64 scale parameters.
                "latent_dim": 128,
                "use_snake": True,
            },
        },
        "decoder": {
            "type": "oobleck",
            "requires_grad": False,
            "config": {
                "out_channels": 2,
                "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                "latent_dim": LATENT_CHANNELS,
                "use_snake": True,
                "final_tanh": False,
            },
        },
        "bottleneck": {"type": "vae"},
        "latent_dim": LATENT_CHANNELS,
        "downsampling_ratio": DOWNSAMPLING_RATIO,
        "io_channels": 2,
    },
}


def load_sao_v1_vae(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """Download (once, cached) and build the frozen SAO v1 VAE."""
    checkpoint_path = hf_hub_download(
        repo_id="stabilityai/stable-audio-open-1.0",
        filename="vae_model.ckpt",
    )
    vae = create_model_from_config(SAO_V1_VAE_CONFIG)
    copy_state_dict(vae, load_ckpt_state_dict(checkpoint_path))
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=dtype)

    assert vae.sample_rate == SAMPLE_RATE
    assert vae.latent_dim == LATENT_CHANNELS
    assert vae.downsampling_ratio == DOWNSAMPLING_RATIO
    assert vae.io_channels == 2
    return vae
