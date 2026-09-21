#%%
"""Encode a folder of wav files in batches, save one small cache per file, then decode and plot a few."""
from pathlib import Path

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure

from utils import encoder_models as em

#%% Settings
wav_dir = Path("/media/maindisk/ksoil/data/e-gmd-v1.0.0/e-gmd-v1.0.0/drummer1/session1")  # 44.1 kHz drum takes
n_files = 8      # how many files to encode
seconds = 5      # every file is cut (or zero-padded) to this length
batch_size = 4   # files loaded and encoded together

model_id="hubertsiuzdak/snac_44khz"
encoder = em.SNACEncoder(model_id)  # any encoder from utils/encoder_models.py

out_dir = Path("features") / type(encoder).__name__
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(wav_dir.glob("*.wav"))[:n_files]
n_samples = seconds * em.SR


def load(path):
    """The first `seconds` of a file as mono audio at 44.1 kHz, zero-padded if the file is shorter."""
    audio, _ = librosa.load(path, sr=em.SR, mono=True, duration=seconds)
    return F.pad(torch.from_numpy(audio), (0, n_samples - len(audio)))


#%% Encode in batches and save one cache per file
for start in range(0, len(files), batch_size):
    batch = files[start:start + batch_size]
    waves = torch.stack([load(path) for path in batch])  # [B, T]

    # seed=start: file k always draws its random VAE sample from seed k, whatever the batch size
    out = encoder.encode(waves, sr=em.SR, seed=start)

    # One cache per file with every representation, e.g. mu, std and sample for a VAE
    for i, path in enumerate(batch):
        torch.save({
            "features": {name: feature.tensor[i] for name, feature in out.items()},
            "fps": {name: feature.fps for name, feature in out.items()},
        }, out_dir / f"{path.stem}.pt")

    print(f"encoded {start + len(batch)}/{len(files)} files")

#%% Load two caches back, decode them, and save an overview plot (to a file only, not shown inline)
shown = files[:2]
rows = len(out) + 2  # input, every representation, reconstruction
fig = Figure(figsize=(12, 2.6 * rows), layout="constrained")
axes = fig.subplots(rows, len(shown), squeeze=False)
fig.suptitle(type(encoder).__name__)

for col, path in enumerate(shown):
    cache = torch.load(out_dir / f"{path.stem}.pt")
    features = {
        name: em.EncodedFeature(tensor[None], name, cache["fps"][name], em.SR, n_samples)
        for name, tensor in cache["features"].items()
    }
    recon = encoder.decode(features[encoder.decode_default])[0]  # [C, T]

    # The exact clip that was encoded, and its reconstruction, for listening side by side
    audio = load(path)
    sf.write(out_dir / f"{path.stem}_input.wav", audio.numpy(), em.SR)
    sf.write(out_dir / f"{path.stem}_recon.wav", recon.T.numpy(), em.SR)

    axes[0, col].specgram(audio.numpy(), Fs=em.SR)
    axes[0, col].set_title(f"{path.stem}: input")
    for row, feature in enumerate(features.values(), start=1):
        feature.plot(ax=axes[row, col], title=f"{path.stem}: {feature.name}")
    axes[-1, col].specgram(recon.mean(0).numpy(), Fs=em.SR)
    axes[-1, col].set_title(f"{path.stem}: reconstruction")

fig.savefig(out_dir / "overview.png")
print("saved", out_dir / "overview.png")
