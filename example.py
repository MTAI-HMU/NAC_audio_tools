#%%
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from utils import encoder_models as em

#%% Two 6-second stereo clips: [B=2, C=2, T]
# Each channel holds a different recording, so a channel mix-up would be easy to spot.
seconds = 6
vibe, _ = librosa.load(librosa.ex("vibeace"), sr=em.SR, duration=seconds)
rag, _ = librosa.load(librosa.ex("pistachio"), sr=em.SR, duration=seconds)
vibe, rag = torch.tensor(vibe), torch.tensor(rag)

stereo = torch.stack([
    torch.stack([vibe, rag]),  # clip 0: left = vibeace, right = pistachio
    torch.stack([rag, vibe]),  # clip 1: the same, swapped
])


def show(encoder, waves):
    """Encode `waves`, plot every representation (a row each, a column per clip), and decode."""
    out = encoder.encode(waves, sr=em.SR)

    fig, axes = plt.subplots(len(out), len(waves), figsize=(12, 2.6 * len(out)), squeeze=False, layout="constrained")
    fig.suptitle(type(encoder).__name__)
    for row, (name, feature) in enumerate(out.items()):
        print(f"{name:12s} {tuple(feature.shape)}  {feature.fps:.2f} fps")
        for clip in range(len(waves)):
            feature.plot(index=clip, ax=axes[row, clip])

    return out, encoder.decode(out)  # decodes codes for codecs, the mean for VAEs


#%% DAC (a mono codec: stereo input is mixed down)
dac = em.DACEncoder()
out, recon = show(dac, stereo)

out["encoder_z"].plot(mode="norm")  # one curve instead of a heatmap
recon_from_latents = dac.decode(out["latents"])  # every representation decodes

#%% SNAC and EnCodec
snac = em.SNACEncoder()
out, recon = show(snac, stereo)

encodec = em.EnCodecEncoder()  # a 32 kHz model: resampled in and out automatically
out, recon = show(encodec, stereo)

del dac, snac, encodec  # free GPU memory before the next models
torch.cuda.empty_cache()

#%% Stable Audio Open VAE (stereo)
sao = em.SAOEncoder()
out, recon = show(sao, stereo)

out_mono = sao.encode(stereo, mono=True)  # the same clips collapsed to mono first

#%% SAME-L
same = em.SAMLEncoder()
out, recon = show(same, stereo)

del sao, same
torch.cuda.empty_cache()

#%% εar-VAE
ear = em.EARVAEEncoder()
out, recon = show(ear, stereo)

# Random draws are seeded: the same call gives the same sample, another seed a different one
sample = ear.encode(stereo)["sample"]
other_sample = ear.encode(stereo, seed=1)["sample"]

#%% Listen to clip 0 and its reconstruction
sf.write("input.wav", stereo[0].T.numpy(), em.SR)
sf.write("recon_ear_vae.wav", recon[0].T.numpy(), em.SR)
