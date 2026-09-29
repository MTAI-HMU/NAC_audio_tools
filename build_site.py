"""
Build a static page that shows, for every encoder in this repo, its representations as plots and its
reconstruction as audio you can play. Output is plain HTML + assets, ready for GitHub Pages.

    python build_site.py --out docs [--device cuda:0] [--seconds 6] [--only DAC "SNAC 44k"]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
if not (REPO / "utils").exists():
    REPO = Path("/media/maindisk/ksoil/repos/NAC_audio_tools")
sys.path.insert(0, str(REPO))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from utils import encoder_models as em

# Model name -> (constructor source shown on the page, factory, kind, audio, per-rep note)
def CASES(dev):
    return [
        ("DAC", "em.DACEncoder()", lambda: em.DACEncoder(device=dev), "RVQ codec", "44.1 kHz mono", "#dac"),
        ("EnCodec 32 kHz", "em.EnCodecEncoder()", lambda: em.EnCodecEncoder(device=dev), "RVQ codec",
         "32 kHz mono", "#encodec"),
        ("EnCodec 24 kHz", 'em.EnCodecEncoder("facebook/encodec_24khz")',
         lambda: em.EnCodecEncoder("facebook/encodec_24khz", device=dev), "RVQ codec", "24 kHz mono", "#encodec"),
        ("SNAC 44 kHz", "em.SNACEncoder()", lambda: em.SNACEncoder(device=dev), "multi-scale RVQ codec",
         "44.1 kHz mono", "#snac"),
        ("SNAC 32 kHz", 'em.SNACEncoder("hubertsiuzdak/snac_32khz")',
         lambda: em.SNACEncoder("hubertsiuzdak/snac_32khz", device=dev), "multi-scale RVQ codec", "32 kHz mono",
         "#snac"),
        ("Stable Audio Open VAE", "em.SAOEncoder()", lambda: em.SAOEncoder(device=dev), "VAE",
         "44.1 kHz stereo", "#sao"),
        ("SAME-L", "em.SAMLEncoder()", lambda: em.SAMLEncoder(device=dev), "autoencoder", "44.1 kHz stereo",
         "#same-l"),
        ("Music2Latent", "em.Music2LatentEncoder()", lambda: em.Music2LatentEncoder(device=dev),
         "consistency autoencoder", "44.1 kHz mono", "#music2latent"),
        ("DACVAE", "em.DACVAEEncoder()", lambda: em.DACVAEEncoder(device=dev), "VAE", "48 kHz mono", "#dacvae"),
        ("ACE-Step 1.5 VAE", "em.ACEStep15Encoder()", lambda: em.ACEStep15Encoder(device=dev), "VAE",
         "48 kHz stereo", "#ace-step-15-vae"),
        ("ACE-Step v1 music DCAE", "em.ACEStepDCAEEncoder()", lambda: em.ACEStepDCAEEncoder(device=dev),
         "mel autoencoder + vocoder", "44.1 kHz stereo", "#ace-step-v1-music-dcae"),
        ("εar-VAE v2 48 kHz", "em.EARVAEEncoder()", lambda: em.EARVAEEncoder(device=dev), "VAE",
         "48 kHz stereo", "#ear-vae"),
        ("εar-VAE 44.1 kHz", 'em.EARVAEEncoder("ear_vae_44k")',
         lambda: em.EARVAEEncoder("ear_vae_44k", device=dev), "VAE", "44.1 kHz stereo", "#ear-vae"),
    ]

AUDIO_FORMATS = [("MP3", "mp3", {}), ("OGG", "ogg", {"subtype": "VORBIS"}), ("WAV", "wav", {"subtype": "PCM_16"})]

# decode() accepts these, but no model decodes them directly: the codecs quantize them and SAME-L
# normalizes pre_softnorm, so they give the same audio as the default representation.
CONVERTED = {"encoder_z", "latents", "pre_softnorm"}


def save_audio(stem: Path, wav: torch.Tensor, sr: int) -> str:
    """Write [C, T] float audio as the smallest format libsndfile offers here. Returns the file name."""
    available = sf.available_formats()
    for fmt, ext, kw in AUDIO_FORMATS:
        if fmt not in available:
            continue
        path = stem.with_suffix("." + ext)
        try:
            sf.write(path, wav.T.numpy(), sr, format=fmt, **kw)
            return path.name
        except Exception:
            continue
    raise RuntimeError("no usable audio format in libsndfile")


def draw_spectrogram(fig, ax, y, sr: int, title: str):
    """Log-frequency STFT of one mono signal, on the same dB scale for every model so cards compare directly."""
    import librosa.display

    db = librosa.amplitude_to_db(abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=1.0, top_db=90)
    im = librosa.display.specshow(db, sr=sr, hop_length=256, x_axis="time", y_axis="log", ax=ax, cmap="magma",
                                  vmin=-50, vmax=40)
    ax.set(title=title, xlabel="Time (s)")
    ax.title.set_fontsize(11)
    fig.colorbar(im, ax=ax, format="%+.0f dB")


def save_plot(out: Path, name: str, feats: dict, recon, sr: int) -> str:
    """One figure per model: its decoded audio as a spectrogram, then a row per representation."""
    rows = len(feats) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.3 * rows), squeeze=False, layout="constrained")
    draw_spectrogram(fig, axes[0, 0], recon.mean(0).numpy(), sr, "decoded audio   (mono mix)")
    for ax, (rep, f) in zip(axes[1:, 0], feats.items()):
        f.plot(index=0, ax=ax, title=f"{rep}   {tuple(f.shape)}   {f.fps:.2f} fps")
        ax.title.set_fontsize(11)  # set_title(loc="left") would add a second title next to this one
    path = out / f"{name}.png"
    fig.savefig(path, dpi=96)
    plt.close(fig)
    return path.name


def build(args):
    out = Path(args.out).resolve()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Source audio: one real stereo recording, shipped next to the page as a .wav so the same input can be
    # reused elsewhere. Everything on the page is this clip, so models are compared on identical audio.
    info = sf.info(args.source)
    seg, sr = sf.read(args.source, start=int(args.offset * info.samplerate),
                      frames=int(args.seconds * info.samplerate), dtype="float32", always_2d=True)
    sf.write(assets / "source.wav", seg, sr, subtype="PCM_16")
    source_file, n = "source.wav", len(seg)
    waves = torch.tensor(seg.T)[None]  # [1, C, T]
    print(f"source: {Path(args.source).name} @{args.offset}s  {n / sr:.1f}s  {sr} Hz  {seg.shape[1]}ch", flush=True)

    models, cases = [], CASES(args.device)
    for name, src, make, kind, audio, anchor in cases:
        if args.only and name not in args.only:
            continue
        print(f"=== {name}", flush=True)
        t0 = time.time()
        try:
            enc = make()
            feats = enc.encode(waves, sr=sr)
            recon = enc.decode(feats)
            # Ask decode() itself which representations it accepts, rather than trusting a hardcoded list.
            can = {}
            for rep, f in feats.items():
                try:
                    enc.decode(f)
                    can[rep] = "yes"
                except ValueError:
                    can[rep] = "no"
            slug = name.lower().replace(" ", "-").replace(".", "").replace("εar", "ear")
            entry = dict(
                name=name, source=src, kind=kind, audio=audio, anchor=anchor, slug=slug,
                default=enc.decode_default, channels=enc.channels, sample_rate=enc.sample_rate,
                plot=save_plot(assets, slug, feats, recon[0], sr),
                recon=save_audio(assets / f"{slug}-recon", recon[0], sr),
                seconds=round(time.time() - t0, 1),
                reps=[dict(name=rep, shape=list(f.shape), fps=round(f.fps, 2), codes=f.is_codes,
                           decodable=can[rep], default=rep == enc.decode_default)
                      for rep, f in feats.items()],
            )
            models.append(entry)
            print(f"    {entry['plot']}, {entry['recon']} in {entry['seconds']}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"    SKIPPED {name}", flush=True)
        finally:
            enc = feats = recon = None
            gc.collect()
            torch.cuda.empty_cache()

    manifest = dict(source=source_file, source_name=Path(args.source).name,
                    sample_rate=sr, seconds=round(n / sr, 2), models=models, built=time.strftime("%Y-%m-%d"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "index.html").write_text(render(manifest))
    print(f"\nwrote {out / 'index.html'} ({len(models)} models)")


# --------------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------------
CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfaf8; --card: #ffffff; --ink: #1a1a19; --muted: #6a6862; --line: #e4e1da;
  --accent: #7a4b2a; --chip: #f1eee8; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #16161a; --card: #1e1e23; --ink: #ecebe8; --muted: #a3a09a; --line: #2e2e35;
          --accent: #e0a06a; --chip: #26262d; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 0 20px 80px; }
header { padding: 56px 0 28px; border-bottom: 1px solid var(--line); margin-bottom: 32px; }
h1 { font-size: 30px; margin: 0 0 8px; letter-spacing: -0.02em; }
h1 .thin { color: var(--muted); font-weight: 400; }
.lede { color: var(--muted); max-width: 70ch; margin: 0 0 18px; }
a { color: var(--accent); }
code, .mono { font-family: var(--mono); font-size: 0.92em; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.chip { background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
        padding: 4px 11px; font-size: 13px; color: var(--muted); text-decoration: none; }
.chip:hover { color: var(--ink); border-color: var(--accent); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        margin: 0 0 14px; scroll-margin-top: 16px; }
.card > summary { cursor: pointer; list-style: none; padding: 16px 22px; display: flex; gap: 6px 14px;
                  align-items: baseline; flex-wrap: wrap; border-radius: 14px; }
.card > summary::-webkit-details-marker { display: none; }
.card > summary::after { content: "+"; margin-left: auto; color: var(--muted); font-size: 18px;
                         font-family: var(--mono); }
.card[open] > summary::after { content: "−"; }
.card > summary:hover { background: var(--chip); }
.card > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.card .body { padding: 4px 22px 22px; }
.sname { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }
.smeta { color: var(--muted); font-size: 13.5px; }
.bar { display: flex; gap: 10px; align-items: center; margin: 0 0 16px; }
.btn { background: var(--chip); border: 1px solid var(--line); color: var(--ink); border-radius: 8px;
       padding: 6px 12px; font-size: 13px; cursor: pointer; font-family: inherit; }
.btn:hover { border-color: var(--accent); }
.meta { color: var(--muted); font-size: 13.5px; margin-bottom: 16px; }
.meta .sep { opacity: 0.45; padding: 0 7px; }
pre { background: var(--chip); border: 1px solid var(--line); border-radius: 9px; padding: 11px 13px;
      overflow-x: auto; margin: 0 0 16px; }
pre code { font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 4px 0 18px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; font-size: 12.5px; text-transform: uppercase;
     letter-spacing: 0.04em; }
.tick { color: var(--accent); }
td.tick { white-space: nowrap; }
.no { color: var(--muted); }
.qual { color: var(--muted); font-size: 12px; }
img { width: 100%; height: auto; border-radius: 9px; border: 1px solid var(--line); display: block; }
audio { width: 100%; margin: 6px 0 2px; }
.players { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.player h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
             margin: 0 0 4px; font-weight: 500; }
figure { margin: 0 0 4px; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 8px; }
footer { color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 22px; }
@media (max-width: 560px) { header { padding-top: 34px; } h1 { font-size: 24px; } .card { padding: 18px; } }
"""


def render(m: dict) -> str:
    # Assets keep their file names across rebuilds, so a version query keeps browsers from serving a stale copy.
    m = dict(m, v=int(time.time()))
    chips = "\n".join(f'<a class="chip" href="#{x["slug"]}">{x["name"]}</a>' for x in m["models"])
    cards = "\n".join(card(x, m) for x in m["models"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAC audio tools — Easy demo listening page</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>NAC audio tools <span class="thin">— Easy demo listening page</span></h1>
  <div class="players">
    <div class="player"><h3>Source — {m["source_name"]}</h3>
      <audio controls preload="none" src="assets/{m["source"]}?v={m["v"]}"></audio></div>
  </div>
  <div class="chips">{chips}</div>
</header>
<div class="bar">
  <button class="btn" id="expand">Expand all</button>
  <button class="btn" id="collapse">Collapse all</button>
</div>
{cards}
<footer>
  Built {m["built"]} from this repository's <code>utils/encoder_models.py</code>. Plots are
  <code>EncodedFeature.plot()</code> heatmaps of batch item 0; every clip is decoded from the
  representation marked <em>default</em>. Source audio is <code>assets/{m["source"]}</code>, a
  {m["seconds"]}-second excerpt of <code>{m["source_name"]}</code> from the
  <a href="https://magenta.tensorflow.org/datasets/groove">Groove MIDI Dataset</a> (CC BY 4.0);
  reconstructions are MP3 to keep the page small.
</footer>
</div>
<script>
const cards = document.querySelectorAll("details.card");
document.getElementById("expand").onclick = () => cards.forEach(c => c.open = true);
document.getElementById("collapse").onclick = () => cards.forEach(c => c.open = false);

// A link to a collapsed model should open it and scroll to it.
function openHash() {{
  const el = document.querySelector(location.hash || "#none");
  if (el && el.tagName === "DETAILS") {{ el.open = true; el.scrollIntoView(); }}
}}
addEventListener("hashchange", openHash);
openHash();
</script>
</body>
</html>
"""


def decodable_cell(r: dict) -> str:
    if r["decodable"] != "yes":
        return '<td class="no">—</td>'
    if r["name"] in CONVERTED:
        return '<td class="tick">✓ <span class="qual">converted first</span></td>'
    return '<td class="tick">✓</td>'


def card(x: dict, m: dict) -> str:
    rows = "\n".join(
        f'      <tr><td><code>{r["name"]}</code>{" <em>(default)</em>" if r["default"] else ""}</td>'
        f'<td class="mono">[{", ".join(str(s) for s in r["shape"])}]</td>'
        f'<td>{r["fps"]:.2f}</td>'
        f"{decodable_cell(r)}</tr>"
        for r in x["reps"])
    ch = "mono" if x["channels"] == 1 else "stereo"
    return f"""<details class="card" id="{x["slug"]}">
  <summary><span class="sname">{x["name"]}</span>
    <span class="smeta">{x["kind"]}<span class="sep">·</span>{x["audio"]}<span class="sep">·</span>
    {len(x["reps"])} representations</span></summary>
  <div class="body">
  <p class="meta">decodes from <code>{x["default"]}</code><span class="sep">·</span>
     <a href="https://github.com/MTAI-HMU/NAC_audio_tools/blob/main/README.md{x["anchor"]}">details</a></p>
  <pre><code>{x["source"]}</code></pre>
  <table>
    <thead><tr><th>Representation</th><th>Shape</th><th>fps</th><th>Decodable</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <figure>
    <img src="assets/{x["plot"]}?v={m["v"]}" alt="{x["name"]} representations" loading="lazy">
    <figcaption>Top: the decoded audio as a log-frequency spectrogram, mono mix, on the same dB scale
    for every model. Below it, each representation of batch item 0, dimensions on the y axis, time on
    the x axis.</figcaption>
  </figure>
  <div class="players">
    <div class="player"><h3>Decoded ({ch})</h3>
      <audio controls preload="none" src="assets/{x["recon"]}?v={m["v"]}"></audio></div>
  </div>
  </div>
</details>"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="site")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--source", default="/media/maindisk/ksoil/data/groove-v1.0.0/groove-v1.0.0/groove/"
                                       "drummer4/session1/1_rock_87_beat_4-4.wav")
    p.add_argument("--offset", type=float, default=41.2)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--render-only", action="store_true", help="rebuild index.html from manifest.json")
    args = p.parse_args()
    if args.render_only:
        out = Path(args.out).resolve()
        manifest = json.loads((out / "manifest.json").read_text())
        (out / "index.html").write_text(render(manifest))
        print(f"rendered {out / 'index.html'} from manifest ({len(manifest['models'])} models)")
    else:
        build(args)
