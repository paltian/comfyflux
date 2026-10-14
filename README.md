# comfyflux

A tiny, dependency-free CLI for **FLUX.2 Klein** text-to-image generation and image
editing on top of a running [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
instance. No browser, no node graph — prompt in, PNG out.

```bash
# text-to-image
python3 comfyflux.py "a hedgehog wearing a tiny party hat, flash photography" -o igel.png

# custom resolution (width x height, rounded to multiples of 16)
python3 comfyflux.py "..." --size 1280x720 -o breit.png

# image editing: reference image + change request
python3 comfyflux.py "change the party hat to bright blue" --image igel.png -o blau.png

# multi-reference editing (up to 10 images; local paths or http(s) URLs)
python3 comfyflux.py "stylize image1 with the colors of image2" --image a.png --image https://example.com/b.jpg
```

## How it works

ComfyUI is an HTTP server; its web UI is just one client. `comfyflux.py` speaks to
the API directly:

1. `POST /prompt` — the pipeline as a JSON graph (loaders, sampler, save node)
2. `GET /history/<id>` — polling until the run is done
3. `GET /view?filename=...` — downloads the finished PNG

Reference images travel the other way via `POST /upload/image` (multipart).
Runs on any machine that can reach the server (`--server http://host:8188`),
Python 3 standard library only.

## Requirements

A running ComfyUI with the FLUX.2 Klein 4B files in place:

| File | Folder | Size | Source |
|---|---|---|---|
| `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/` | 3.9 GB | [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) |
| `qwen_3_4b_fp4_flux2.safetensors` | `models/text_encoders/` | 3.9 GB | [Comfy-Org/flux2-klein-4B](https://huggingface.co/Comfy-Org/flux2-klein-4B) |
| `flux2-vae.safetensors` | `models/vae/` | 336 MB | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) |

Total VRAM footprint: ~8.1 GB — fits a 12 GB card (RTX 3060) fully resident.

To use different checkpoints (e.g. the `base` variant for CFG > 1):

```bash
python3 comfyflux.py "..." --model flux-2-klein-base-4b-fp8.safetensors --cfg 5 --steps 20
```

## Options

| Option | Default | Purpose |
|---|---|---|
| `prompt` | — | text-to-image prompt, or change request when editing |
| `--size` | `1024` | `1024` (square) or `BxH` (e.g. `1280x720`); in edit mode it is the megapixel cap |
| `--steps` | `4` | sampling steps (4 is the distilled sweet spot) |
| `--image` | — | reference image (local path, server-side filename or http(s) URL), repeatable → editing mode |
| `--cfg` | `1.0` | guidance scale (meaningful for the `base` variant) |
| `--negative` | — | negative prompt |
| `--seed` | random | reproducibility |
| `--batch` | `1` | images per run |
| `--sampler` | `euler` | sampler selection |
| `--model/--te/--vae` | FLUX.2 Klein files | swap checkpoints |
| `--server` | `http://127.0.0.1:8188` | any reachable ComfyUI instance |
| `-o`, `--prefix`, `--timeout` | — | output file, filename prefix, run limit |

Resolutions are rounded down to multiples of 16 (VAE factor 8, patch size 2).
In editing mode the output keeps the reference image's aspect ratio.

## Shell alias

```bash
alias flux='python3 ~/dev/comfyflux/comfyflux.py --server http://your-server:8188'
```

## Security note

ComfyUI itself has no authentication. Keep it on `127.0.0.1` and use an SSH
tunnel (`ssh -L 8188:localhost:8188 host`) or put an authenticating reverse
proxy in front of it before exposing it to a network.

## License

MIT — see [LICENSE](LICENSE).
