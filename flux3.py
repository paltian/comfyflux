#!/usr/bin/env python3
"""flux3.py - FLUX.3 Image Klein LOKAL ueber ComfyUI: Text-to-Image und Editing.

Ein Modell fuer beides: Text-to-Image, Einzel-Referenz-Editing und
Multi-Referenz (bis 10 Bilder) laufen ueber denselben Graphen. Beim Editing
haengt ReferenceMultiLatent die Referenz-Latents an die Konditionierung; die
Ausgabegroesse kommt dann aus der Referenz, nicht aus --size.

Voraussetzungen auf dem Server (siehe README):
  ComfyUI >= 0.41 (FLUX3 in CLIPType), Custom Node ComfyUI-GGUF, und die Dateien
  flux-3-image-klein-UD-Q5_K_XL.gguf / qwen3vl_4b_int8_convrot.safetensors /
  flux2-vae.safetensors.

Beispiele:
  flux3 "ein Igel mit Partyhut" --size 1024x1024 -o igel.png
  flux3 "dieselbe Szene nachts bei Vollmond" --image igel.png -o nacht.png
  flux3 "kombiniere beide Motive" --image a.png --image b.png -o mix.png
  flux3 "..." --seed 42 --json          # reproduzierbar, maschinenlesbar

Ausgabezeile:  fertig in Xs -> PFAD  [seed=N | reproduzieren: --seed N]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # auch als Symlink startbar

from comfy_client import (AddRef, add_common_args, collect_refs, ensure_server, node_available,
                          out_path, parse_size, preflight, run_graph)

PROFILE = {
    "model": "flux-3-image-klein-UD-Q5_K_XL.gguf",
    "te": "qwen3vl_4b_int8_convrot.safetensors",
    "vae": "flux2-vae.safetensors",
    "steps": 15,
    "sampler": "er_sde",
    "scheduler": "simple",
    "cfg": 1.0,
    "ref_megapixels": 1.0,
}
GGUF_NODE = "UnetLoaderGGUF"


def build_t2i(p):
    return {
        "1": {"class_type": GGUF_NODE, "inputs": {"unet_name": p.model}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": p.w, "height": p.h, "batch_size": p.batch}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
            "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": p.prefix}},
    }


def build_edit(p, refs):
    g = {
        "1": {"class_type": GGUF_NODE, "inputs": {"unet_name": p.model}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
    }
    ref_inputs = {}
    for i, name in enumerate(refs):
        g[str(100 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_inputs["images.image%d" % i] = [str(100 + i), 0]
    # ReferenceMultiLatent liefert positive, negative UND das leere Latent -
    # ein eigener EmptyFlux2LatentImage-Knoten ist hier nicht noetig.
    g["20"] = {"class_type": "ReferenceMultiLatent", "inputs": {
        "positive": ["4", 0], "negative": ["6", 0], "vae": ["3", 0],
        "resolution": "resize" if p.ref_megapixels else "keep resolution",
        **({"resolution.megapixels": p.ref_megapixels} if p.ref_megapixels else {}),
        **ref_inputs}}
    g["21"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2],
        "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["3", 0]}}
    g["23"] = {"class_type": "SaveImage", "inputs": {"images": ["22", 0], "filename_prefix": p.prefix}}
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="Szenen-Beschreibung (Text-to-Image) ODER Aenderungswunsch (Editing)")
    add_common_args(ap)
    ap.add_argument("--negative", default="", help="Negativ-Prompt (Default: leer)")
    ap.add_argument("--size", default="1024x1024", metavar="WxH",
                    help="Ausgabegroesse beim Text-to-Image (Default 1024x1024). "
                         "Beim Editing ohne Wirkung: dort kommt die Groesse aus der Referenz.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=PROFILE["steps"])
    ap.add_argument("--cfg", type=float, default=PROFILE["cfg"])
    ap.add_argument("--sampler", default=PROFILE["sampler"],
                    help="Default er_sde. Das offizielle Editing-Template nutzt euler.")
    ap.add_argument("--scheduler", default=PROFILE["scheduler"])
    ap.add_argument("--ref-megapixels", type=float, default=PROFILE["ref_megapixels"],
                    help="Referenzbilder vor dem Encoding auf diese Groesse bringen "
                         "(1.0 ~ 1024x1024, Default). 0 = Originalmasse behalten.")
    ap.add_argument("--model", default=PROFILE["model"], help="DiT (GGUF) ueberschreiben")
    ap.add_argument("--te", default=PROFILE["te"], help="Text-Encoder ueberschreiben")
    ap.add_argument("--vae", default=PROFILE["vae"], help="VAE ueberschreiben")
    p = ap.parse_args()

    p.prefix = p.prefix or "flux3-klein"
    p.w, p.h = parse_size(p.size)
    p.seed = p.seed if p.seed is not None else random.randrange(2 ** 48)

    ensure_server(p.server, autostart=not p.no_autostart)
    if not node_available(p.server, GGUF_NODE):
        raise SystemExit("Knoten %s fehlt auf dem Server - Custom Node ComfyUI-GGUF installieren:\n"
                         "  git clone https://github.com/city96/ComfyUI-GGUF "
                         "~/dev/comfyui/custom_nodes/ComfyUI-GGUF" % GGUF_NODE)

    refs, _ = collect_refs(p.server, p.refs)
    if refs and p.size != "1024x1024":
        print("Hinweis: --size wird beim Editing ignoriert "
              "(Groesse = Referenz x --ref-megapixels).")

    preflight(p.server, [(GGUF_NODE, "unet_name", p.model),
                         ("CLIPLoader", "clip_name", p.te),
                         ("VAELoader", "vae_name", p.vae)])
    graph = build_edit(p, refs) if refs else build_t2i(p)
    mode = "edit" if refs else "t2i"
    dest = out_path(p.out, p.prefix, p.seed)

    t0 = time.time()
    saved, secs, pid = run_graph(p.server, graph, p.timeout, out=dest, prefix=p.prefix,
                                 metadata=p.metadata, quiet=p.json)
    if p.json:
        print(json.dumps({"out": os.path.abspath(saved[0]), "seed": p.seed, "width": p.w,
                          "height": p.h, "steps": p.steps, "sampler": p.sampler, "mode": mode,
                          "refs": len(refs), "seconds": round(secs, 1), "prompt_id": pid,
                          "prompt": p.prompt, "model": p.model}))
    else:
        print("fertig in %.1fs -> %s  [seed=%d | reproduzieren: --seed %d]"
              % (secs, ", ".join(saved), p.seed, p.seed))


if __name__ == "__main__":
    main()
