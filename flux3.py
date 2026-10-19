#!/usr/bin/env python3
"""flux3.py - FLUX.3 Image Klein LOKAL ueber ComfyUI: Text-to-Image und Editing.

Ein Modell fuer beides: Text-to-Image, Einzel-Referenz-Editing und
Multi-Referenz (bis 10 Bilder) laufen ueber denselben Graphen. Beim Editing
haengt ReferenceMultiLatent die Referenz-Latents an die Konditionierung; die
Ausgabegroesse kommt dann aus der Referenz (--ref-mp), nicht aus --size.

Groessen: --size (absolut) ODER --scale (Faktor auf --ref-mp), nie beide.
Basis = --ref-mp (bei Referenzen deren Format, bei t2i 1:1). --scale-mode
entscheidet, wer das Ziel erzeugt: dit (DiT direkt), post-dit (DiT + Upscale),
upscale-no-dit (nur Upscale).

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

from comfy_client import (UPSCALE_MODEL, AddRef, add_common_args, add_upscale_tail, collect_refs,
                          dims_for, ensure_server, node_available, out_path, parse_size, preflight,
                          resolve_sizes, run_graph)

MODE_ALIASES = {"post-dit": "post-dit", "post": "post-dit",
                "dit": "dit", "regen": "dit",
                "upscale-no-dit": "upscale-no-dit", "upscale": "upscale-no-dit"}
MODE_HELP = {
    "pre-dit": "nicht unterstuetzt; --ref-mp skaliert Referenzen (Lanczos)",
}

PROFILE = {
    "model": "flux-3-image-klein-UD-Q5_K_XL.gguf",
    "te": "qwen3vl_4b_int8_convrot.safetensors",
    "vae": "flux2-vae.safetensors",
    "steps": 15,
    "sampler": "er_sde",
    "scheduler": "simple",
    "cfg": 1.0,
    "ref_mp": 1.0,
}
GGUF_NODE = "UnetLoaderGGUF"


def build_t2i(p):
    # dit: der DiT erzeugt die Zielgroesse selbst; post-dit: er erzeugt die
    # Basis, der KI-Upscaler macht das Ziel.
    gen_w, gen_h = (int(p.w * p.scale), int(p.h * p.scale)) if p.scale_mode == "dit" else (p.w, p.h)
    g = {
        "1": {"class_type": GGUF_NODE, "inputs": {"unet_name": p.model}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": gen_w, "height": gen_h, "batch_size": p.batch}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
            "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": p.prefix}},
    }
    if p.scale > 1 and p.scale_mode == "post-dit":
        add_upscale_tail(g, ["8", 0], int(p.w * p.scale), int(p.h * p.scale), "9")
    return g


def build_edit(p, refs, sizes):
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
    # Das Latent kommt aus der Referenz, also bestimmt ihre Flaeche die
    # Ausgabegroesse: bei dit muss sie das Ziel sein (x scale^2), bei post-dit
    # die Basis.
    area, _ = ref_area_mp(p, sizes)
    if p.scale_mode == "dit":
        area *= p.scale ** 2
    mp = area / 1048576.0
    g["20"] = {"class_type": "ReferenceMultiLatent", "inputs": {
        "positive": ["4", 0], "negative": ["6", 0], "vae": ["3", 0],
        "resolution": "resize" if mp else "keep resolution",
        **({"resolution.megapixels": round(mp, 4)} if mp else {}),
        **ref_inputs}}
    g["21"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2],
        "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["3", 0]}}
    g["23"] = {"class_type": "SaveImage", "inputs": {"images": ["22", 0], "filename_prefix": p.prefix}}
    if p.scale > 1 and p.scale_mode == "post-dit":
        # Ausgabegroesse kommt aus der Referenz: Flaeche x scale^2
        area, _ = ref_area_mp(p, sizes)
        side = int(round((area * p.scale ** 2) ** 0.5))
        add_upscale_tail(g, ["22", 0], side, side, "23")
    return g


def build_upscale_only(p, ref_name, sizes):
    """Kein DiT: nur KI-Upscale der ersten Referenz."""
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
         "5": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": p.prefix}}}
    area, _ = ref_area_mp(p, sizes)
    side = int(round((area * p.scale ** 2) ** 0.5))
    add_upscale_tail(g, ["1", 0], side, side, "5")
    return g


def build_t2i(p, base, target):
    """dit: der DiT erzeugt das Ziel. post-dit: er erzeugt die Basis, der
    KI-Upscaler macht das Ziel."""
    gen = target if p.scale_mode == "dit" else base
    g = {
        "1": {"class_type": GGUF_NODE, "inputs": {"unet_name": p.model}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": gen[0], "height": gen[1], "batch_size": p.batch}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
            "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": p.prefix}},
    }
    if p.scale_mode == "post-dit" and target != base:
        add_upscale_tail(g, ["8", 0], target[0], target[1], "9")
    return g


def build_edit(p, refs, base, target):
    """Das Latent kommt aus der Referenz: bei dit muss sie das Ziel sein,
    bei post-dit die Basis."""
    ref_area = target[0] * target[1] if p.scale_mode == "dit" else base[0] * base[1]
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
    g["20"] = {"class_type": "ReferenceMultiLatent", "inputs": {
        "positive": ["4", 0], "negative": ["6", 0], "vae": ["3", 0],
        "resolution": "resize" if ref_area else "keep resolution",
        **({"resolution.megapixels": round(ref_area / 1048576.0, 4)} if ref_area else {}),
        **ref_inputs}}
    g["21"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2],
        "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["3", 0]}}
    g["23"] = {"class_type": "SaveImage", "inputs": {"images": ["22", 0], "filename_prefix": p.prefix}}
    if p.scale_mode == "post-dit" and target != base:
        add_upscale_tail(g, ["22", 0], target[0], target[1], "23")
    return g


def build_upscale_only(p, ref_name, target):
    """Kein DiT: nur KI-Upscale der ersten Referenz."""
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
         "5": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": p.prefix}}}
    add_upscale_tail(g, ["1", 0], target[0], target[1], "5")
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="Szenen-Beschreibung (Text-to-Image) ODER Aenderungswunsch (Editing)")
    add_common_args(ap)
    ap.add_argument("--negative", default="", help="Negativ-Prompt (Default: leer)")
    g_size = ap.add_mutually_exclusive_group()
    g_size.add_argument("--size", default=None, metavar="WxH",
                        help="Zielgroesse absolut (schliesst --scale aus). Ohne Angabe: "
                             "Basis x --scale bzw. die Basis selbst.")
    g_size.add_argument("--scale", type=float, default=None, metavar="N",
                        help="Ziel = Basis x N, Flaeche x N^2 (schliesst --size aus). "
                             "Basis ist --ref-mp.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=PROFILE["steps"])
    ap.add_argument("--cfg", type=float, default=PROFILE["cfg"])
    ap.add_argument("--sampler", default=PROFILE["sampler"],
                    help="Default er_sde. Das offizielle Editing-Template nutzt euler.")
    ap.add_argument("--scheduler", default=PROFILE["scheduler"])
    ap.add_argument("--scale-mode", default="post-dit", metavar="MODUS",
                    help="dit (regen): DiT erzeugt die Zielgroesse direkt; "
                         "post-dit (post, Default): DiT in Basisgroesse + KI-Upscale auf das Ziel; "
                         "upscale-no-dit (upscale): nur KI-Upscale, kein DiT (braucht eine Referenz)")
    ap.add_argument("--ref-mp", type=float, default=PROFILE["ref_mp"], metavar="MP",
                    help="Basisflaeche in Megapixeln (Default 1.0 ~ 1024x1024). Bei "
                         "Referenzen deren Groesse (Lanczos, Format bleibt), bei t2i die "
                         "Basis des DiT (1:1). --scale bezieht sich darauf.")
    ap.add_argument("--model", default=PROFILE["model"], help="DiT (GGUF) ueberschreiben")
    ap.add_argument("--te", default=PROFILE["te"], help="Text-Encoder ueberschreiben")
    ap.add_argument("--vae", default=PROFILE["vae"], help="VAE ueberschreiben")
    p = ap.parse_args()

    p.prefix = p.prefix or "flux3-klein"
    if p.scale is not None and p.scale <= 0:
        raise SystemExit("--scale muss > 0 sein")
    p.seed = p.seed if p.seed is not None else random.randrange(2 ** 48)

    ensure_server(p.server, autostart=not p.no_autostart)
    if not node_available(p.server, GGUF_NODE):
        raise SystemExit("Knoten %s fehlt auf dem Server - Custom Node ComfyUI-GGUF installieren:\n"
                         "  git clone https://github.com/city96/ComfyUI-GGUF "
                         "~/dev/comfyui/custom_nodes/ComfyUI-GGUF" % GGUF_NODE)

    mode_name = MODE_ALIASES.get(str(p.scale_mode).lower())
    if mode_name is None:
        hint = MODE_HELP.get(str(p.scale_mode).lower())
        raise SystemExit("--scale-mode '%s' gibt es bei flux3 nicht%s. Erlaubt: dit (regen), "
                         "post-dit (post), upscale-no-dit (upscale)"
                         % (p.scale_mode, " - " + hint if hint else ""))
    p.scale_mode = mode_name

    refs, sizes = collect_refs(p.server, p.refs)
    if refs and p.size:
        raise SystemExit("--size gibt es beim Editing nicht: die Groesse kommt aus der "
                         "Referenz. Nutze --ref-mp (Basis) und --scale (Faktor).")

    base = target = None
    if p.scale_mode == "upscale-no-dit":
        if not refs:
            raise SystemExit("upscale-no-dit braucht eine Referenz (--image/--url/--server-image)")
        if p.scale is None or p.scale <= 1:
            raise SystemExit("upscale-no-dit braucht --scale > 1")
        if len(refs) > 1:
            print("Hinweis: upscale-no-dit nutzt nur die erste Referenz (%d angegeben)." % len(refs))
        aspect = sizes[0][0] / float(sizes[0][1]) if (sizes and sizes[0]) else 1.0
        base = dims_for(p.ref_mp * 1048576.0, aspect)
        target = dims_for(p.ref_mp * p.scale ** 2 * 1048576.0, aspect)
    else:
        bw, bh, tw, th, base_mp, target_mp = resolve_sizes(p, refs, sizes)
        base, target = (bw, bh), (tw, th)
    needs_upscaler = p.scale_mode == "upscale-no-dit" or (p.scale_mode == "post-dit" and target != base)

    checks = [(GGUF_NODE, "unet_name", p.model), ("CLIPLoader", "clip_name", p.te),
              ("VAELoader", "vae_name", p.vae)]
    if needs_upscaler:
        checks.append(("UpscaleModelLoader", "model_name", UPSCALE_MODEL))
    preflight(p.server, checks)

    if p.scale_mode == "upscale-no-dit":
        graph, mode = build_upscale_only(p, refs[0], target), "upscale"
    else:
        graph = build_edit(p, refs, base, target) if refs else build_t2i(p, base, target)
        mode = "edit" if refs else "t2i"
    dest = out_path(p.out, p.prefix, p.seed)

    t0 = time.time()
    saved, secs, pid = run_graph(p.server, graph, p.timeout, out=dest, prefix=p.prefix,
                                 metadata=p.metadata, quiet=p.json)
    out_w, out_h = target
    if not p.json:
        print("Basis %dx%d -> Ziel %dx%d [%s]" % (base[0], base[1], out_w, out_h, p.scale_mode))
    if p.json:
        print(json.dumps({"out": os.path.abspath(saved[0]), "seed": p.seed, "width": out_w,
                          "height": out_h, "steps": p.steps, "sampler": p.sampler, "mode": mode,
                          "base": list(base), "scale": p.scale, "scale_mode": p.scale_mode,
                          "refs": len(refs), "seconds": round(secs, 1), "prompt_id": pid,
                          "prompt": p.prompt, "model": p.model}))
    else:
        print("fertig in %.1fs -> %s  [seed=%d | reproduzieren: --seed %d]"
              % (secs, ", ".join(saved), p.seed, p.seed))


if __name__ == "__main__":
    main()
