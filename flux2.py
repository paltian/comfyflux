#!/usr/bin/env python3
"""flux2.py - FLUX.2-Klein Text-to-Image UND Image-Editing gegen eine ComfyUI-Instanz.

Zielgroesse = Basisgroesse x --scale (Default 1).  Bei Referenzen ist die Basis
die Eingabe, bei t2i gilt --base-size.  Der --scale-mode bestimmt NUR den Weg
dorthin, nicht die Groesse:

  post-dit (post)      DiT rechnet in Basisgroesse, das Ergebnis wird mit dem
                       KI-Upscaler vergroessert      -> schnell, treu
  dit (regen)          DiT generiert direkt in Zielgroesse  -> kreativ
  upscale-no-dit (upscale)  kein DiT, reines KI-Upscaling   -> fuer vorhandene Bilder

Beispiele:
  flux2 "ein Igel mit Partyhut" -o igel.png
  flux2 "ein Igel mit Partyhut" --base-size 1280x720 -o breit.png
  flux2 "change the party hat to blue" --image igel.png -o blau.png
  flux2 "mehr Details" --image foto.png --scale 4 -o foto-4x.png
  flux2 "mehr Details" --image foto.png --scale 4 --ref-mp 2 -o scharf.png
  flux2 "vorhandenes Foto aufwerten" --image foto.png --scale 2 --scale-mode upscale
  flux2 "combine image1 and image2" --image a.png --image https://example.com/b.jpg
  flux2 "mit Workflow-Info" --metadata -o archiv.png
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # auch als Symlink startbar

from comfy_client import (DEFAULT_SERVER, UPSCALE_MODEL, AddRef, Replaced, add_upscale_tail,
                          collect_refs, dims_for, ensure_server, resolve_sizes,
                          get_json, out_path, preflight, run_graph, widget_choices)

PROFILES = {
    "4b":      {"model": "flux-2-klein-4b-fp8.safetensors",      "te": "qwen_3_4b_fp4_flux2.safetensors",
                "vae": "flux2-vae.safetensors", "steps": 8,  "cfg": 1.0},
    "4b-base": {"model": "flux-2-klein-base-4b-fp8.safetensors", "te": "qwen_3_4b_fp4_flux2.safetensors",
                "vae": "flux2-vae.safetensors", "steps": 20, "cfg": 5.0},
    "9b":      {"model": "flux-2-klein-9b-fp8.safetensors",      "te": "qwen_3_8b_fp4mixed.safetensors",
                "vae": "flux2-vae.safetensors", "steps": 8,  "cfg": 1.0},
    "9b-base": {"model": "flux-2-klein-base-9b-fp8.safetensors", "te": "qwen_3_8b_fp4mixed.safetensors",
                "vae": "flux2-vae.safetensors", "steps": 20, "cfg": 5.0},
}

# Lange und Kurzformen der Skalierungs-Modi
MODE_ALIASES = {
    "post-dit": "post-dit", "post": "post-dit",
    "dit": "dit", "regen": "dit",
    "upscale-no-dit": "upscale-no-dit", "upscale": "upscale-no-dit",
}

Replaced.HINT = {
    "--pre-upscale": "--ref-mp (Groesse der Referenzen)",
    "--post-upscale": "--scale N --scale-mode post-dit",
}


def target_dims(base_w, base_h, scale):
    return (max(16, int(round(base_w * scale / 16.0)) * 16),
            max(16, int(round(base_h * scale / 16.0)) * 16))

def note_output(p, base, target, note):
    p.out_wh = target
    print("Basis %dx%d -> Ziel %dx%d [%s] %s" %
          (base[0], base[1], target[0], target[1], p.scale_mode, note))
    if target[0] * target[1] > 4_000_000:
        print("Hinweis: >4 MP - das wird langsam und braucht VRAM.")

def build_upscale_only(p, ref_name, target):
    note_output(p, target, target, "(nur KI-Upscale, kein DiT)")
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
         "5": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": p.prefix}}}
    add_upscale_tail(g, ["1", 0], target[0], target[1], "5")
    return g

def build_t2i(p, base, target):
    # dit: der DiT erzeugt das Ziel. post-dit: er erzeugt die Basis, der
    # KI-Upscaler macht das Ziel.
    gen_w, gen_h = base if p.scale_mode == "post-dit" else target
    g = {
        "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": p.model, "weight_dtype": "default"}},
        "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": p.te, "type": "flux2", "device": "default"}},
        "3":  {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6":  {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7":  {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["6", 0], "cfg": p.cfg}},
        "8":  {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}},
        "9":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}},
        "10": {"class_type": "Flux2Scheduler", "inputs": {"steps": p.steps, "width": gen_w, "height": gen_h}},
        "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": gen_w, "height": gen_h, "batch_size": p.batch}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
                   "noise": ["8", 0], "guider": ["7", 0], "sampler": ["9", 0],
                   "sigmas": ["10", 0], "latent_image": ["11", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": p.prefix}},
    }
    if p.scale_mode == "post-dit" and target != base:
        add_upscale_tail(g, ["13", 0], target[0], target[1], "14")
        note_output(p, base, target, "DiT %dx%d + KI-Upscale" % (gen_w, gen_h))
    else:
        note_output(p, base, target, "DiT in Zielgroesse")
    return g

def build_edit(p, refs, sizes, base, target):
    sizes = list(sizes or [])
    # Der DiT erzeugt bei dit das Ziel, bei post-dit die Basis. Das Latent haengt
    # in flux2 NICHT von der Referenz ab - ihre Groesse steuert nur --ref-mp.
    gen_w, gen_h = base if p.scale_mode == "post-dit" else target
    ref_mps = []
    for s in sizes:
        if p.ref_mp:
            ref_mps.append(round(p.ref_mp, 3))
        elif s:
            ref_mps.append(round(min((s[0] * s[1]) / 1048576.0, 4.0), 3))
        else:
            ref_mps.append(round(min(gen_w * gen_h / 1048576.0, 4.0), 3))
    while len(ref_mps) < len(refs):
        ref_mps.append(round(p.ref_mp if p.ref_mp else min(gen_w * gen_h / 1048576.0, 4.0), 3))

    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": p.model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.te, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}},
        "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}},
        "23": {"class_type": "Flux2Scheduler", "inputs": {"steps": p.steps, "width": gen_w, "height": gen_h}},
        "24": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": gen_w, "height": gen_h, "batch_size": p.batch}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {
                   "noise": ["21", 0], "guider": ["20", 0], "sampler": ["22", 0],
                   "sigmas": ["23", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["3", 0]}},
        "27": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": p.prefix}},
    }
    cond = ["4", 0]
    for i, name in enumerate(refs):
        lid, sid, eid, rid = 100 + i, 110 + i, 120 + i, 130 + i
        g[str(lid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        src = [str(lid), 0]
        g[str(sid)] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": {"image": src,
                                  "upscale_method": "lanczos" if p.ref_mp else "nearest-exact",
                                  "megapixels": ref_mps[i], "resolution_steps": 16}}
        g[str(eid)] = {"class_type": "VAEEncode", "inputs": {"pixels": [str(sid), 0], "vae": ["3", 0]}}
        g[str(rid)] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [str(eid), 0]}}
        cond = [str(rid), 0]
    g["20"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": cond, "negative": ["6", 0], "cfg": p.cfg}}
    if p.scale_mode == "post-dit" and target != base:
        add_upscale_tail(g, ["26", 0], target[0], target[1], "27")
        note_output(p, base, target, "DiT %dx%d + KI-Upscale" % (gen_w, gen_h))
    else:
        note_output(p, base, target, "DiT in Zielgroesse")
    return g

def pick_profile(server):
    """Groesstes Profil, das zur GPU des Servers passt und dessen Gewichte da sind.
    9b braucht ~16.6 GB VRAM: ab ~15 GB Gesamt-VRAM ist der Spill verkraftbar."""
    try:
        devs = get_json(server, "/system_stats").get("devices", [])
        vram = devs[0].get("vram_total", 0) if devs else 0
    except Exception:
        vram = 0
    avail = {}
    try:
        for node, key in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name")):
            avail[key] = widget_choices(get_json(server, "/object_info/" + node)[node]["input"]["required"][key])
    except Exception:
        avail = {"unet_name": set(), "clip_name": set()}
    for name in ("9b", "4b"):
        prof = PROFILES[name]
        if prof["model"] in avail["unet_name"] and prof["te"] in avail["clip_name"]:
            if name == "9b" and vram < 15_000_000_000:
                continue
            return name, "GPU %.1f GB VRAM, Gewichte vorhanden" % (vram / 1e9)
    return "4b", "Fallback (Gewichte/VRAM unvollstaendig)"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="Text-to-Image-Prompt ODER Aenderungswunsch beim Editing")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--image", action=AddRef, default=None, dest="refs",
                    help="Referenzbild: lokale Datei ODER http(s)-URL (mehrfach moeglich)")
    ap.add_argument("--url", action=AddRef, default=None, dest="refs",
                    help="wie --image, aber explizit als URL (Alias)")
    ap.add_argument("--server-image", action=AddRef, default=None, dest="refs",
                    help="Referenzbild per Dateiname aus ComfyUIs input-Ordner (mehrfach moeglich)")
    ap.add_argument("--negative", default="")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help="ComfyUI-Adresse (Default: $COMFYFLUX_SERVER oder %s)" % DEFAULT_SERVER)
    ap.add_argument("--profile", default="auto", choices=["auto"] + sorted(PROFILES),
                    help="Modellprofil; auto = groesstes passendes fuer die GPU des Servers "
                         "(Default), sonst 4b/4b-base/9b/9b-base")
    ap.add_argument("--steps", type=int, default=None,
                    help="Sampling-Schritte (Default: 8 distilled, 20 base)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="Guidance (Default: 1.0 distilled, 5.0 base)")
    g_size = ap.add_mutually_exclusive_group()
    g_size.add_argument("--size", "--base-size", dest="size", default=None, metavar="WxH",
                        help="Zielgroesse absolut (schliesst --scale aus). Ohne Angabe: "
                             "Basis x --scale bzw. die Basis selbst.")
    g_size.add_argument("--scale", type=float, default=None, metavar="N",
                        help="Ziel = Basis x N, Flaeche x N^2 (schliesst --size aus). "
                             "Basis ist --ref-mp.")
    ap.add_argument("--scale-mode", default="post-dit", metavar="MODUS",
                    help="post-dit (post, Default): DiT in Basisgroesse + KI-Upscale; "
                         "dit (regen): DiT direkt in Zielgroesse; "
                         "upscale-no-dit (upscale): nur KI-Upscale, kein DiT")
    ap.add_argument("--ref-mp", type=float, default=1.0, metavar="MP",
                    help="Basisflaeche in Megapixeln (Default 1.0 ~ 1024x1024). Bei "
                         "Referenzen deren Groesse (Lanczos, Format bleibt), bei t2i die "
                         "Basis des DiT (1:1). --scale bezieht sich darauf. "
                         "0 = Originalmasse der Referenz.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--metadata", action="store_true",
                    help="Workflow/Prompts in die PNG-Metadaten schreiben "
                         "(Standard: Metadaten werden entfernt)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--model", default=None, help="ueberschreibt das DiT des Profils")
    ap.add_argument("--te", default=None, help="ueberschreibt den Text-Encoder des Profils")
    ap.add_argument("--vae", default=None, help="ueberschreibt den VAE des Profils")
    ap.add_argument("--prefix", default="flux2-klein")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Wartezeit in Sekunden auf das Ergebnis (Default: 900). "
                     "Bei Timeout wird der eigene Auftrag abgebrochen.")
    ap.add_argument("--no-autostart", action="store_true",
                    help="Server nicht selbst starten, auch wenn er lokal eingerichtet ist")
    ap.add_argument("--json", action="store_true", help="Ergebnis als JSON statt Textzeile")
    # abgeloeste Flags
    ap.add_argument("--pre-upscale", nargs="?", const="1", default=None, action=Replaced, help=argparse.SUPPRESS)
    ap.add_argument("--post-upscale", nargs="?", const="1", default=None, action=Replaced, help=argparse.SUPPRESS)
    p = ap.parse_args()

    p.scale_mode = MODE_ALIASES.get(str(p.scale_mode).lower())
    if p.scale_mode is None:
        raise SystemExit("unbekannter --scale-mode. Erlaubt: post-dit (post), dit (regen), "
                         "upscale-no-dit (upscale)")
    if p.scale is not None and p.scale <= 0:
        raise SystemExit("--scale muss > 0 sein")
    if p.scale_mode == "upscale-no-dit" and (p.scale is None or p.scale <= 1):
        raise SystemExit("upscale-no-dit braucht --scale > 1")

    ensure_server(p.server, autostart=not p.no_autostart)
    if p.profile == "auto":
        p.profile, why = pick_profile(p.server)
        print("auto-Profil: %s (%s)" % (p.profile, why))
    prof = PROFILES[p.profile]
    p.model = p.model or prof["model"]
    p.te = p.te or prof["te"]
    p.vae = p.vae or prof["vae"]
    p.steps = p.steps if p.steps is not None else prof["steps"]
    p.cfg = p.cfg if p.cfg is not None else prof["cfg"]
    p.seed = p.seed if p.seed is not None else random.randrange(2 ** 48)
    if p.scale is not None and p.scale <= 0:
        raise SystemExit("--scale muss > 0 sein")

    if p.scale_mode == "upscale-no-dit" and not (p.refs or []):
        raise SystemExit("%s braucht mindestens eine Referenz (--image/--url/--server-image)"
                         % p.scale_mode)
    refs, sizes = collect_refs(p.server, p.refs)
    if refs and p.size:
        raise SystemExit("--size gibt es beim Editing nicht: die Groesse kommt aus der "
                         "Referenz. Nutze --ref-mp (Basis) und --scale (Faktor).")
    bw, bh, tw, th, _, _ = resolve_sizes(p, refs, sizes)
    base, target = (bw, bh), (tw, th)

    if p.scale_mode == "upscale-no-dit":
        if len(refs) > 1:
            print("Hinweis: upscale-no-dit nutzt nur image1 (%d Referenzen angegeben)." % len(refs))
        preflight(p.server, [("UpscaleModelLoader", "model_name", UPSCALE_MODEL)])
        graph = build_upscale_only(p, refs[0], target)
    else:
        checks = [("UNETLoader", "unet_name", p.model), ("CLIPLoader", "clip_name", p.te)]
        if p.scale_mode == "post-dit" and target != base:
            checks.append(("UpscaleModelLoader", "model_name", UPSCALE_MODEL))
        preflight(p.server, checks)
        graph = build_edit(p, refs, sizes, base, target) if refs else build_t2i(p, base, target)

    mode = "edit" if refs else "t2i"
    dest = out_path(p.out, p.prefix, p.seed)
    ow, oh = getattr(p, "out_wh", target)
    if not p.json:
        print("seed=%d %dx%d [%s] modus=%s ..." % (p.seed, ow, oh, p.profile, mode), flush=True)

    saved, secs, pid = run_graph(p.server, graph, p.timeout, out=dest, prefix=p.prefix,
                                 metadata=p.metadata, quiet=p.json)
    if p.json:
        print(json.dumps({"out": os.path.abspath(saved[0]), "seed": p.seed, "width": ow, "height": oh,
                          "steps": p.steps, "profile": p.profile, "mode": mode,
                          "base": list(base), "scale": p.scale, "scale_mode": p.scale_mode,
                          "seconds": round(secs, 1),
                          "prompt_id": pid, "prompt": p.prompt}))
    else:
        print("fertig in %.1fs -> %s  [seed=%d | reproduzieren: --seed %d]"
              % (secs, ", ".join(saved), p.seed, p.seed))


if __name__ == "__main__":
    main()
