#!/usr/bin/env python3
"""comfyflux.py - FLUX.2-Klein Text-to-Image UND Image-Editing gegen eine ComfyUI-Instanz.

Zielgroesse = Basisgroesse x --scale (Default 1).  Bei Referenzen ist die Basis
die Eingabe, bei t2i gilt --base-size.  Der --scale-mode bestimmt den Weg:

  post-dit (post)      DiT rechnet in Basisgroesse, das Ergebnis wird mit dem
                       KI-Upscaler vergroessert      -> schnell, treu
  pre-dit (pre)        Referenz wird zuerst geupscaled, dann refine in Zielgroesse
                                               -> treu + echte Details
  dit (regen)          DiT generiert direkt in Zielgroesse  -> kreativ
  upscale-no-dit (upscale)  kein DiT, reines KI-Upscaling   -> fuer vorhandene Bilder

Beispiele:
  python3 comfyflux.py "ein Igel mit Partyhut" -o igel.png
  python3 comfyflux.py "ein Igel mit Partyhut" --base-size 1280x720 -o breit.png
  python3 comfyflux.py "change the party hat to blue" --image igel.png -o blau.png
  python3 comfyflux.py "mehr Details" --image foto.png --scale 4 -o foto-4x.png
  python3 comfyflux.py "mehr Details" --image foto.png --scale 4 --scale-mode pre -o scharf.png
  python3 comfyflux.py "vorhandenes Foto aufwerten" --image foto.png --scale 2 --scale-mode upscale
  python3 comfyflux.py "combine image1 and image2" --image a.png --image https://example.com/b.jpg
  python3 comfyflux.py "mit Workflow-Info" --metadata -o archiv.png
"""
import argparse, json, os, random, struct, sys, time, urllib.error, urllib.parse, urllib.request

UPSCALE_MODEL = "4x-UltraSharp.pth"

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
    "pre-dit": "pre-dit", "pre": "pre-dit",
    "dit": "dit", "regen": "dit",
    "upscale-no-dit": "upscale-no-dit", "upscale": "upscale-no-dit",
}

def target_dims(base_w, base_h, scale):
    return (max(16, int(round(base_w * scale / 16.0)) * 16),
            max(16, int(round(base_h * scale / 16.0)) * 16))

def note_output(p, out_w, out_h, note):
    p.out_wh = (out_w, out_h)
    print("Ausgabe: %dx%d (%.2gx Basis, %.1f MP) [%s] %s" %
          (out_w, out_h, p.scale, out_w * out_h / 1048576.0, p.scale_mode, note))
    if out_w * out_h > 4_000_000:
        print("Hinweis: >4 MP - das wird langsam und braucht VRAM.")

def add_upscale_tail(g, p, src, target_w, target_h, save_node):
    """KI-Upscaler + exakte Zielgroesse VOR den SaveImage-Knoten."""
    g["40"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}}
    g["41"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["40", 0], "image": src}}
    g["42"] = {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["41", 0], "upscale_method": "lanczos",
                          "megapixels": round(target_w * target_h / 1048576.0, 3),
                          "resolution_steps": 16}}
    g[save_node]["inputs"]["images"] = ["42", 0]

def build_upscale_only(p, ref_name, base_w, base_h):
    tw, th = target_dims(base_w, base_h, p.scale)
    note_output(p, tw, th, "(nur KI-Upscale, kein DiT)")
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                         "megapixels": round(tw * th / 1048576.0, 3), "resolution_steps": 16}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": p.prefix}},
    }

def build_t2i(p):
    tw, th = target_dims(p.w, p.h, p.scale)
    post = p.scale_mode == "post-dit" and (tw, th) != (p.w, p.h)
    gen_w, gen_h = (p.w, p.h) if post else (tw, th)
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
    if post:
        add_upscale_tail(g, p, ["13", 0], tw, th, "14")
        note_output(p, tw, th, "DiT %dx%d + KI-Upscale" % (gen_w, gen_h))
    else:
        note_output(p, tw, th, "DiT in Zielgroesse" if (tw, th) != (p.w, p.h) else "")
    return g

def build_edit(p, refs, sizes=None):
    sizes = list(sizes or [])
    base_known = bool(sizes and sizes[0])
    bw, bh = sizes[0] if base_known else (p.w, p.h)
    if not base_known:
        print("Hinweis: Bildmasse nicht ermittelbar - Basis = --base-size.")
    tw, th = target_dims(bw, bh, p.scale)
    post = p.scale_mode == "post-dit" and (tw, th) != (bw, bh)
    gen_w, gen_h = (bw, bh) if post else (tw, th)

    ref_mps = []
    for s in sizes:
        if s:
            mp = (s[0] * s[1]) / 1048576.0
            if p.scale_mode == "pre-dit":
                mp = mp * p.scale ** 2
            ref_mps.append(round(min(mp, 4.0), 3))
        else:
            ref_mps.append(round(min(gen_w * gen_h / 1048576.0, 4.0), 3))
    while len(ref_mps) < len(refs):
        ref_mps.append(round(min(gen_w * gen_h / 1048576.0, 4.0), 3))

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
    if p.scale_mode == "pre-dit":
        g["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}}
    cond = ["4", 0]
    for i, name in enumerate(refs):
        lid, sid, eid, rid = 100 + i, 110 + i, 120 + i, 130 + i
        g[str(lid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        src = [str(lid), 0]
        if p.scale_mode == "pre-dit":
            uid = 150 + i
            g[str(uid)] = {"class_type": "ImageUpscaleWithModel",
                           "inputs": {"upscale_model": ["30", 0], "image": src}}
            src = [str(uid), 0]
        g[str(sid)] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": {"image": src,
                                  "upscale_method": "lanczos" if p.scale_mode == "pre-dit" else "nearest-exact",
                                  "megapixels": ref_mps[i], "resolution_steps": 16}}
        g[str(eid)] = {"class_type": "VAEEncode", "inputs": {"pixels": [str(sid), 0], "vae": ["3", 0]}}
        g[str(rid)] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [str(eid), 0]}}
        cond = [str(rid), 0]
    g["20"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": cond, "negative": ["6", 0], "cfg": p.cfg}}
    if post:
        add_upscale_tail(g, p, ["26", 0], tw, th, "27")
        note_output(p, tw, th, "DiT %dx%d + KI-Upscale" % (gen_w, gen_h))
    else:
        note_output(p, tw, th, "DiT in Zielgroesse" if (tw, th) != (bw, bh) else "Basisgroesse")
    return g

def post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("Server-Fehler HTTP %s: %s" % (e.code, e.read().decode()[:800]))

def pick_profile(server):
    """Groesstes Profil, das zur GPU des Servers passt und dessen Gewichte da sind.
    9b braucht ~16.6 GB VRAM: ab ~15 GB Gesamt-VRAM ist der Spill verkraftbar."""
    try:
        with urllib.request.urlopen(server + "/system_stats", timeout=30) as r:
            devs = json.load(r).get("devices", [])
        vram = devs[0].get("vram_total", 0) if devs else 0
    except Exception:
        vram = 0
    avail = {}
    try:
        for node, key in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name")):
            with urllib.request.urlopen(server + "/object_info/" + node, timeout=30) as r:
                avail[key] = widget_choices(json.load(r)[node]["input"]["required"][key])
    except Exception:
        avail = {"unet_name": set(), "clip_name": set()}
    for name in ("9b", "4b"):
        prof = PROFILES[name]
        if prof["model"] in avail["unet_name"] and prof["te"] in avail["clip_name"]:
            if name == "9b" and vram < 15_000_000_000:
                continue
            return name, "GPU %.1f GB VRAM, Gewichte vorhanden" % (vram / 1e9)
    return "4b", "Fallback (Gewichte/VRAM unvollstaendig)"

def strip_png_metadata(data):
    """Entfernt PNG-Text-Metadaten (tEXt/zTXt/iTXt/eXIf = Workflow+Prompts),
    Pixel bleiben unveraendert."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return data
    out = bytearray(data[:8])
    i, n = 8, len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i:i + 4], "big")
        ctype = data[i + 4:i + 8]
        end = i + 12 + length
        if end > n:
            out += data[i:]
            break
        if ctype not in (b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
            out += data[i:end]
        if ctype == b"IEND":
            break
        i = end
    return bytes(out)

def widget_choices(val):
    """Dateilisten aus object_info-Widgets: [liste, meta] oder ["COMBO", {options}]."""
    if not isinstance(val, list):
        return set()
    if val and isinstance(val[0], list):
        return set(val[0])
    if len(val) > 1 and val[0] == "COMBO" and isinstance(val[1], dict):
        return set(val[1].get("options", []))
    return set(v for v in val if isinstance(v, str) and v != "COMBO")

def preflight(server, model=None, te=None, upscale=None):
    """Prueft VOR dem Sampling, ob die Gewichte auf dem Server bekannt sind."""
    problems = []
    checks = []
    if model:
        checks.append(("UNETLoader", "unet_name", model))
    if te:
        checks.append(("CLIPLoader", "clip_name", te))
    if upscale:
        checks.append(("UpscaleModelLoader", "model_name", upscale))
    for node, key, fname in checks:
        with urllib.request.urlopen(server + "/object_info/" + node, timeout=30) as r:
            avail = widget_choices(json.load(r)[node]["input"]["required"][key])
        if fname not in avail:
            problems.append("  %s: %s\n     vorhanden: %s" %
                            (fname, "FEHLT auf dem Server", ", ".join(avail) or "(keine)"))
    if problems:
        raise SystemExit("Modell-Problem(e):\n" + "\n".join(problems))

class AddRef(argparse.Action):
    """Sammelt alle Referenzquellen in EINER Liste, damit die Eingabereihenfolge
    (image1, image2, ...) ueber verschiedene Flags hinweg erhalten bleibt."""
    KIND = {"--image": "auto", "--url": "url", "--server-image": "server"}

    def __call__(self, parser, namespace, values, option_string=None):
        refs = list(getattr(namespace, "refs", None) or [])
        refs.append((self.KIND[option_string], values))
        namespace.refs = refs

class Replaced(argparse.Action):
    """Alte Flags freundlich zu ihrer Abloesung fuehren."""
    HINT = {
        "--size": "--base-size (t2i-Basis) und --scale (Zielfaktor)",
        "--pre-upscale": "--scale N --scale-mode pre-dit",
        "--post-upscale": "--scale N --scale-mode post-dit",
    }

    def __call__(self, parser, namespace, values, option_string=None):
        raise SystemExit("%s wurde ersetzt durch: %s" % (option_string, self.HINT[option_string]))

def image_size(data):
    """(width, height) aus PNG/JPEG/GIF/BMP/WebP-Bytes, sonst None."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", data[16:24])
        if data[:3] == b"\xff\xd8\xff":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                m = data[i + 1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
            return None
        if data[:4] == b"GIF8":
            return struct.unpack("<HH", data[6:10])
        if data[:2] == b"BM":
            w, h = struct.unpack("<ii", data[18:26])
            return w, abs(h)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            if data[12:16] == b"VP8X":
                return (int.from_bytes(data[24:27], "little") + 1,
                        int.from_bytes(data[27:30], "little") + 1)
            if data[12:16] == b"VP8L":
                bits = struct.unpack("<I", data[21:25])[0]
                return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            if data[12:16] == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
    except Exception:
        pass
    return None

def download_input(server, name):
    """Holt ein Bild aus ComfyUIs input-Ordner (fuer Groessenbestimmung)."""
    q = urllib.parse.urlencode({"filename": os.path.basename(name),
                                "subfolder": os.path.dirname(name), "type": "input"})
    try:
        with urllib.request.urlopen(server + "/view?" + q, timeout=60) as r:
            return r.read()
    except Exception:
        return None

def fetch_ref(kind, value, server):
    """Liefert (Upload-Name, Bytes) fuer lokale/URL-Quellen, (None, Name) fuer
    Server-Dateien. Klare Fehler statt stiller Fehlinterpretation."""
    if kind == "server":
        with urllib.request.urlopen(server + "/object_info/LoadImage", timeout=30) as r:
            avail = widget_choices(json.load(r)["LoadImage"]["input"]["required"]["image"])
        if value not in avail:
            raise SystemExit("Server-Bild nicht vorhanden: %s\n  vorhanden: %s" %
                             (value, ", ".join(sorted(avail)[:12]) or "(keine)"))
        return None, value
    if value.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(value, timeout=120) as r:
                data = r.read()
        except Exception as e:
            raise SystemExit("Download fehlgeschlagen (%s): %s" % (value, e))
        base = os.path.basename(urllib.parse.urlparse(value).path)
        name = base if os.path.splitext(base)[1] else "url-image.png"
        return name, data
    if kind == "url":
        raise SystemExit("--url erwartet eine http(s)-Adresse, bekam: %s" % value)
    if not os.path.isfile(value):
        raise SystemExit("Datei nicht gefunden: %s"
                         " (Datei auf dem Server? -> --server-image)" % value)
    return os.path.basename(value), open(value, "rb").read()

def upload_image(server, name, data):
    boundary = "----comfyflux" + str(random.randrange(10 ** 12))
    disp_img = "Content-Disposition: form-data; name=\"image\"; filename=\"" + name + "\"\r\n"
    disp_over = "Content-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
    body = (("--" + boundary + "\r\n").encode() + disp_img.encode()
            + b"Content-Type: application/octet-stream\r\n\r\n" + data + b"\r\n"
            + ("--" + boundary + "\r\n").encode() + disp_over.encode()
            + ("--" + boundary + "--\r\n").encode())
    req = urllib.request.Request(server + "/upload/image", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("name", name)

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
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--profile", default="auto", choices=["auto"] + sorted(PROFILES),
                    help="Modellprofil; auto = groesstes passendes fuer die GPU des Servers "
                         "(Default), sonst 4b/4b-base/9b/9b-base")
    ap.add_argument("--steps", type=int, default=None,
                    help="Sampling-Schritte (Default: 8 distilled, 20 base)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="Guidance (Default: 1.0 distilled, 5.0 base)")
    ap.add_argument("--base-size", default="1024x1024", metavar="WxH",
                    help="Basisgroesse fuer t2i ohne Eingabebild (Default: 1024x1024)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Zielgroesse = Basisgroesse * Faktor (Default: 1)")
    ap.add_argument("--scale-mode", default="post-dit", metavar="MODUS",
                    help="post-dit (post, Default): DiT in Basisgroesse + KI-Upscale; "
                         "pre-dit (pre): Referenz erst upscaled, dann Refine; "
                         "dit (regen): DiT direkt in Zielgroesse; "
                         "upscale-no-dit (upscale): nur KI-Upscale, kein DiT")
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
    ap.add_argument("--timeout", type=int, default=600)
    # abgeloeste Flags
    ap.add_argument("--size", nargs="?", const="1", default=None, action=Replaced, help=argparse.SUPPRESS)
    ap.add_argument("--pre-upscale", nargs="?", const="1", default=None, action=Replaced, help=argparse.SUPPRESS)
    ap.add_argument("--post-upscale", nargs="?", const="1", default=None, action=Replaced, help=argparse.SUPPRESS)
    p = ap.parse_args()

    p.scale_mode = MODE_ALIASES.get(str(p.scale_mode).lower())
    if p.scale_mode is None:
        raise SystemExit("unbekannter --scale-mode. Erlaubt: post-dit (post), pre-dit (pre), "
                         "dit (regen), upscale-no-dit (upscale)")
    if p.scale <= 0:
        raise SystemExit("--scale muss > 0 sein")
    if p.scale <= 1 and p.scale_mode == "upscale-no-dit":
        raise SystemExit("upscale-no-dit braucht --scale > 1")

    try:
        with urllib.request.urlopen(p.server + "/system_stats", timeout=10) as r:
            r.read(1)
    except Exception as e:
        raise SystemExit("ComfyUI nicht erreichbar unter %s (%s)\n"
                         "  Server starten mit: bash ~/dev/comfyflux/start-comfy.sh" % (p.server, e))
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
    if "x" in str(p.base_size).lower():
        p.w, p.h = (int(v) for v in str(p.base_size).lower().split("x"))
    else:
        p.w = p.h = int(p.base_size)
    p.w -= p.w % 16
    p.h -= p.h % 16

    if p.scale_mode in ("pre-dit", "upscale-no-dit") and not (p.refs or []):
        raise SystemExit("%s braucht mindestens eine Referenz (--image/--url/--server-image)"
                         % p.scale_mode)
    refs = []
    sizes = []
    for kind, value in (p.refs or []):
        name, data = fetch_ref(kind, value, p.server)
        if name is not None:
            # laufende Nummer: gleiche Basenamen (z.B. zwei URLs ohne Endung)
            # wuerden sich beim Upload sonst gegenseitig ueberschreiben
            name = "ref%d-%s" % (len(refs) + 1, name)
            sizes.append(image_size(data))
            refs.append(upload_image(p.server, name, data))
        else:
            sizes.append(image_size(download_input(p.server, value) or b""))
            refs.append(value)
        print("Referenz: " + refs[-1])

    if p.scale_mode == "upscale-no-dit":
        if len(refs) > 1:
            print("Hinweis: upscale-no-dit nutzt nur image1 (%d Referenzen angegeben)." % len(refs))
        preflight(p.server, upscale=UPSCALE_MODEL)
        bw, bh = sizes[0] if (sizes and sizes[0]) else (p.w, p.h)
        graph = build_upscale_only(p, refs[0], bw, bh)
    else:
        preflight(p.server, p.model, p.te,
                  UPSCALE_MODEL if (p.scale_mode == "pre-dit" or
                                    (p.scale_mode == "post-dit" and p.scale != 1)) else None)
        graph = build_edit(p, refs, sizes) if refs else build_t2i(p)

    mode = "edit" if refs else "t2i"
    t0 = time.time()
    res = post_json(p.server + "/prompt", {"prompt": graph})
    if "prompt_id" not in res:
        sys.exit("Server-Fehler: " + json.dumps(res)[:500])
    pid = res["prompt_id"]
    ow, oh = getattr(p, "out_wh", (p.w, p.h))
    print("prompt_id=%s seed=%d %dx%d [%s] modus=%s ..." %
          (pid, p.seed, ow, oh, p.profile, mode), flush=True)

    hist = {}
    while time.time() - t0 < p.timeout:
        with urllib.request.urlopen(p.server + "/history/" + pid, timeout=30) as r:
            hist = json.load(r)
        if pid in hist:
            break
        time.sleep(2)
    else:
        sys.exit("Timeout nach %ds" % p.timeout)

    entry = hist.get(pid, {})
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        sys.exit("Sampling-Fehler: " + json.dumps(status)[:600])
    imgs = [im for out in entry.get("outputs", {}).values() for im in out.get("images", [])]
    if not imgs:
        sys.exit("keine Bilder in outputs: " + json.dumps(entry)[:600])

    saved = []
    for i, im in enumerate(imgs):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                    "type": im.get("type", "output")})
        dest = p.out or ("%s_%d.png" % (p.prefix, p.seed))
        if len(imgs) > 1:
            dest = dest.replace(".png", "_%d.png" % i)
        with urllib.request.urlopen(p.server + "/view?" + q, timeout=120) as r:
            data = r.read()
        if not p.metadata:
            data = strip_png_metadata(data)
        with open(dest, "wb") as f:
            f.write(data)
        saved.append(dest)
    print("fertig in %.1fs -> %s  [seed=%d | reproduzieren: --seed %d]" %
          (time.time() - t0, ", ".join(saved), p.seed, p.seed))

if __name__ == "__main__":
    main()
