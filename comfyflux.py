#!/usr/bin/env python3
"""comfyflux.py - FLUX.2-Klein Text-to-Image UND Image-Editing gegen eine ComfyUI-Instanz.

Beispiele:
  python3 comfyflux.py "ein Igel mit Partyhut" -o igel.png
  python3 comfyflux.py "ein Igel mit Partyhut" --size 1280x720 -o breit.png
  python3 comfyflux.py "change the party hat to blue" --image igel.png -o blau.png
"""
import argparse, json, os, random, sys, time, urllib.error, urllib.parse, urllib.request

DEFAULTS = {
    "model": "flux-2-klein-4b-fp8.safetensors",
    "te": "qwen_3_4b_fp4_flux2.safetensors",
    "vae": "flux2-vae.safetensors",
}

def build_t2i(p):
    return {
        "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": p.model, "weight_dtype": "default"}},
        "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": p.te, "type": "flux2", "device": "default"}},
        "3":  {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6":  {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7":  {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["6", 0], "cfg": p.cfg}},
        "8":  {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}},
        "9":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}},
        "10": {"class_type": "Flux2Scheduler", "inputs": {"steps": p.steps, "width": p.w, "height": p.h}},
        "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": p.w, "height": p.h, "batch_size": p.batch}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
                   "noise": ["8", 0], "guider": ["7", 0], "sampler": ["9", 0],
                   "sigmas": ["10", 0], "latent_image": ["11", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": p.prefix}},
    }

def build_edit(p, refs):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": p.model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.te, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.negative}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}},
        "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}},
        "23": {"class_type": "Flux2Scheduler", "inputs": {"steps": p.steps, "width": ["140", 0], "height": ["140", 1]}},
        "24": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["140", 0], "height": ["140", 1], "batch_size": p.batch}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {
                   "noise": ["21", 0], "guider": ["20", 0], "sampler": ["22", 0],
                   "sigmas": ["23", 0], "latent_image": ["24", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["3", 0]}},
        "27": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": p.prefix}},
        "140": {"class_type": "GetImageSize", "inputs": {"image": ["110", 0]}},
    }
    mp = round((p.w * p.h) / 1048576.0, 3)
    cond = ["4", 0]
    for i, name in enumerate(refs):
        lid, sid, eid, rid = 100 + i, 110 + i, 120 + i, 130 + i
        g[str(lid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[str(sid)] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": {"image": [str(lid), 0], "upscale_method": "nearest-exact",
                                  "megapixels": mp, "resolution_steps": 16}}
        g[str(eid)] = {"class_type": "VAEEncode", "inputs": {"pixels": [str(sid), 0], "vae": ["3", 0]}}
        g[str(rid)] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [str(eid), 0]}}
        cond = [str(rid), 0]
    g["20"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": cond, "negative": ["6", 0], "cfg": p.cfg}}
    return g

def post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("Server-Fehler HTTP %s: %s" % (e.code, e.read().decode()[:800]))

def upload_image(server, path):
    boundary = "----comfyflux" + str(random.randrange(10 ** 12))
    data = open(path, "rb").read()
    name = os.path.basename(path)
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
    ap.add_argument("--image", action="append", default=None,
                    help="Referenzbild (mehrfach moeglich) -> aktiviert Editing")
    ap.add_argument("--negative", default="")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--size", default="1024",
                    help="Zielgroesse: 1024 (quadratisch) oder 1280x720; beim Editing ist es die "
                         "MP-Obergrenze (1024 = 1 MP), das Seitenverhaeltis folgt dem Referenzbild")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--te", default=DEFAULTS["te"])
    ap.add_argument("--vae", default=DEFAULTS["vae"])
    ap.add_argument("--prefix", default="flux2-klein")
    ap.add_argument("--timeout", type=int, default=600)
    p = ap.parse_args()
    p.seed = p.seed if p.seed is not None else random.randrange(2 ** 48)
    if "x" in str(p.size).lower():
        p.w, p.h = (int(v) for v in str(p.size).lower().split("x"))
    else:
        p.w = p.h = int(p.size)
    p.w -= p.w % 16
    p.h -= p.h % 16

    refs = []
    for img in (p.image or []):
        refs.append(upload_image(p.server, img) if os.path.isfile(img) else img)
        print("Referenz: " + refs[-1])

    mode = "edit" if refs else "t2i"
    graph = build_edit(p, refs) if refs else build_t2i(p)
    t0 = time.time()
    res = post_json(p.server + "/prompt", {"prompt": graph})
    if "prompt_id" not in res:
        sys.exit("Server-Fehler: " + json.dumps(res)[:500])
    pid = res["prompt_id"]
    print("prompt_id=%s seed=%d %dx%d modus=%s ..." % (pid, p.seed, p.w, p.h, mode), flush=True)

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
        with urllib.request.urlopen(p.server + "/view?" + q, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        saved.append(dest)
    print("fertig in %.1fs -> %s" % (time.time() - t0, ", ".join(saved)))

if __name__ == "__main__":
    main()
