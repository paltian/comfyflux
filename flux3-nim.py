#!/usr/bin/env python3
"""flux3-nim.py - FLUX.3 Image Klein ueber NVIDIA NIM (Cloud).

Gleicher CLI-Aufbau wie flux2.py/flux3.py (die lokal gegen ComfyUI laufen),
aber Cloud-Endpoint: kein VRAM, kein Server. Reines Text-to-Image - Bild-Editing
gibt es nur lokal, NIM verlangt dafuer Asset-Referenzen (example_id) statt Bildern.

Bildmasse: NIM erlaubt nur 512..1568 in 16er-Schritten. Gewuenschtes Format
wird auf das naechste erlaubte Raster gesetzt, Seitenverhaeltnis gehalten;
reicht das nicht (Extremformate), gibt es eine Warnung auf stderr.

API-Key, in dieser Reihenfolge:
  1. $NVIDIA_API_KEY
  2. ~/.config/flux/nvidia-api-key   (Datei mit dem Key, chmod 600)
  3. ~/.pi/agent/auth.json           (Eintrag "nvidia", von pi verwaltet)
Der Key gehoert NICHT ins Repo - alle drei Quellen liegen ausserhalb.

Beispiele:
  flux3-nim "ein Igel mit Partyhut" -o igel.jpg
  flux3-nim "ein Igel mit Partyhut" --size 1280x720 -o breit.jpg

Ausgabezeile am Ende:  fertig in Xs -> PFAD  [seed=N | reproduzieren: --seed N]
"""

import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

MODEL = "flux-3-image-klein"
ENDPOINT = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/" + MODEL
KEY_FILE = os.path.expanduser("~/.config/flux/nvidia-api-key")
PI_AUTH = os.path.expanduser("~/.pi/agent/auth.json")
LO, HI = 512, 1568  # NIM akzeptiert nur 512..1568, Schritt 16


def api_key():
    """Key aus Umgebung, eigener Datei oder pi-Ablage - nie aus dem Repo."""
    env = os.environ.get("NVIDIA_API_KEY")
    if env and env.strip():
        return env.strip()
    if os.path.isfile(KEY_FILE):
        key = open(KEY_FILE).read().strip()
        if key:
            return key
    try:
        entry = json.load(open(PI_AUTH)).get("nvidia")
        if entry and entry.get("key"):
            return entry["key"]
    except (OSError, ValueError):
        pass
    sys.exit("Kein API-Key gefunden. Eine der drei Quellen einrichten:\n"
             "  export NVIDIA_API_KEY=...\n"
             "  mkdir -p ~/.config/flux && echo KEY > %s && chmod 600 %s\n"
             "  oder /login nvidia in pi" % (KEY_FILE, KEY_FILE))


def fit(w, h):
    """Auf das 16er-Raster biegen, dabei das Seitenverhaeltnis halten."""

    def snap(v):
        return max(16, round(v / 16) * 16)

    w, h = snap(w), snap(h)
    if min(w, h) < LO:
        w, h = snap(w * LO / min(w, h)), snap(h * LO / min(w, h))
    if max(w, h) > HI:
        w, h = snap(w * HI / max(w, h)), snap(h * HI / max(w, h))
    return min(max(w, LO), HI), min(max(h, LO), HI)


def parse_size(text):
    try:
        w, h = (int(p) for p in text.lower().split("x", 1))
    except ValueError:
        sys.exit(f"--size erwartet WxH, erhalten: {text}")
    return w, h


def main():
    ap = argparse.ArgumentParser(description="FLUX 3 Image Klein via NVIDIA NIM")
    ap.add_argument("prompt", help="Text-to-Image-Prompt")
    ap.add_argument("-o", "--out", default=None,
                    help="Zieldatei, oder Verzeichnis (mit /) fuer <prefix>-<seed>.jpg")
    ap.add_argument("--size", default="1024x1024", metavar="WxH",
                    help="gewuenschtes Format, wird auf NIMs Raster gesetzt (Default 1024x1024)")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prefix", default="flux3-klein", help="Dateiname ohne -o")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", action="store_true", help="Ergebnis als JSON statt Textzeile")
    a = ap.parse_args()

    want = parse_size(a.size)
    width, height = fit(*want)
    if (width, height) != want and abs(want[0] / want[1] - width / height) > 0.06:
        print("Hinweis: %dx%d ist innerhalb 512..1568 nicht darstellbar - nehme %dx%d."
              % (*want, width, height), file=sys.stderr)
    elif (width, height) != want:
        print("Hinweis: %dx%d -> %dx%d (NIM-Raster)." % (*want, width, height), file=sys.stderr)

    seed = a.seed if a.seed is not None else random.randrange(2 ** 31)

    body = {"prompt": a.prompt, "width": width, "height": height, "seed": seed, "steps": a.steps}
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key()},
    )
    if not a.json:
        print("NIM %s: seed=%d %dx%d steps=%d ..."
              % (MODEL, seed, width, height, a.steps))

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as exc:
        sys.exit("NIM HTTP %d: %s" % (exc.code, exc.read().decode()[:400]))
    except Exception as exc:
        sys.exit("NIM-Anfrage fehlgeschlagen: %s" % exc)

    art = (reply.get("artifacts") or [{}])[0]
    if not art.get("base64"):
        sys.exit("NIM lieferte kein Bild: %s" % json.dumps(reply)[:400])
    if a.out and (a.out.endswith(os.sep) or os.path.isdir(a.out)):
        os.makedirs(a.out, exist_ok=True)
        out = os.path.join(a.out, "%s-%d.jpg" % (a.prefix, seed))
    else:
        out = a.out or "%s-%d.jpg" % (a.prefix, seed)
    with open(out, "wb") as fh:
        fh.write(base64.b64decode(art["base64"]))
    secs = time.time() - t0

    if a.json:
        print(json.dumps({"out": os.path.abspath(out), "seed": seed, "width": width,
                          "height": height, "steps": a.steps, "seconds": round(secs, 1),
"prompt": a.prompt}))
    else:
        print("fertig in %.1fs -> %s  [seed=%d | reproduzieren: --seed %d]"
              % (secs, os.path.abspath(out), seed, seed))


if __name__ == "__main__":
    main()
