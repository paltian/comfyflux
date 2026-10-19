#!/usr/bin/env python3
"""comfy_client.py - gemeinsamer Unterbau der ComfyUI-Clients (flux2.py, flux3.py).

Enthaelt alles, was beide brauchen: HTTP gegen die ComfyUI-Instanz, Bild-Upload
und -Download, PNG-Metadaten entfernen, Graph abgeben und einsammeln, Autostart
des lokalen Servers sowie die gemeinsamen Argparse-Bausteine.

Kein Key, kein Zustand, keine Modelle - nur Transport und Werkzeug.
"""
import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UPSCALE_MODEL = "4x-UltraSharp.pth"
# Standard-Server: $COMFYFLUX_SERVER erlaubt es, auf einem Rechner ohne lokale
# Instanz auf einen entfernten Server zu zeigen, ohne --server zu tippen.
DEFAULT_SERVER = os.environ.get("COMFYFLUX_SERVER", "http://127.0.0.1:8188")
START_SCRIPT = "start-comfy.sh"
SYSTEMD_UNIT = "comfyui.service"
LOCAL_COMFY = "~/dev/comfyui"
START_TIMEOUT = 180


# ---------------------------------------------------------------- HTTP

def post_json(server, path, payload, timeout=60):
    req = urllib.request.Request(server + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("Server-Fehler HTTP %s: %s" % (e.code, e.read().decode()[:800]))


def get_json(server, path, timeout=30):
    with urllib.request.urlopen(server + path, timeout=timeout) as r:
        return json.load(r)


def server_alive(server, timeout=10):
    try:
        with urllib.request.urlopen(server + "/system_stats", timeout=timeout) as r:
            r.read(1)
        return True
    except Exception:
        return False


def is_loopback(server):
    host = urllib.parse.urlparse(server).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def systemd_unit_exists():
    """Ist eine User-Unit fuer den Server vorhanden?"""
    if not shutil.which("systemctl"):
        return False
    try:
        r = subprocess.run(["systemctl", "--user", "cat", SYSTEMD_UNIT],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def start_server(quiet=False):
    """Startet den lokalen Server ueber die systemd-Unit, falls es eine gibt -
    sonst ueber start-comfy.sh. Liefert die verwendete Methode."""
    if systemd_unit_exists():
        subprocess.run(["systemctl", "--user", "start", SYSTEMD_UNIT], check=False)
        return "systemctl --user start %s" % SYSTEMD_UNIT
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), START_SCRIPT)
    if not os.path.isfile(script):
        raise SystemExit("Startskript nicht gefunden: %s" % script)
    subprocess.run(["bash", script], check=False)
    return script


def ensure_server(server, autostart=True, quiet=False):
    """Stellt sicher, dass die Instanz antwortet. Startet sie NUR lokal und nur,
    wenn hier ein ComfyUI eingerichtet ist - bei entfernter URL nie."""
    if server_alive(server):
        return False
    if not autostart:
        raise SystemExit("ComfyUI nicht erreichbar unter %s (--no-autostart)." % server)
    if not is_loopback(server):
        raise SystemExit(
            "ComfyUI nicht erreichbar unter %s.\n"
            "  Entfernte Instanz - Autostart gibt es nur lokal. Auf dem Server starten:\n"
            "    bash ~/dev/comfyflux/%s" % (server, START_SCRIPT))
    if not os.path.isfile(os.path.expanduser(LOCAL_COMFY) + "/main.py"):
        raise SystemExit(
            "ComfyUI nicht erreichbar unter %s und hier nicht eingerichtet (%s fehlt).\n"
            "  Entweder dort starten oder mit --server auf eine laufende Instanz zeigen."
            % (server, LOCAL_COMFY))
    how = start_server(quiet=quiet)
    if not quiet:
        print("ComfyUI laeuft nicht - starte lokal (%s) ..." % how)
    t0 = time.time()
    while time.time() - t0 < START_TIMEOUT:
        time.sleep(2)
        if server_alive(server):
            if not quiet:
                print("ComfyUI bereit nach %.0fs." % (time.time() - t0))
            return True
    raise SystemExit("ComfyUI nach %ds nicht bereit geworden." % START_TIMEOUT)


def widget_choices(val):
    """Dateilisten aus object_info-Widgets: [liste, meta] oder ["COMBO", {options}]."""
    if not isinstance(val, list):
        return set()
    if val and isinstance(val[0], list):
        return set(val[0])
    if len(val) > 1 and val[0] == "COMBO" and isinstance(val[1], dict):
        return set(val[1].get("options", []))
    return set(v for v in val if isinstance(v, str) and v != "COMBO")


def preflight(server, checks, quiet=False):
    """Prueft VOR dem Sampling, ob die Gewichte auf dem Server bekannt sind.
    checks: Liste von (Node-Klasse, Eingabename, Dateiname)."""
    problems = []
    for node, key, fname in checks:
        if not fname:
            continue
        try:
            info = get_json(server, "/object_info/" + node)
            avail = widget_choices(info[node]["input"]["required"][key])
        except Exception as e:
            raise SystemExit("Knoten %s nicht abfragbar: %s" % (node, e))
        if fname not in avail:
            problems.append("  %s: FEHLT auf dem Server\n     vorhanden: %s"
                            % (fname, ", ".join(sorted(avail)) or "(keine)"))
    if problems:
        raise SystemExit("Modell-Problem(e):\n" + "\n".join(problems))


def node_available(server, node):
    try:
        get_json(server, "/object_info/" + node)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- Bilder

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
        info = get_json(server, "/object_info/LoadImage")
        avail = widget_choices(info["LoadImage"]["input"]["required"]["image"])
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


def collect_refs(server, refs, quiet=False):
    """Referenzen hochladen bzw. pruefen. Liefert (Namen, Groessen)."""
    names, sizes = [], []
    for kind, value in (refs or []):
        name, data = fetch_ref(kind, value, server)
        if name is not None:
            # laufende Nummer: gleiche Basenamen (z.B. zwei URLs ohne Endung)
            # wuerden sich beim Upload sonst gegenseitig ueberschreiben
            name = "ref%d-%s" % (len(names) + 1, name)
            sizes.append(image_size(data))
            names.append(upload_image(server, name, data))
        else:
            sizes.append(image_size(download_input(server, value) or b""))
            names.append(value)
        if not quiet:
            print("Referenz: " + names[-1])
    return names, sizes


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


# ---------------------------------------------------------------- Ausfuehrung

def dims_for(area_px, aspect):
    """Breite/Hoehe fuer eine Flaeche mit gegebenem Seitenverhaeltnis, 16er-Raster."""
    w = int(round((area_px * aspect) ** 0.5))
    h = int(round((area_px / aspect) ** 0.5))
    return max(16, w - w % 16), max(16, h - h % 16)


def resolve_sizes(p, refs, sizes):
    """Basis- und Zielgroesse fuer beide CLIs.

    --size und --scale schliessen sich aus (argparse erzwingt das). Basis ist
    --ref-mp: bei Referenzen deren Flaeche mit dem Format der Referenz, bei t2i
    1:1. 0 = Originalmasse der Referenz. Ziel ist --size (absolut) oder
    Basis x --scale. Wer das Ziel erzeugt, sagt --scale-mode.

    Liefert (basis_w, basis_h, ziel_w, ziel_h, basis_mp, ziel_mp)."""
    aspect = 1.0
    if refs and sizes and sizes[0]:
        aspect = sizes[0][0] / float(sizes[0][1])
    if p.ref_mp:
        base_area = p.ref_mp * 1048576.0
    elif refs and sizes and sizes[0]:
        base_area = float(sizes[0][0] * sizes[0][1])
    else:
        base_area = 1048576.0
    if p.scale is None:
        if p.size:
            tw, th = parse_size(p.size)
        else:
            tw, th = dims_for(base_area, aspect)
        bw, bh = tw, th
    else:
        bw, bh = dims_for(base_area, aspect)
        tw, th = dims_for(base_area * p.scale ** 2, aspect)
    return bw, bh, tw, th, base_area / 1048576.0, (tw * th) / 1048576.0


def add_upscale_tail(g, src, target_w, target_h, save_node, base=40):
    """KI-Upscaler + exakte Zielgroesse VOR den SaveImage-Knoten haengen.
    src ist der [node, slot]-Verweis auf das zu vergroessernde Bild."""
    a, b, c = str(base), str(base + 1), str(base + 2)
    g[a] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}}
    g[b] = {"class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": [a, 0], "image": src}}
    g[c] = {"class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": [b, 0], "upscale_method": "lanczos",
                       "megapixels": round(target_w * target_h / 1048576.0, 3),
                       "resolution_steps": 16}}
    g[save_node]["inputs"]["images"] = [c, 0]


def post_action(server, path, payload, timeout=30):
    """POST, dessen Antwort kein JSON ist: /interrupt und /queue liefern einen
    leeren Body - json.load wuerde daran scheitern."""
    req = urllib.request.Request(server + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read(1)


def running_prompt_id(server):
    """prompt_id des Auftrags, der gerade rechnet - oder None."""
    try:
        running = get_json(server, "/queue").get("queue_running") or []
        return running[0][1] if running else None
    except Exception:
        return None


def cancel_prompt(server, pid):
    """Bricht NUR den eigenen Auftrag ab und raeumt ihn aus der Queue.

    /interrupt ist global: es stoppt, was gerade rechnet - nicht den Auftrag mit
    der eigenen prompt_id. Auf einem Server, auf dem auch andere Auftraege laufen,
    wuerde ein blindes /interrupt fremde Arbeit toeten. Deshalb erst pruefen, wer
    laeuft; ein eigener, noch wartender Eintrag wird nur aus der Queue entfernt."""
    stopped = []
    try:
        current = running_prompt_id(server)
        if current == pid:
            post_action(server, "/interrupt", {})
            stopped.append("interrupt")
        elif current:
            stopped.append("anderer Auftrag laeuft (%s) - nicht abgebrochen" % current[:8])
        else:
            stopped.append("nichts lief")
    except Exception as e:
        stopped.append("interrupt fehlgeschlagen: %s" % e)
    try:
        post_action(server, "/queue", {"delete": [pid]})
        stopped.append("queue-delete")
    except Exception:
        pass
    return ", ".join(stopped)


def run_graph(server, graph, timeout, out=None, prefix="flux", metadata=False, quiet=False):
    """Graph abgeben, auf das Ergebnis warten, Bilder speichern.
    Liefert (pfade, sekunden, prompt_id)."""
    res = post_json(server, "/prompt", {"prompt": graph})
    if "prompt_id" not in res:
        raise SystemExit("Server-Fehler: " + json.dumps(res)[:500])
    pid = res["prompt_id"]
    t0 = time.time()
    hist = {}
    try:
        while time.time() - t0 < timeout:
            hist = get_json(server, "/history/" + pid)
            if pid in hist:
                break
            time.sleep(2)
        else:
            # Ohne Abbruch rechnet der Server weiter und belegt die GPU.
            what = cancel_prompt(server, pid)
            raise SystemExit(
                "Timeout nach %ds - Auftrag abgebrochen (%s, prompt_id=%s).\n"
                "  Fuer laengere Laeufe: --timeout erhoehen. Laeuft trotzdem noch etwas:\n"
                "    curl -s -X POST %s/interrupt -d '{}' -H 'Content-Type: application/json'"
                % (timeout, what, pid, server))
    except KeyboardInterrupt:
        what = cancel_prompt(server, pid)
        raise SystemExit("Abgebrochen (%s, prompt_id=%s)" % (what, pid))

    entry = hist.get(pid, {})
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for m in status.get("messages", []):
            if m[0] == "execution_error":
                raise SystemExit("Sampling-Fehler: " + json.dumps(m[1])[:800])
        raise SystemExit("Sampling-Fehler: " + json.dumps(status)[:600])
    imgs = [im for o in entry.get("outputs", {}).values() for im in o.get("images", [])]
    if not imgs:
        raise SystemExit("keine Bilder in outputs: " + json.dumps(entry)[:600])

    saved = []
    for i, im in enumerate(imgs):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                    "type": im.get("type", "output")})
        dest = out or ("%s_%d.png" % (prefix, i))
        if out and len(imgs) > 1:
            dest = out.replace(".png", "_%d.png" % i)
        with urllib.request.urlopen(server + "/view?" + q, timeout=120) as r:
            data = r.read()
        if not metadata:
            data = strip_png_metadata(data)
        with open(dest, "wb") as f:
            f.write(data)
        saved.append(dest)
    return saved, time.time() - t0, pid


# ---------------------------------------------------------------- Argparse

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
    HINT = {}

    def __call__(self, parser, namespace, values, option_string=None):
        raise SystemExit("%s wurde ersetzt durch: %s" % (option_string, self.HINT[option_string]))


def add_common_args(ap, default_server=None):
    """Argumente, die flux2 und flux3 gleich haben."""
    ap.add_argument("-o", "--out", default=None,
                    help="Zieldatei, oder Verzeichnis (mit /) fuer <prefix>_<seed>.png")
    ap.add_argument("--image", action=AddRef, default=None, dest="refs",
                    help="Referenzbild: lokale Datei ODER http(s)-URL (mehrfach moeglich)")
    ap.add_argument("--url", action=AddRef, default=None, dest="refs",
                    help="wie --image, aber explizit als URL (Alias)")
    ap.add_argument("--server-image", action=AddRef, default=None, dest="refs",
                    help="Referenzbild per Dateiname aus ComfyUIs input-Ordner (mehrfach moeglich)")
    ap.add_argument("--server", default=default_server or DEFAULT_SERVER,
                    help="ComfyUI-Adresse (Default: $COMFYFLUX_SERVER oder %s)" % DEFAULT_SERVER)
    ap.add_argument("--no-autostart", action="store_true",
                    help="Server nicht selbst starten, auch wenn er lokal eingerichtet ist")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prefix", default=None, help="Dateiname ohne -o")
    ap.add_argument("--metadata", action="store_true",
                    help="Workflow/Prompts in die PNG-Metadaten schreiben "
                         "(Standard: Metadaten werden entfernt)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Wartezeit in Sekunden auf das Ergebnis (Default: 900). "
                     "Bei Timeout wird der eigene Auftrag abgebrochen.")
    ap.add_argument("--json", action="store_true", help="Ergebnis als JSON statt Textzeile")


def out_path(out, prefix, seed):
    """-o als Datei oder Verzeichnis; sonst <prefix>_<seed>.png im cwd."""
    name = "%s_%d.png" % (prefix, seed)
    if out and (out.endswith(os.sep) or os.path.isdir(out)):
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, name)
    return out or name


def parse_size(text, multiple=16, minimum=16):
    """'1024x1024' oder '1024' -> (w, h) auf ein Vielfaches gerundet."""
    t = str(text).lower()
    if "x" in t:
        try:
            w, h = (int(v) for v in t.split("x", 1))
        except ValueError:
            raise SystemExit("Format erwartet WxH, erhalten: %s" % text)
    else:
        try:
            w = h = int(t)
        except ValueError:
            raise SystemExit("Format erwartet WxH, erhalten: %s" % text)
    return max(minimum, w - w % multiple), max(minimum, h - h % multiple)
