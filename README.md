# comfyflux

Bildgenerierung mit FLUX über ComfyUI (lokal) und NVIDIA NIM (Cloud). Ein Repo
für Client **und** Server: die Clients sprechen HTTP mit ComfyUI, die Skripte
`start-comfy.sh` / `stop-comfy.sh` verwalten die lokale Instanz.

| Befehl | Modell | Läuft wo |
|---|---|---|
| `flux2` | FLUX.2 Klein (4B/9B, fp8) | ComfyUI |
| `flux3` | FLUX.3 Image Klein (GGUF) | ComfyUI |
| `flux3-nim` | FLUX.3 Image Klein | NVIDIA NIM (Cloud) |

`flux2` und `flux3` teilen sich `comfy_client.py` (HTTP, Upload, Graph-Ausführung,
Autostart). `flux3-nim` ist eigenständig – anderer Anbieter, anderes Protokoll.

## Installation

```bash
git clone git@github.com:paltian/comfyflux.git ~/dev/comfyflux
ln -sf ~/dev/comfyflux/flux2.py     ~/.local/bin/flux2
ln -sf ~/dev/comfyflux/flux3.py     ~/.local/bin/flux3
ln -sf ~/dev/comfyflux/flux3-nim.py ~/.local/bin/flux3-nim
```

Nur Python-Standardbibliothek, keine Abhängigkeiten.

## Server-Adresse

Standard ist `http://127.0.0.1:8188`. Auf einem Rechner ohne lokale Instanz zeigt
`COMFYFLUX_SERVER` auf den entfernten Server:

```bash
export COMFYFLUX_SERVER='http://volleymasterL:8188'
```

`--server` schlägt die Variable, die Variable schlägt den Standard.

## Autostart

ComfyUI wird **nur lokal und nur bei Bedarf** gestartet:

| Lage | Verhalten |
|---|---|
| Server antwortet | nichts |
| nicht erreichbar, Adresse entfernt | Fehler mit dem Startbefehl für den Server – **kein** Startversuch |
| nicht erreichbar, Adresse lokal, `~/dev/comfyui` vorhanden | `start-comfy.sh`, dann bis 180 s auf Bereitschaft warten |
| nicht erreichbar, Adresse lokal, kein `~/dev/comfyui` | Fehler mit Hinweis auf `--server` |
| `--no-autostart` | nie starten |

Auf einem Laptop ohne ComfyUI greift der Autostart also nie; auf dem Server, auf
dem ComfyUI eingerichtet ist, schon. Gestartet wird zuerst über eine
systemd-User-Unit `comfyui.service`, falls es sie gibt – sonst über
`start-comfy.sh`.

### Als systemd-Dienst

```bash
cp comfyui.service.example ~/.config/systemd/user/comfyui.service
# CUDA_VISIBLE_DEVICES anpassen (./start-comfy.sh gpus zeigt die Karten)
systemctl --user daemon-reload && systemctl --user enable --now comfyui.service
loginctl enable-linger $USER      # läuft auch ohne Login-Sitzung
```

Die Unit legt die GPU fest. `stop-comfy.sh` beendet den Dienst mit SIGTERM;
systemd zählt das als sauberes Ende und startet nicht neu – `Restart=on-failure`
greift nur bei Absturz oder OOM-Kill. Ohne `[Install]`-Sektion startet der Dienst
nicht beim Booten, dann nur manuell.

## Key für `flux3-nim`

Der Key steht **nie** im Repo. Reihenfolge:

1. `$NVIDIA_API_KEY`
2. `~/.config/flux/nvidia-api-key` (nur der Key, `chmod 600`)
3. `~/.pi/agent/auth.json`, Eintrag `nvidia` (von pi verwaltet)

```bash
mkdir -p ~/.config/flux && echo 'nvapi-...' > ~/.config/flux/nvidia-api-key
chmod 600 ~/.config/flux/nvidia-api-key
```

## Server einrichten (einmalig)

**ComfyUI ≥ 0.41** – ältere Versionen kennen `flux3` nicht im `CLIPType`:

```bash
cd ~/dev/comfyui && git pull && venv/bin/python -m pip install -r requirements.txt
```

**Custom Node für GGUF** (ComfyUI selbst kann kein GGUF):

```bash
git clone https://github.com/city96/ComfyUI-GGUF ~/dev/comfyui/custom_nodes/ComfyUI-GGUF
~/dev/comfyui/venv/bin/python -m pip install --upgrade gguf
```

**Modelle** nach `~/models/comfyui/`:

```bash
# DiT, GGUF Q5_K_XL (4,17 GiB) - gemessen: LPIPS 0,020 / SSIM 0,966 gegen bf16
hf download unsloth/flux-3-image-klein-GGUF flux-3-image-klein-UD-Q5_K_XL.gguf \
  --local-dir ~/models/comfyui/diffusion_models

# Text-Encoder Qwen3-VL-4B int8 (4,52 GiB).
# BFLs Repo ist gated (HTTP 401 ohne Token); der Spiegel ist byte-identisch:
hf download ml-resources/flux-3-image-klein-comfy \
  text_encoders/qwen3vl_4b_int8_convrot.safetensors --local-dir ~/models/comfyui

# VAE: flux2-vae.safetensors, meist schon vorhanden (0,31 GiB)
```

Gemessen auf einer RTX 3060 12 GB: Peak 9,6 GB von 12,3 GB, ~50 s für
1024×1024 mit 15 Schritten. Größere Quants (Q8_0, int8) passen neben einem
zweiten GPU-Prozess nicht mehr.

## Beispiele

```bash
# Text zu Bild
flux3 "ein Igel mit Partyhut, Studiofoto" --size 1024x1024 -o igel.png

# Editing: die Ausgabegröße kommt aus der Referenz, --size ist hier ohne Wirkung
flux3 "dieselbe Szene nachts bei Vollmond" --image igel.png -o nacht.png

# Multi-Reference: bis 10 Bilder, in Reihenfolge
flux3 "kombiniere beide Motive" --image a.png --image b.png -o mix.png

# reproduzierbar und maschinenlesbar
flux3 "..." --seed 42 --json

# FLUX.2 mit automatischer Profilwahl (4b auf 12 GB, 9b ab ~15 GB VRAM)
flux2 "ein Igel mit Partyhut" -o igel2.png
flux2 "mehr Details" --image foto.png --scale 2 --ref-mp 2 -o scharf.png

# Cloud, ohne VRAM-Bedarf
flux3-nim "a lighthouse in a storm" --size 1280x720 -o leuchtturm.jpg

# gross: Basis 1 MP, Ziel 4 MP - DiT klein, danach KI-Upscale
flux3 "..." --scale 2 -o gross.png
flux2 "..." --scale 2 -o gross2.png

# gross ohne Upscaler: der DiT rechnet direkt in der Zielgroesse (langsamer)
flux3 "..." --scale 2 --scale-mode dit -o gross-dit.png

# festes Format statt Faktor (schliesst --scale aus)
flux3 "..." --size 1280x720 -o breit.png

# vorhandenes Bild nur aufwerten (kein DiT)
flux3 "unused" --image foto.png --scale 2 --scale-mode upscale -o foto-2x.png
```

## Hinweise

- **`--size` und `--scale` schließen sich aus** – argparse verhindert die
  Kombination. `--size WxH` ist das Ziel absolut, `--scale N` das Ziel als
  Faktor auf die Basis (`--ref-mp`). Ohne beide ist das Ziel die Basis selbst.
  Beim Editing gibt es kein `--size`: dort kommt die Größe aus der Referenz.
- **`--ref-mp`** ist die Basisfläche in Megapixeln (Default 1.0 ≈ 1024×1024):
  bei Referenzen deren Größe (Lanczos, Format bleibt), bei t2i die Basis des
  DiT (1:1). `0` = Originalmaße der Referenz. Bei t2i ohne `--size` ist das
  Ergebnis quadratisch.
- **Zielgröße und Weg sind getrennt.** Die Zielgröße ergibt sich aus `--size`
  und `--scale` – bei Referenzen ist die Basis die
  Referenzfläche. Der `--scale-mode` bestimmt nur, **wer** diese Zielgröße
  erzeugt:

  | Modus | Wer erzeugt das Ziel | Typische Zeit (2048²) |
  |---|---|---|
  | `dit` (regen) | der DiT selbst | 170 s |
  | `post-dit` (post, Default) | DiT in Basisgröße + KI-Upscale | 48 s |
  | `upscale-no-dit` (upscale) | nur KI-Upscale, kein DiT | 12 s |

  `pre-dit` gibt es nicht mehr: die Referenzgröße steuert `--ref-mp`, nicht der
  Modus.
- **`--ref-mp`** steuert die Größe *jeder Referenz* vor dem Encoding:
  Zielfläche = `MP` × 1024², Seitenverhältnis bleibt, **Lanczos** – also eine
  Interpolation, keine Detailgewinnung. Ein kleines Bild auf 1 MP hochzuziehen
  bringt kein neues Detail, kostet aber Latent-Tokens. `1.0` ≈ 1024×1024
  (genau 1,048576 MP), `0` = Originalmaße behalten. Bei Referenzen folgt die
  Ausgabegröße dieser Fläche (× `--scale`), weil das Latent aus der Referenz
  kommt.
- **Sampler**: Default `er_sde` mit 15 Schritten (offizielles T2I-Template).
  Das offizielle Editing-Template nutzt `euler` mit 16 Schritten –
  überschreibbar mit `--sampler euler --steps 16`.
- **NIM-Grenzen**: `flux3-nim` akzeptiert nur 512…1568 in 16er-Schritten;
  andere Formate werden mit Erhalt des Seitenverhältnisses eingepasst.
  Editing gibt es nur lokal – NIM verlangt Asset-Referenzen (`example_id`).
- **PNG-Metadaten** (Workflow, Prompts) werden standardmäßig entfernt;
  `--metadata` behält sie.
- **`flux2`-Profile** `4b-base` und `9b-base` brauchen Dateien, die nicht Teil
  dieses Setups sind.

## Lizenz

MIT (siehe `LICENSE`). Modelle haben eigene Lizenzen: FLUX 3 Image Klein steht
unter der FLUX Kommunity License (nicht-kommerziell frei, kommerziell für
Qualifying Users unter 5 Mio. USD Jahresumsatz).
