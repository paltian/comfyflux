#!/bin/bash
# start-comfy.sh - ComfyUI-Instanz starten, GPU waehlbar
#
# Aufruf:   start-comfy.sh [auto|<GPU-Index>|status|gpus]
#   auto (Default) : GPU mit dem meisten freien VRAM
#                    (Tie-Break: groesster Gesamt-VRAM)
#   <GPU-Index>    : expliziter nvidia-smi-Index (PCI-Bus-Reihenfolge)
#   status         : Zustand der Instanz anzeigen
#   gpus           : Kartentabelle (Index, PCI-Bus, VRAM, PCIe-Link, Belegung)
#
# GPUs mit weniger als MIN_FREE_MB freiem VRAM kommen bei auto nicht in
# Frage; bei expliziter Auswahl gibt es nur eine Warnung.

CODE_DIR="$HOME/dev/comfyui"
MODEL_CFG="/home/markus/dev/comfyui/extra-model-paths.yaml"
PORT=8188
MIN_FREE_MB=8192
PATTERN="venv/bin/python main.py"

usage() {
  grep "^#" "$0" | sed "s/^# \{0,1\}//" | head -12
}

gpu_table() {
  nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits
}

gpus() {
  local tmp1 tmp2
  tmp1=$(mktemp) && tmp2=$(mktemp)
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used,memory.free,memory.total,pcie.link.width.current,pcie.link.gen.max \
    --format=csv,noheader,nounits > "$tmp1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$tmp2"
  awk -F", " '
    NR == FNR {
      if (NF < 4) next
      name = $3; sub(".*/", "", name)
      procs[$1] = procs[$1] sprintf("%s%s (%.1f GB)", (procs[$1] ? ", " : ""), name, $4 / 1024)
      next
    }
    {
      printf "%-3s %-12s %5.1f/%5.1f/%5.1f GB   x%-2s Gen%-1s   %s\n", \
        $1, substr($3, 1, 12), $5 / 1024, $6 / 1024, $7 / 1024, $8, $9, (procs[$2] ? procs[$2] : "-")
    }
  ' "$tmp2" "$tmp1"
  rm -f "$tmp1" "$tmp2"
  echo
  local row idx free
  row=$(gpu_table | awk -F", " -v min="$MIN_FREE_MB" \
          '$3 >= min {print $1 "|" $3 "|" $4}' | sort -t"|" -k2,2nr -k3,3nr | head -1)
  if [ -n "$row" ]; then
    idx="${row%%|*}"
    free=$(echo "$row" | cut -d"|" -f2)
    echo "Auto-Auswahl waere: GPU $idx ($((free / 1024)) GB frei)"
  else
    echo "Auto-Auswahl: keine GPU mit >= $((MIN_FREE_MB / 1024)) GB frei"
  fi
  echo "Hinweis: PCIe-Werte zeigen die Verhandlung im Leerlauf; Gen wird erst unter Last hochgetaktet."
}

status() {
  local pid
  pid=$(pgrep -f "$PATTERN" | head -1)
  if [ -z "$pid" ]; then
    echo "ComfyUI: laeuft nicht"
    ss -tln | grep -q ":$PORT" && echo "Warnung: Port $PORT ist trotzdem belegt"
    exit 0
  fi
  echo "ComfyUI: laeuft (PID $pid)"
  ss -tln | grep -q ":$PORT" && echo "Port $PORT: offen" || echo "Port $PORT: NICHT offen"
  local visible
  visible=$(tr "\0" "\n" < /proc/$pid/environ | sed -n "s/^CUDA_VISIBLE_DEVICES=//p")
  echo "GPU-Index (nvidia-smi): $visible"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$visible"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid" || true
}

case "${1:-auto}" in
  status) status; exit 0 ;;
  gpus|list|cards) gpus; exit 0 ;;
  -h|--help|help) usage; exit 0 ;;
esac

pid=$(pgrep -f "$PATTERN" | head -1)
if [ -n "$pid" ]; then
  echo "ComfyUI laeuft bereits (PID $pid). Erst stop-comfy.sh ausfuehren."
  exit 1
fi

SEL="${1:-auto}"
if [ "$SEL" = "auto" ]; then
  ROW=$(gpu_table | awk -F", " -v min="$MIN_FREE_MB" \
          '$3 >= min {print $1 "|" $3 "|" $4}' | sort -t"|" -k2,2nr -k3,3nr | head -1)
  if [ -z "$ROW" ]; then
    echo "FEHLER: keine GPU mit >= $((MIN_FREE_MB / 1024)) GB freiem VRAM:"
    gpu_table
    exit 1
  fi
  GPU_IDX="${ROW%%|*}"
  FREE_MB=$(echo "$ROW" | cut -d"|" -f2)
  echo "Auto-Auswahl: GPU $GPU_IDX mit $((FREE_MB / 1024)) GB frei"
else
  GPU_IDX="$SEL"
  if ! echo "$GPU_IDX" | grep -qE "^[0-9]+$"; then
    echo "FEHLER: erwartet 'auto', 'status' oder eine GPU-Nummer."
    usage
    exit 1
  fi
  if ! gpu_table | grep -q "^$GPU_IDX,"; then
    echo "FEHLER: GPU $GPU_IDX gibt es nicht:"
    gpu_table
    exit 1
  fi
  FREE_MB=$(gpu_table | awk -F", " -v i="$GPU_IDX" '$1 == i {print $3}')
  if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
    echo "Warnung: GPU $GPU_IDX hat nur $((FREE_MB / 1024)) MB frei (< $((MIN_FREE_MB / 1024)) GB)"
  fi
fi

echo "Starte auf: $(gpu_table | awk -F", " -v i="$GPU_IDX" '$1 == i {print "GPU " $1 " (" $2 ")"}')"

cd "$CODE_DIR" || exit 1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDX" \
  nohup venv/bin/python main.py --listen 0.0.0.0 --port "$PORT" \
  --extra-model-paths-config "$MODEL_CFG" > comfyui.log 2>&1 &

for i in $(seq 1 30); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/system_stats" >/dev/null && break
  sleep 1
done

if curl -s --max-time 2 "http://127.0.0.1:$PORT/system_stats" >/dev/null; then
  echo "SERVER LÄUFT auf http://$(hostname):$PORT"
  status
else
  echo "FEHLER: Server nicht erreichbar. Log:"
  tail -10 "$CODE_DIR/comfyui.log"
  exit 1
fi
