#!/bin/bash
# stop-comfy.sh - ComfyUI-Instanz auf volleymasterL sauber beenden
# Gegenstueck zu start-comfy.sh. Beendet nur den ComfyUI-Prozess,
# strata/strata-vision bleiben unberuehrt.

PATTERN="venv/bin/python main.py"

PIDS=$(pgrep -f "$PATTERN")
if [ -z "$PIDS" ]; then
  ss -tln | grep -q ":8188" && echo "Kein ComfyUI-Prozess, aber Port 8188 noch belegt?!" || echo "ComfyUI laeuft nicht (Port 8188 frei)."
  exit 0
fi

echo "Beende ComfyUI (PID: $(echo $PIDS | tr '\n' ' '))..."
kill $PIDS 2>/dev/null

for i in $(seq 1 10); do
  pgrep -f "$PATTERN" >/dev/null || break
  sleep 1
done

if pgrep -f "$PATTERN" >/dev/null; then
  echo "haengt noch nach 10s - SIGKILL"
  kill -9 $(pgrep -f "$PATTERN") 2>/dev/null
  sleep 1
fi

if pgrep -f "$PATTERN" >/dev/null; then
  echo "FEHLER: Prozess laeuft weiterhin."
  exit 1
fi

if ss -tln | grep -q ":8188"; then
  echo "Warnung: Prozess tot, aber Port 8188 noch belegt."
  exit 1
fi

echo "ComfyUI gestoppt, Port 8188 frei."
