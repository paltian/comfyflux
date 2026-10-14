#!/bin/bash
pkill -f "main\.py" 2>/dev/null
sleep 2
cd ~/dev/comfyui
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  nohup venv/bin/python main.py --listen 0.0.0.0 --port 8188 \
  --extra-model-paths-config /home/markus/dev/comfyui/extra-model-paths.yaml \
  > comfyui.log 2>&1 &
sleep 12
ss -tln | grep -q 8188 && echo "SERVER LÄUFT" || { echo "FEHLER:"; tail -5 ~/dev/comfyui/comfyui.log; }
