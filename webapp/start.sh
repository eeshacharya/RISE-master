#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# XAI Saliency Lab — quick-start script
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         XAI Saliency Comparison Lab                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Install Python dependencies ───────────────────────────────────────────
echo "▶ Installing Python dependencies…"
pip install -r requirements.txt -q

# ── 2. Check if ngrok is available ───────────────────────────────────────────
NGROK_URL=""
if command -v ngrok &>/dev/null; then
  echo "▶ ngrok found — starting tunnel on port 8080…"
  # Start ngrok in background, wait for it to bind
  ngrok http 8080 --log=stdout > /tmp/ngrok.log 2>&1 &
  NGROK_PID=$!
  sleep 3
  # Extract the public URL
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null || true)
  if [ -n "$NGROK_URL" ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────┐"
    echo "│  🌍 Public URL (share this or generate a QR code): │"
    echo "│                                                      │"
    echo "│   $NGROK_URL"
    echo "│                                                      │"
    echo "│  Generate QR code: https://qrcode.tec-it.com        │"
    echo "│  (paste the URL above and screenshot the QR)        │"
    echo "└─────────────────────────────────────────────────────┘"
  fi
else
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  ngrok NOT found. To share with your audience:           ║"
  echo "║                                                          ║"
  echo "║  Option A — Install ngrok (recommended, free):          ║"
  echo "║    brew install ngrok/ngrok/ngrok                        ║"
  echo "║    ngrok config add-authtoken <your-token>               ║"
  echo "║    ngrok http 8080    ← run in a second terminal         ║"
  echo "║                                                          ║"
  echo "║  Option B — Use cloudflared (no account needed):         ║"
  echo "║    brew install cloudflare/cloudflare/cloudflared        ║"
  echo "║    cloudflared tunnel --url http://localhost:8080         ║"
  echo "║                                                          ║"
  echo "║  Then paste the public URL into a QR code generator:     ║"
  echo "║    https://qrcode.tec-it.com                             ║"
  echo "╚══════════════════════════════════════════════════════════╝"
fi

echo ""
echo "▶ Starting Flask server on http://0.0.0.0:8080 …"
echo "  (first startup downloads ResNet-50 weights ~100 MB)"
echo "  Press Ctrl+C to stop."
echo ""
python app.py
