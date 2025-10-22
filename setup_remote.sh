#!/bin/bash
# EdgeVLM Remote Setup with Ngrok
# Exposes EdgeVLM services remotely for camera connections

set -e

echo "======================================"
echo "EdgeVLM Remote Setup with Ngrok"
echo "======================================"

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed!"
    echo "Please install ngrok first:"
    echo "  curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
    echo "  echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list"
    echo "  sudo apt update && sudo apt install ngrok"
    exit 1
fi

echo "✅ ngrok is installed"

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "⚠️  ngrok authentication required"
    echo "Please run: ngrok config add-authtoken YOUR_TOKEN"
    echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
fi

echo "✅ ngrok is authenticated"

# Create directories
mkdir -p logs
mkdir -p configs

# Start EdgeVLM services
echo "Starting EdgeVLM services..."

# Start main API
echo "Starting main API (port 8000)..."
python main.py --port 8000 > logs/main_api.log 2>&1 &
MAIN_PID=$!

# Start camera registration API
echo "Starting camera registration API (port 8002)..."
python api_camera_registration.py --port 8002 > logs/registration_api.log 2>&1 &
REG_PID=$!

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Check if services are running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Main API failed to start"
    exit 1
fi

if ! curl -s http://localhost:8002/health > /dev/null; then
    echo "❌ Registration API failed to start"
    exit 1
fi

echo "✅ All services started successfully"

# Start ngrok tunnels
echo "Starting ngrok tunnels..."

# Start ngrok for camera registration API (most important)
echo "Exposing camera registration API..."
ngrok http 8002 --log=stdout > logs/ngrok_registration.log 2>&1 &
NGROK_REG_PID=$!

# Start ngrok for main API
echo "Exposing main API..."
ngrok http 8000 --log=stdout > logs/ngrok_main.log 2>&1 &
NGROK_MAIN_PID=$!

# Wait for ngrok to start
echo "Waiting for ngrok tunnels to establish..."
sleep 10

# Get tunnel URLs
echo "Getting tunnel URLs..."

# Get registration API URL
REG_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tunnel in data['tunnels']:
    if tunnel['config']['addr'] == 'localhost:8002':
        print(tunnel['public_url'])
        break
")

# Get main API URL
MAIN_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tunnel in data['tunnels']:
    if tunnel['config']['addr'] == 'localhost:8000':
        print(tunnel['public_url'])
        break
")

if [ -z "$REG_URL" ]; then
    echo "❌ Failed to get registration API tunnel URL"
    exit 1
fi

if [ -z "$MAIN_URL" ]; then
    echo "❌ Failed to get main API tunnel URL"
    exit 1
fi

echo ""
echo "======================================"
echo "EdgeVLM Services Exposed Remotely!"
echo "======================================"
echo ""
echo "📹 Camera Registration API (Recommended):"
echo "   URL: $REG_URL"
echo "   Registration: $REG_URL/register"
echo "   Health: $REG_URL/health"
echo ""
echo "🖼️  Main API (Local Image Processing):"
echo "   URL: $MAIN_URL"
echo "   Caption: $MAIN_URL/caption"
echo "   VQA: $MAIN_URL/vqa"
echo "   Health: $MAIN_URL/health"
echo ""

# Save configuration
cat > configs/remote_config.json << EOF
{
  "services": {
    "camera_registration": {
      "url": "$REG_URL",
      "port": 8002,
      "description": "Camera Registration API (self-registration)"
    },
    "main_api": {
      "url": "$MAIN_URL",
      "port": 8000,
      "description": "Main EdgeVLM API (local image processing)"
    }
  },
  "camera_configs": {
    "registration": {
      "api_base_url": "$REG_URL",
      "registration_endpoint": "$REG_URL/register",
      "heartbeat_endpoint": "$REG_URL/cameras/{camera_id}/heartbeat",
      "upload_endpoint": "$REG_URL/cameras/{camera_id}/upload",
      "status_endpoint": "$REG_URL/cameras/{camera_id}/status"
    }
  },
  "timestamp": $(date +%s),
  "ngrok_status": true
}
EOF

echo "📁 Configuration saved to: configs/remote_config.json"
echo ""

# Create example camera client command
cat > configs/camera_client_example.sh << EOF
#!/bin/bash
# Example camera client command

python examples/remote_camera_client.py \\
  --ngrok-url "$REG_URL" \\
  --name "Remote Camera" \\
  --location "Remote Location" \\
  --type usb \\
  --device 0
EOF

chmod +x configs/camera_client_example.sh

echo "📝 Example camera client command:"
echo "   ./configs/camera_client_example.sh"
echo ""

# Create monitoring script
cat > monitor_services.sh << 'EOF'
#!/bin/bash
echo "EdgeVLM Remote Services Monitor"
echo "================================"

echo "Service Status:"
echo "Main API (8000): $(curl -s http://localhost:8000/health | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo 'DOWN')"
echo "Registration API (8002): $(curl -s http://localhost:8002/health | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo 'DOWN')"

echo ""
echo "Ngrok Tunnels:"
curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tunnel in data['tunnels']:
    print(f'{tunnel[\"name\"]}: {tunnel[\"public_url\"]} -> {tunnel[\"config\"][\"addr\"]}')
"

echo ""
echo "Registered Cameras:"
curl -s http://localhost:8002/cameras | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data:
    for camera_id, info in data.items():
        print(f'{info[\"name\"]}: {info[\"status\"]} (uptime: {info[\"uptime_seconds\"]:.1f}s)')
else:
    print('No cameras registered')
" 2>/dev/null || echo "No cameras registered"
EOF

chmod +x monitor_services.sh

echo "📊 Monitoring script created: ./monitor_services.sh"
echo ""

echo "🎉 Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Test camera registration: curl $REG_URL/health"
echo "2. Run camera client: ./configs/camera_client_example.sh"
echo "3. Monitor services: ./monitor_services.sh"
echo "4. View ngrok dashboard: http://localhost:4040"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $MAIN_PID $REG_PID $NGROK_REG_PID $NGROK_MAIN_PID 2>/dev/null || true
    echo "All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Keep running
while true; do
    sleep 1
done
