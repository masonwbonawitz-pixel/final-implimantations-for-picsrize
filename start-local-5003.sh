#!/bin/bash
# Start the server locally on port 5003 with frontend serving

cd "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING"

echo "🚀 Starting FULL STACK server on localhost:5003..."
echo ""
echo "📂 Main app: http://localhost:5003"
echo "🖥️  Desktop: http://localhost:5003/desktop.html"
echo "📱 Mobile: http://localhost:5003/mobile/"
echo "📱 Mobile (direct): http://localhost:5003/mobile/index.html"
echo "⚙️  Admin: http://localhost:5003/admin.html"
echo "🛒 Cart API: http://localhost:5003/api/create-cart"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Set environment variable for port
export PORT=5003
export FLASK_ENV=development

python3 server.py

