# Run Full Stack Locally on Port 5003

This folder contains a complete full-stack version that serves both frontend and backend.

## Quick Start

```bash
cd "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING"
./start-local-5003.sh
```

## URLs (Port 5003)

### Main Application
- **Home**: http://localhost:5003/
- **Mobile**: http://localhost:5003/mobile/
- **Desktop**: http://localhost:5003/desktop.html
- **Admin**: http://localhost:5003/admin.html

### API Endpoints

#### Cart Creation (NEW)
```bash
curl -X POST http://localhost:5003/api/create-cart \
  -H "Content-Type: application/json" \
  -d '{
    "size": 75,
    "quantity": 1,
    "customization_id": "test-123",
    "addons": {"stand": true, "mounting_dots": false}
  }'
```

#### Other APIs
- `GET /api/prices` - Get current prices
- `GET /api/images` - Get product images
- `GET /api/content` - Get editable content
- `GET /api/shopify/variants` - Get Shopify variant IDs
- `POST /upload-for-checkout` - Save customization files
- `POST /generate-obj` - Generate OBJ/MTL files

## Features

✅ Serves frontend HTML (mobile/desktop)
✅ Backend API for 3D file generation
✅ Shopify cart creation via Storefront API
✅ Admin panel for prices/images/content
✅ File uploads and order management

## Testing in Browser

Open http://localhost:5003/mobile/ or http://localhost:5003/desktop.html

### Test Cart Creation from Console

```javascript
// Test cart creation
async function testCart() {
    const response = await fetch('http://localhost:5003/api/create-cart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            size: 75,
            quantity: 1,
            customization_id: 'test-' + Date.now(),
            addons: { stand: true, mounting_dots: false }
        })
    });
    
    const data = await response.json();
    console.log('Cart response:', data);
    
    if (data.checkout_url) {
        console.log('✅ Checkout URL:', data.checkout_url);
        // window.location.href = data.checkout_url; // Uncomment to redirect
    }
}

testCart();
```

## Environment Variables

The server uses these environment variables (optional, has defaults):

```bash
# Shopify Storefront API
SHOPIFY_SHOP_DOMAIN=yourstore.myshopify.com
SHOPIFY_STOREFRONT_TOKEN=your_token
SHOPIFY_STOREFRONT_API_VERSION=2025-01

# Variant IDs (has defaults from previous setup)
SHOPIFY_VARIANT_48=gid://shopify/ProductVariant/...
SHOPIFY_VARIANT_75=gid://shopify/ProductVariant/...
SHOPIFY_VARIANT_96=gid://shopify/ProductVariant/...
SHOPIFY_VARIANT_STAND=gid://shopify/ProductVariant/...
SHOPIFY_VARIANT_MOUNTING=gid://shopify/ProductVariant/...
```

## File Structure

```
FOR_HOSTING/
├── server.py                    # Main Flask app (serves frontend + API)
├── mobile/
│   └── index.html              # Mobile frontend
├── desktop.html                # Desktop frontend
├── admin.html                  # Admin panel
├── shopify_api.py              # Shopify Admin API
├── shopify_storefront_api.py   # Shopify Storefront API (cart creation)
├── start-local-5003.sh         # Start on port 5003
└── assets/                     # Static files (CSS, JS, images)
```

## Differences from Main 5001 Folder

| Feature | 5001/ | FOR_HOSTING/ |
|---------|-------|--------------|
| Purpose | Backend API only | Full stack (frontend + backend) |
| Frontend | Separate (Netlify) | Included (served by Flask) |
| Port | 5000 (default) | 5003 (via script) |
| Files | Separate HTML files | All in one folder |

## Troubleshooting

### Port Already in Use
If port 5003 is taken, edit `start-local-5003.sh` and change `PORT=5003` to another port.

### "Module not found" errors
```bash
cd "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING"
pip3 install -r ../requirements.txt
```

### Shopify API not configured
The server will still work without Shopify configured. Cart creation will fail but file generation works fine.

## Development Tips

1. **Auto-reload enabled**: Flask watches for file changes (when `FLASK_ENV=development`)
2. **Check console**: Server logs show all requests and errors
3. **Test APIs first**: Use curl to test endpoints before frontend integration
4. **Static files**: Assets folder contains CSS/JS/images served by Flask

## Next Steps

1. ✅ Start server with `./start-local-5003.sh`
2. ✅ Open http://localhost:5003/mobile/ in browser
3. ✅ Test cart creation with curl or browser console
4. ✅ Make your finishing touches to the frontend
5. ✅ Test full flow: upload image → customize → checkout

---

**Ready to rock! 🎸** Server runs both frontend and backend on one port for easy local development.

