# 🚀 START HERE - Port 5003 Full Stack

## Run the Server

```bash
cd "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING"
./start-local-5003.sh
```

That's it! The server runs both frontend and backend on **localhost:5003**

---

## 🔗 All Your URLs (Port 5003)

### 📱 Main Application
- **Home (auto-detect mobile/desktop)**: http://localhost:5003/
- **Mobile Version**: http://localhost:5003/mobile/
- **Desktop Version**: http://localhost:5003/desktop.html

### 🧪 Testing
- **Test Cart Creation**: http://localhost:5003/test-cart.html
- **Admin Panel**: http://localhost:5003/admin.html

### 🔧 API Endpoints
```bash
# Create Shopify cart
curl -X POST http://localhost:5003/api/create-cart \
  -H "Content-Type: application/json" \
  -d '{"size":75,"quantity":1,"customization_id":"test-123","addons":{"stand":true,"mounting_dots":false}}'

# Get prices
curl http://localhost:5003/api/prices

# Get images
curl http://localhost:5003/api/images

# Get content
curl http://localhost:5003/api/content

# Get Shopify variants
curl http://localhost:5003/api/shopify/variants
```

---

## 🎯 Quick Test

### Option 1: Use Test Page (Easiest)
1. Start server: `./start-local-5003.sh`
2. Open: http://localhost:5003/test-cart.html
3. Click "Create Cart" button
4. See results instantly

### Option 2: Browser Console
1. Open: http://localhost:5003/mobile/
2. Open browser console (F12)
3. Paste this:

```javascript
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
    console.log('Cart created:', data);
    if (data.checkout_url) {
        console.log('Checkout URL:', data.checkout_url);
    }
}
testCart();
```

### Option 3: Terminal curl
```bash
curl -X POST http://localhost:5003/api/create-cart \
  -H "Content-Type: application/json" \
  -d '{
    "size": 75,
    "quantity": 1,
    "customization_id": "test-curl-123",
    "addons": {
      "stand": true,
      "mounting_dots": false
    }
  }'
```

---

## 📂 What's Running?

```
┌─────────────────────────────────────────────────┐
│  Flask Server (localhost:5003)                  │
│  ├── Frontend: Serves HTML/CSS/JS               │
│  │   ├── /mobile/index.html                     │
│  │   ├── /desktop.html                          │
│  │   └── /admin.html                            │
│  └── Backend API:                               │
│      ├── POST /api/create-cart (NEW)            │
│      ├── POST /upload-for-checkout              │
│      ├── POST /generate-obj                     │
│      ├── GET /api/prices                        │
│      ├── GET /api/images                        │
│      └── ... and more                           │
└─────────────────────────────────────────────────┘
```

---

## 🎨 Making Your Finishing Touches

### Edit Frontend Files
```bash
# Mobile version
open -a "Cursor" "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING/mobile/index.html"

# Desktop version
open -a "Cursor" "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING/desktop.html"
```

### Test Changes Live
1. Edit HTML/CSS/JS in Cursor
2. Save file (Cmd+S)
3. Refresh browser (Cmd+R)
4. Changes appear instantly!

Flask auto-reloads when files change (because `FLASK_ENV=development`)

---

## 🛒 Integrate Cart Creation in Your Frontend

Add this to your checkout button handler:

```javascript
async function handleCheckout() {
    try {
        // 1. Save customization (you already have this)
        const customizationId = await saveCustomization();
        
        // 2. Create Shopify cart (NEW)
        const response = await fetch('/api/create-cart', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                size: currentGridSize, // 48, 75, or 96
                quantity: 1,
                customization_id: customizationId,
                addons: {
                    stand: document.getElementById('standCheckbox').checked,
                    mounting_dots: document.getElementById('mountingCheckbox').checked
                }
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // 3. Redirect to Shopify checkout
        window.location.href = data.checkout_url;
        
    } catch (error) {
        console.error('Checkout error:', error);
        alert('Failed to create cart. Please try again.');
    }
}
```

---

## 🔍 Debugging

### Server not starting?
```bash
# Check if port 5003 is already in use
lsof -i :5003

# Kill process on port 5003 if needed
kill -9 $(lsof -t -i:5003)

# Try again
./start-local-5003.sh
```

### Can't connect to server?
1. Make sure script is running (you should see logs in terminal)
2. Check: http://localhost:5003/api/prices
3. If you see JSON data, server is working!

### Cart creation fails?
- Check server logs in terminal
- Try test page: http://localhost:5003/test-cart.html
- Verify Shopify credentials (optional for testing)

---

## ✅ You're All Set!

**Server is running on:** http://localhost:5003

**Test cart creation:** http://localhost:5003/test-cart.html

**Edit your frontend:** Mobile: `mobile/index.html` | Desktop: `desktop.html`

**See changes:** Just refresh browser after saving!

---

## 📝 Notes

- Frontend and backend run together on one port (5003)
- No need for separate Netlify - everything is here
- Auto-reload enabled - just save and refresh
- Shopify cart creation integrated and ready
- All your files are in this FOR_HOSTING folder

**Happy coding! 🎉**

