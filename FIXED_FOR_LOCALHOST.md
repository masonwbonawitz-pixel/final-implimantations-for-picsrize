# ✅ FIXED: All Files Now Connect to localhost:5003

## What Was Fixed

All frontend files were pointing to the wrong ports. Now they all use **port 5003** for local development.

### Files Updated:
1. ✅ **mobile/index.html** - Changed `localhost:5001` → `localhost:5003`
2. ✅ **desktop.html** - Changed `localhost:5001` → `localhost:5003`
3. ✅ **admin.html** - Changed `localhost:5002` → `localhost:5003`

---

## How to Test

### 1. Start the Server

```bash
cd "/Users/masonbonawitz/Desktop/Cursor Code for Rize albums/5001/FOR_HOSTING"
./start-local-5003.sh
```

You should see:
```
🚀 Starting FULL STACK server on localhost:5003...

📂 Main app: http://localhost:5003
🖥️  Desktop: http://localhost:5003/desktop.html
📱 Mobile: http://localhost:5003/mobile/
...
```

### 2. Test Each Page

Open each URL and check browser console (F12):

#### Mobile: http://localhost:5003/mobile/

**You should see:**
```
🔗 API Call: http://localhost:5003/api/prices?t=...
🔗 API Call: http://localhost:5003/api/content?t=...
🔗 API Call: http://localhost:5003/api/images?t=...
```

#### Desktop: http://localhost:5003/desktop.html

**You should see:**
```
🔗 API Call: http://localhost:5003/api/prices
🔗 API Call: http://localhost:5003/api/content
🔗 API Call: http://localhost:5003/api/images
```

#### Admin: http://localhost:5003/admin.html

**You should see:**
```
Backend URL: http://localhost:5003
Loading prices...
Loading content...
```

---

## What Should Work Now

### ✅ Prices Load
- Mobile and desktop show correct prices from `/api/prices`
- Admin panel can edit prices

### ✅ Images Load
- Stand and mounting dots images appear
- Admin panel can upload new images

### ✅ Content Updates
- All text/labels load from `/api/content`
- Admin panel can edit text

### ✅ Cart Creation (NEW)
- Checkout button creates Shopify cart
- Returns checkout URL
- Test at: http://localhost:5003/test-cart.html

---

## Troubleshooting

### Prices/Images Still Not Loading?

**Check browser console (F12):**

❌ **If you see errors like:**
```
Failed to fetch http://localhost:5003/api/prices
```

**Solution:**
1. Make sure server is running: `./start-local-5003.sh`
2. Verify server shows: `Running on http://0.0.0.0:5003`
3. Check terminal for errors

---

### Clear Browser Cache

Sometimes old settings get cached. Try:

**Option 1: Hard Refresh**
- Mac: `Cmd + Shift + R`
- Windows: `Ctrl + Shift + R`

**Option 2: Clear Cache**
1. Open DevTools (F12)
2. Right-click refresh button
3. Select "Empty Cache and Hard Reload"

---

### Still Not Working?

**Check Server Terminal Output:**

You should see API calls like:
```
🔗 API Call: /api/prices
📊 Returning prices: {'48x48': 99.99, ...}

🔗 API Call: /api/content  
📝 Returning content: {title: '...', ...}

🔗 API Call: /api/images
🖼️  Returning images: {stand: '/product_images/...', ...}
```

If you DON'T see these, the frontend isn't calling the API.

**Quick Test:**
```bash
# Should return JSON
curl http://localhost:5003/api/prices

# Should return JSON
curl http://localhost:5003/api/content

# Should return JSON
curl http://localhost:5003/api/images
```

---

## Test Everything is Connected

### Open Browser Console (F12) and Run:

```javascript
// Test prices
fetch('http://localhost:5003/api/prices')
  .then(r => r.json())
  .then(d => console.log('Prices:', d));

// Test content
fetch('http://localhost:5003/api/content')
  .then(r => r.json())
  .then(d => console.log('Content:', d));

// Test images
fetch('http://localhost:5003/api/images')
  .then(r => r.json())
  .then(d => console.log('Images:', d));
```

**Expected Output:**
```
Prices: {48x48: 99.99, 75x75: 149.99, ...}
Content: {title: "3D Album Cover...", ...}
Images: {stand: "/product_images/...", ...}
```

---

## Summary

✅ **Mobile frontend** → `http://localhost:5003`  
✅ **Desktop frontend** → `http://localhost:5003`  
✅ **Admin panel** → `http://localhost:5003`  
✅ **All API calls** → `http://localhost:5003/api/*`

**Everything now points to the same server on port 5003!**

---

## Next Steps

1. ✅ Server running? Check!
2. ✅ Pages loading? Check!
3. ✅ Prices/images/content loading? Check!
4. 🎨 **Now you can make your finishing touches!**

Just edit the HTML files and refresh the browser to see changes instantly.

---

**All fixed and ready to go! 🚀**

