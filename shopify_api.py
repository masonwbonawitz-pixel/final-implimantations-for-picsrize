"""
Shopify API Wrapper
Handles all Shopify Admin API interactions
"""
import os
import hmac
import hashlib
import json
from typing import Optional, Dict, List, Any

try:
    # Try importing shopify (PyShopify package)
    try:
        import shopify
    except ImportError:
        # Try alternative import for pyshopify package
        from pyshopify import shopify
    SHOPIFY_AVAILABLE = True
except ImportError:
    SHOPIFY_AVAILABLE = False
    print("⚠️  PyShopify not installed. Install with: pip3 install PyShopify")


class ShopifyAPI:
    """Wrapper for Shopify Admin API operations"""
    
    def __init__(self):
        self.store_url = os.getenv('SHOPIFY_STORE_URL')
        self.api_key = os.getenv('SHOPIFY_API_KEY')
        self.api_secret = os.getenv('SHOPIFY_API_SECRET')
        self.api_token = os.getenv('SHOPIFY_API_TOKEN')  # Admin API access token
        self.api_version = os.getenv('SHOPIFY_API_VERSION', '2024-01')
        
        # Check if we have either API key/secret or API token
        has_credentials = (self.store_url and (self.api_key and self.api_secret)) or (self.store_url and self.api_token)
        
        if not has_credentials:
            print("⚠️  Shopify credentials not configured")
            self.initialized = False
            return
        
        self.initialized = True
        
        if SHOPIFY_AVAILABLE:
            try:
                # Initialize Shopify session (works with PyShopify 0.9.x)
                shopify.ShopifyResource.set_site(f"https://{self.api_key}:{self.api_secret}@{self.store_url}/admin/api/{self.api_version}")
                print("✅ Shopify API initialized")
            except Exception as e:
                print(f"❌ Error initializing Shopify API: {e}")
                # Try alternative initialization for older versions
                try:
                    shopify.Session.setup(api_key=self.api_key, secret=self.api_secret)
                    session = shopify.Session(f"{self.store_url}.myshopify.com", self.api_version)
                    shopify.ShopifyResource.activate_session(session)
                    print("✅ Shopify API initialized (alternative method)")
                except Exception as e2:
                    print(f"❌ Alternative initialization also failed: {e2}")
                    self.initialized = False
    
    def is_configured(self) -> bool:
        """Check if Shopify API is properly configured"""
        return self.initialized and SHOPIFY_AVAILABLE
    
    def verify_webhook_signature(self, data: bytes, hmac_header: str) -> bool:
        """
        Verify Shopify webhook signature
        Args:
            data: Raw request body as bytes
            hmac_header: X-Shopify-Hmac-Sha256 header value
        Returns:
            True if signature is valid
        """
        if not self.api_secret:
            return False
        
        calculated_hmac = hmac.new(
            self.api_secret.encode('utf-8'),
            data,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(calculated_hmac, hmac_header)
    
    def upload_file_to_shopify(self, file_path: str, filename: str) -> Optional[str]:
        """
        Upload a file to Shopify Files API
        Note: PyShopify 0.9.x doesn't support File API directly
        This is a placeholder - you may need to use REST API directly
        Args:
            file_path: Local path to file
            filename: Name for the file in Shopify
        Returns:
            File URL if successful, None otherwise
        """
        if not self.is_configured():
            print("⚠️  Shopify API not configured")
            return None
        
        try:
            # PyShopify 0.9.x doesn't have File API, use REST API directly
            import requests
            import base64
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Encode file as base64
            file_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Upload via REST API
            url = f"https://{self.api_key}:{self.api_secret}@{self.store_url}/admin/api/{self.api_version}/files.json"
            
            payload = {
                'file': {
                    'filename': filename,
                    'contents': file_base64
                }
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 201:
                file_data = response.json().get('file', {})
                file_url = file_data.get('url')
                if file_url:
                    print(f"✅ File uploaded to Shopify: {file_url}")
                    return file_url
                else:
                    print("❌ No URL in response")
                    return None
            else:
                print(f"❌ Failed to upload file: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error uploading file to Shopify: {e}")
            return None
    
    def create_draft_order(self, order_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Create a draft order in Shopify
        Args:
            order_data: Order data including line_items, customer, etc.
        Returns:
            Draft order object if successful, None otherwise
        """
        if not self.is_configured():
            print("⚠️  Shopify API not configured")
            return None
        
        try:
            draft_order = shopify.DraftOrder.create(order_data)
            
            if draft_order:
                print(f"✅ Draft order created: {draft_order.id}")
                return draft_order
            else:
                print("❌ Failed to create draft order")
                return None
                
        except Exception as e:
            print(f"❌ Error creating draft order: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order by ID
        Args:
            order_id: Shopify order ID
        Returns:
            Order object if found, None otherwise
        """
        if not self.is_configured():
            return None
        
        try:
            order = shopify.Order.find(order_id)
            return order
        except Exception as e:
            print(f"❌ Error getting order {order_id}: {e}")
            return None
    
    def get_orders(self, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """
        Get list of orders
        Args:
            limit: Maximum number of orders to return
            status: Filter by status (e.g., 'any', 'open', 'closed')
        Returns:
            List of order objects
        """
        if not self.is_configured():
            return []
        
        try:
            if status:
                orders = shopify.Order.find(limit=limit, status=status)
            else:
                orders = shopify.Order.find(limit=limit)
            
            return list(orders) if orders else []
        except Exception as e:
            print(f"❌ Error getting orders: {e}")
            return []
    
    def add_metafield_to_order(self, order_id: str, namespace: str, key: str, value: str, type: str = 'single_line_text_field') -> bool:
        """
        Add metafield to order
        Args:
            order_id: Shopify order ID
            namespace: Metafield namespace
            key: Metafield key
            value: Metafield value
            type: Metafield type
        Returns:
            True if successful
        """
        if not self.is_configured():
            return False
        
        try:
            order = shopify.Order.find(order_id)
            if not order:
                return False
            
            metafield = shopify.Metafield({
                'namespace': namespace,
                'key': key,
                'value': value,
                'type': type,
                'owner_resource': 'order',
                'owner_id': order_id
            })
            
            metafield.save()
            print(f"✅ Metafield added to order {order_id}")
            return True
        except Exception as e:
            print(f"❌ Error adding metafield: {e}")
            return False
    
    def attach_files_to_order(self, order_id: str, file_urls: List[str]) -> bool:
        """
        Attach files to order as fulfillment attachments
        Args:
            order_id: Shopify order ID
            file_urls: List of file URLs
        Returns:
            True if successful
        """
        if not self.is_configured():
            return False
        
        try:
            order = shopify.Order.find(order_id)
            if not order:
                return False
            
            # Create fulfillment with attachments
            # Note: tracking_urls may not be supported in older PyShopify
            fulfillment_data = {
                'order_id': order_id,
                'status': 'success',
                'tracking_company': 'Digital Download',
                'tracking_number': 'N/A',
                'notify_customer': True,
                'line_items': [{'id': item.id} for item in order.line_items]
            }
            
            # Try to add tracking_urls if supported
            try:
                fulfillment_data['tracking_urls'] = file_urls
            except:
                pass
            
            fulfillment = shopify.Fulfillment.create(fulfillment_data)
            
            if fulfillment:
                print(f"✅ Files attached to order {order_id}")
                # Also add note with download links
                note = f"Download links:\n" + "\n".join(file_urls)
                order.note = (order.note or "") + "\n\n" + note
                order.save()
                return True
            else:
                print("❌ Failed to create fulfillment")
                return False
        except Exception as e:
            print(f"❌ Error attaching files to order: {e}")
            return False
    
    def get_product_variants(self, product_id: Optional[str] = None) -> List[Dict]:
        """
        Get product variants
        Args:
            product_id: Optional product ID to filter
        Returns:
            List of variant objects
        """
        if not self.is_configured():
            return []
        
        try:
            if product_id:
                product = shopify.Product.find(product_id)
                return list(product.variants) if product else []
            else:
                # Get all products and their variants
                products = shopify.Product.find(limit=250)
                variants = []
                for product in products:
                    variants.extend(product.variants)
                return variants
        except Exception as e:
            print(f"❌ Error getting variants: {e}")
            return []
    
    def get_variant_prices(self, variant_ids: Dict[str, str]) -> Dict[str, float]:
        """
        Get prices for multiple variants by their IDs using REST API
        Args:
            variant_ids: Dictionary mapping product keys to variant IDs
                       e.g., {'48x48': '10470738559281', '75x75': '10470738952497', ...}
        Returns:
            Dictionary mapping product keys to prices in dollars
            e.g., {'48x48': 29.99, '75x75': 48.99, ...}
        """
        prices = {}
        
        # Use REST API directly to fetch variant prices
        import requests
        
        # Get API token (prefer SHOPIFY_API_TOKEN, fallback to API_KEY)
        api_token = self.api_token or self.api_key
        
        if not api_token or not self.store_url:
            print("⚠️  Shopify API token or store URL not configured")
            return prices
        
        # Ensure store_url doesn't have protocol
        store_url_clean = self.store_url.replace('https://', '').replace('http://', '').split('/')[0]
        
        # Fetch each variant by ID
        for key, variant_id in variant_ids.items():
            if not variant_id:
                continue
                
            try:
                # Use REST API to get variant by ID
                url = f"https://{store_url_clean}/admin/api/{self.api_version}/variants/{variant_id}.json"
                
                headers = {
                    'X-Shopify-Access-Token': api_token,
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    variant_data = response.json().get('variant', {})
                    price_str = variant_data.get('price', '0')
                    # Shopify prices are stored as strings, convert to float dollars
                    try:
                        # Price is already in dollars as a string, or might be in cents
                        # Try parsing as float first (if already dollars)
                        price = float(price_str)
                        # If price seems like it's in cents (e.g., > 10000), divide by 100
                        if price > 10000:
                            price = price / 100.0
                        prices[key] = price
                        print(f"✅ Fetched price for {key}: ${price:.2f} (variant {variant_id})")
                    except ValueError:
                        print(f"⚠️  Invalid price format for {key}: {price_str}")
                else:
                    error_text = response.text[:200] if hasattr(response, 'text') else 'Unknown error'
                    print(f"⚠️  Failed to fetch variant {variant_id} for {key}: {response.status_code} - {error_text}")
                    
            except Exception as e:
                print(f"❌ Error fetching price for {key} (variant {variant_id}): {e}")
                import traceback
                traceback.print_exc()
        
        return prices


# Global instance
_shopify_api = None

def get_shopify_api() -> ShopifyAPI:
    """Get or create Shopify API instance"""
    global _shopify_api
    if _shopify_api is None:
        _shopify_api = ShopifyAPI()
    return _shopify_api

