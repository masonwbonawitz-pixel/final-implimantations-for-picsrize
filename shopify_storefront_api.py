"""
Shopify Storefront API Wrapper
Handles cart creation and checkout URL generation via Storefront API
"""
import os
import json
import requests
from typing import Optional, Dict, List, Any


class ShopifyStorefrontAPI:
    """Wrapper for Shopify Storefront API operations (GraphQL)"""
    
    def __init__(self):
        self.shop_domain = os.getenv('SHOPIFY_SHOP_DOMAIN')  # e.g., 'yourstore.myshopify.com'
        self.storefront_token = os.getenv('SHOPIFY_STOREFRONT_TOKEN')
        self.api_version = os.getenv('SHOPIFY_STOREFRONT_API_VERSION', '2025-01')
        
        if not all([self.shop_domain, self.storefront_token]):
            print("⚠️  Shopify Storefront credentials not configured")
            print("   Set SHOPIFY_SHOP_DOMAIN and SHOPIFY_STOREFRONT_TOKEN in environment")
            self.initialized = False
            return
        
        self.initialized = True
        self.graphql_url = f"https://{self.shop_domain}/api/{self.api_version}/graphql.json"
        print(f"✅ Shopify Storefront API initialized (version {self.api_version})")
    
    def is_configured(self) -> bool:
        """Check if Storefront API is properly configured"""
        return self.initialized
    
    def _execute_graphql(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
        """
        Execute a GraphQL query against Storefront API
        Args:
            query: GraphQL query string
            variables: Optional variables for the query
        Returns:
            Response data or None on error
        """
        if not self.is_configured():
            print("❌ Storefront API not configured")
            return None
        
        headers = {
            'Content-Type': 'application/json',
            'X-Shopify-Storefront-Access-Token': self.storefront_token
        }
        
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            response = requests.post(
                self.graphql_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for GraphQL errors
                if 'errors' in data:
                    print(f"❌ GraphQL errors: {json.dumps(data['errors'], indent=2)}")
                    return None
                
                return data.get('data')
            else:
                print(f"❌ Storefront API request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error executing GraphQL query: {e}")
            return None
    
    def create_cart(
        self,
        variant_gid: str,
        quantity: int = 1,
        customization_id: Optional[str] = None,
        attributes: Optional[List[Dict[str, str]]] = None
    ) -> Optional[Dict]:
        """
        Create a cart with a single line item using cartCreate mutation
        
        Args:
            variant_gid: Shopify variant GID (e.g., 'gid://shopify/ProductVariant/12345')
            quantity: Quantity to add
            customization_id: Your internal customization ID to store as attribute
            attributes: Additional cart/line attributes as list of {key, value} dicts
        
        Returns:
            Dict with 'cart_id', 'checkout_url', and 'cart' object, or None on error
        """
        if not self.is_configured():
            return None
        
        # Build line item attributes
        line_attributes = []
        if customization_id:
            line_attributes.append({
                'key': 'customization_id',
                'value': customization_id
            })
        
        if attributes:
            line_attributes.extend(attributes)
        
        # Build GraphQL mutation
        mutation = """
        mutation cartCreate($input: CartInput!) {
          cartCreate(input: $input) {
            cart {
              id
              checkoutUrl
              createdAt
              updatedAt
              lines(first: 10) {
                edges {
                  node {
                    id
                    quantity
                    merchandise {
                      ... on ProductVariant {
                        id
                        title
                        price {
                          amount
                          currencyCode
                        }
                      }
                    }
                    attributes {
                      key
                      value
                    }
                  }
                }
              }
              attributes {
                key
                value
              }
              cost {
                totalAmount {
                  amount
                  currencyCode
                }
                subtotalAmount {
                  amount
                  currencyCode
                }
              }
            }
            userErrors {
              field
              message
            }
          }
        }
        """
        
        # Build variables
        line_items = [{
            'merchandiseId': variant_gid,
            'quantity': quantity
        }]
        
        if line_attributes:
            line_items[0]['attributes'] = line_attributes
        
        variables = {
            'input': {
                'lines': line_items
            }
        }
        
        # Execute mutation
        print(f"🛒 Creating cart with variant {variant_gid} (qty: {quantity})")
        if customization_id:
            print(f"   Customization ID: {customization_id}")
        
        data = self._execute_graphql(mutation, variables)
        
        if not data:
            return None
        
        cart_data = data.get('cartCreate', {})
        user_errors = cart_data.get('userErrors', [])
        
        if user_errors:
            print(f"❌ Cart creation errors:")
            for error in user_errors:
                print(f"   - {error.get('field')}: {error.get('message')}")
            return None
        
        cart = cart_data.get('cart')
        if not cart:
            print("❌ No cart returned from cartCreate")
            return None
        
        cart_id = cart.get('id')
        checkout_url = cart.get('checkoutUrl')
        
        print(f"✅ Cart created successfully!")
        print(f"   Cart ID: {cart_id}")
        print(f"   Checkout URL: {checkout_url}")
        
        return {
            'cart_id': cart_id,
            'checkout_url': checkout_url,
            'cart': cart
        }
    
    def add_cart_lines(
        self,
        cart_id: str,
        variant_gid: str,
        quantity: int = 1,
        attributes: Optional[List[Dict[str, str]]] = None
    ) -> Optional[Dict]:
        """
        Add line items to an existing cart using cartLinesAdd mutation
        
        Args:
            cart_id: Cart GID from cartCreate
            variant_gid: Shopify variant GID to add
            quantity: Quantity to add
            attributes: Line attributes as list of {key, value} dicts
        
        Returns:
            Updated cart dict or None on error
        """
        if not self.is_configured():
            return None
        
        mutation = """
        mutation cartLinesAdd($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
              id
              checkoutUrl
              lines(first: 50) {
                edges {
                  node {
                    id
                    quantity
                    merchandise {
                      ... on ProductVariant {
                        id
                        title
                      }
                    }
                  }
                }
              }
              cost {
                totalAmount {
                  amount
                  currencyCode
                }
              }
            }
            userErrors {
              field
              message
            }
          }
        }
        """
        
        line_item = {
            'merchandiseId': variant_gid,
            'quantity': quantity
        }
        
        if attributes:
            line_item['attributes'] = attributes
        
        variables = {
            'cartId': cart_id,
            'lines': [line_item]
        }
        
        print(f"🛒 Adding variant {variant_gid} to cart {cart_id}")
        
        data = self._execute_graphql(mutation, variables)
        
        if not data:
            return None
        
        cart_data = data.get('cartLinesAdd', {})
        user_errors = cart_data.get('userErrors', [])
        
        if user_errors:
            print(f"❌ Error adding to cart:")
            for error in user_errors:
                print(f"   - {error.get('field')}: {error.get('message')}")
            return None
        
        cart = cart_data.get('cart')
        print(f"✅ Line added to cart successfully")
        
        return cart
    
    def get_cart(self, cart_id: str) -> Optional[Dict]:
        """
        Retrieve cart details by cart ID
        
        Args:
            cart_id: Cart GID
        
        Returns:
            Cart object or None
        """
        if not self.is_configured():
            return None
        
        query = """
        query getCart($cartId: ID!) {
          cart(id: $cartId) {
            id
            checkoutUrl
            createdAt
            updatedAt
            lines(first: 50) {
              edges {
                node {
                  id
                  quantity
                  merchandise {
                    ... on ProductVariant {
                      id
                      title
                      price {
                        amount
                        currencyCode
                      }
                    }
                  }
                  attributes {
                    key
                    value
                  }
                }
              }
            }
            cost {
              totalAmount {
                amount
                currencyCode
              }
              subtotalAmount {
                amount
                currencyCode
              }
            }
          }
        }
        """
        
        variables = {'cartId': cart_id}
        
        data = self._execute_graphql(query, variables)
        
        if data and 'cart' in data:
            return data['cart']
        
        return None


# Global instance
_storefront_api = None

def get_storefront_api() -> ShopifyStorefrontAPI:
    """Get or create Storefront API instance"""
    global _storefront_api
    if _storefront_api is None:
        _storefront_api = ShopifyStorefrontAPI()
    return _storefront_api

