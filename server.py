"""
Flask backend API for 3MF color mapping
Exposes a /generate endpoint that accepts STL + PNG and returns 3MF
"""
import io
import json
import os
import time
import uuid
from typing import List, Tuple, Optional

import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import trimesh

# Try to load pillow-heif for HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT_AVAILABLE = True
    print("✅ pillow-heif loaded - HEIC/HEIF conversion supported")
except ImportError:
    HEIF_SUPPORT_AVAILABLE = False
    print("⚠️  pillow-heif not installed - HEIC/HEIF conversion will fallback to client-side")
except Exception as e:
    HEIF_SUPPORT_AVAILABLE = False
    print(f"⚠️  pillow-heif failed to load: {e} - HEIC/HEIF conversion will fallback to client-side")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# Import Shopify API wrapper
try:
    from shopify_api import get_shopify_api
    SHOPIFY_API_AVAILABLE = True
except ImportError:
    SHOPIFY_API_AVAILABLE = False
    print("⚠️  shopify_api.py not found")

# Import webhook handlers
try:
    from webhook_handlers import handle_order_create, handle_order_paid
    from shopify_api import get_shopify_api as get_shopify_api_for_webhook
    WEBHOOK_HANDLERS_AVAILABLE = True
except ImportError:
    WEBHOOK_HANDLERS_AVAILABLE = False
    print("⚠️  webhook_handlers.py not found")

# Import Shopify Storefront API wrapper
try:
    from shopify_storefront_api import get_storefront_api
    SHOPIFY_STOREFRONT_API_AVAILABLE = True
except ImportError:
    SHOPIFY_STOREFRONT_API_AVAILABLE = False
    print("⚠️  shopify_storefront_api.py not found")

# Try both bindings; some environments publish lib3mf as 'lib3mf', others as 'py3mf'
_three_mf = None
_three_mf_error = None

# On macOS, sometimes we need to set library path before importing
import sys
import os
if sys.platform == 'darwin':
    # Try to find lib3mf library path and set DYLD_LIBRARY_PATH
    try:
        import site
        # Check both system and user site-packages
        all_site_packages = site.getsitepackages() + [site.getusersitepackages()] if hasattr(site, 'getusersitepackages') else site.getsitepackages()
        
        for sp in all_site_packages:
            lib3mf_path = os.path.join(sp, 'lib3mf')
            if os.path.exists(lib3mf_path):
                dylib_path = os.path.join(lib3mf_path, 'lib3mf.dylib')
                if os.path.exists(dylib_path):
                    # Set DYLD_LIBRARY_PATH to the directory containing the .dylib
                    if 'DYLD_LIBRARY_PATH' not in os.environ:
                        os.environ['DYLD_LIBRARY_PATH'] = lib3mf_path
                    elif lib3mf_path not in os.environ['DYLD_LIBRARY_PATH']:
                        os.environ['DYLD_LIBRARY_PATH'] = f"{os.environ['DYLD_LIBRARY_PATH']}:{lib3mf_path}"
                    print(f"✅ Set DYLD_LIBRARY_PATH to {lib3mf_path} for lib3mf")
                    break
    except Exception as e:
        pass

for _modname in ("lib3mf", "py3mf"):
    try:
        _three_mf = __import__(_modname)
        # Test if it actually works by trying to create a wrapper
        try:
            if hasattr(_three_mf, "get_wrapper"):
                wrapper = _three_mf.get_wrapper()
            elif hasattr(_three_mf, "Wrapper"):
                wrapper = _three_mf.Wrapper()
            else:
                raise AttributeError("No wrapper method found")
            # Test creating a model to ensure it really works
            if hasattr(wrapper, "CreateModel"):
                wrapper.CreateModel()
            elif hasattr(_three_mf, "CreateModel"):
                _three_mf.CreateModel()
            print(f"✅ {_modname} library loaded and working")
            break
        except Exception as e:
            _three_mf_error = f"{_modname} loaded but failed to initialize: {e}"
            _three_mf = None
            continue
    except Exception as e:
        _three_mf_error = str(e)
        continue


FOUR_COLORS_RGB: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (85, 85, 85),
    (170, 170, 170),
    (255, 255, 255),
]


def quantize_to_four_colors(rgb_colors: np.ndarray) -> np.ndarray:
    """
    Quantize RGB colors to the nearest of the 4 allowed colors.
    Uses Euclidean distance in RGB space to find the closest match.
    CRITICAL: Ensures dark gray (85,85,85) is preserved correctly.
    For grayscale colors (R≈G≈B), uses a more lenient threshold to preserve dark gray.
    Args:
        rgb_colors: Array of shape (N, 3) with RGB values 0-255
    Returns:
        Quantized colors array of shape (N, 3) with only the 4 allowed colors
    """
    # Convert to numpy array for easier computation
    colors_array = np.array(FOUR_COLORS_RGB, dtype=np.float32)  # Shape: (4, 3)
    
    # Reshape input to (N, 1, 3) for broadcasting
    rgb_colors_float = rgb_colors.astype(np.float32)
    rgb_colors_reshaped = rgb_colors_float[:, np.newaxis, :]  # (N, 1, 3)
    
    # PRE-PROCESSING: For grayscale colors (R≈G≈B), use threshold-based mapping
    # This ensures black stays black and dark gray stays dark gray
    # CRITICAL: Use proper thresholds to prevent black from mapping to dark gray
    is_grayscale = (np.abs(rgb_colors_float[:, 0] - rgb_colors_float[:, 1]) < 5) & \
                   (np.abs(rgb_colors_float[:, 1] - rgb_colors_float[:, 2]) < 5)
    
    # For grayscale colors, use threshold-based mapping with proper boundaries
    if np.any(is_grayscale):
        gray_values = rgb_colors_float[is_grayscale].mean(axis=1)  # Average of R,G,B
        grayscale_indices = np.where(is_grayscale)[0]
        
        # Use threshold-based mapping to ensure proper color boundaries:
        # Black: 0-42 (midpoint between 0 and 85)
        # Dark Gray: 43-127 (midpoint between 85 and 170)
        # Light Gray: 128-212 (midpoint between 170 and 255)
        # White: 213-255
        for i, gray_val in enumerate(gray_values):
            orig_idx = grayscale_indices[i]
            if gray_val <= 42.5:
                # Map to black (0,0,0)
                rgb_colors_float[orig_idx] = colors_array[0]
            elif gray_val <= 127.5:
                # Map to dark gray (85,85,85)
                rgb_colors_float[orig_idx] = colors_array[1]
            elif gray_val <= 212.5:
                # Map to light gray (170,170,170)
                rgb_colors_float[orig_idx] = colors_array[2]
            else:
                # Map to white (255,255,255)
                rgb_colors_float[orig_idx] = colors_array[3]
    
    # ADDITIONAL SAFEGUARD: Very dark colors (max component < 43) should map to black
    # This prevents near-black colors from being mapped to dark gray
    max_component = np.max(rgb_colors_float, axis=1)
    very_dark_mask = max_component < 43
    if np.any(very_dark_mask):
        rgb_colors_float[very_dark_mask] = colors_array[0]  # Force to black
    
    # Re-reshape after preprocessing
    rgb_colors_reshaped = rgb_colors_float[:, np.newaxis, :]  # (N, 1, 3)
    
    # Calculate Euclidean distance to each of the 4 colors
    # Shape: (N, 4) - distance from each input color to each palette color
    distances = np.sqrt(np.sum((rgb_colors_reshaped - colors_array) ** 2, axis=2))
    
    # Find index of closest color for each input color
    closest_indices = np.argmin(distances, axis=1)
    
    # Map to the closest palette color - ensure EXACT values
    quantized = np.zeros_like(rgb_colors, dtype=np.uint8)
    for i, exact_color in enumerate(FOUR_COLORS_RGB):
        mask = closest_indices == i
        quantized[mask] = np.array(exact_color, dtype=np.uint8)
    
    # Debug: Print color usage statistics with sample colors
    unique_colors_used = set(closest_indices)
    color_names = ["Black (0,0,0)", "Dark Gray (85,85,85)", "Light Gray (170,170,170)", "White (255,255,255)"]
    print(f"📊 Quantization: Using {len(unique_colors_used)} out of 4 colors")
    
    # Show sample of input colors that map to each output color
    for idx in sorted(unique_colors_used):
        mask = closest_indices == idx
        count = np.sum(mask)
        percentage = count / len(closest_indices) * 100
        print(f"   - {color_names[idx]}: {count} triangles ({percentage:.1f}%)")
        
        # Show sample of input colors that mapped to this output color
        if count > 0:
            sample_inputs = rgb_colors[mask][:5]  # First 5 examples
            print(f"      Sample input colors: {sample_inputs.tolist()}")
    
    # Check if all 4 colors are present
    missing_colors = set(range(4)) - unique_colors_used
    if missing_colors:
        print(f"⚠️  Warning: Missing colors: {[color_names[i] for i in missing_colors]}")
        
        # Debug: Check if input colors are close to missing colors
        for missing_idx in missing_colors:
            target_color = np.array(FOUR_COLORS_RGB[missing_idx], dtype=np.float32)
            # Find colors that are close to the missing color
            color_diffs = np.abs(rgb_colors_float - target_color)
            max_diff = np.max(color_diffs, axis=1)
            close_colors = rgb_colors[max_diff < 20]  # Within 20 RGB units
            if len(close_colors) > 0:
                print(f"      Found {len(close_colors)} colors close to {color_names[missing_idx]}:")
                unique_close = np.unique(close_colors, axis=0)[:5]
                print(f"      Sample: {unique_close.tolist()}")
                # Check what they're being mapped to
                close_indices = closest_indices[max_diff < 20]
                mapped_to = [color_names[i] for i in np.unique(close_indices)]
                print(f"      These are being mapped to: {mapped_to}")
    
    return quantized


def require_three_mf():
    if _three_mf is None:
        raise RuntimeError(
            "Neither lib3mf nor py3mf is available. Install one: 'pip install lib3mf' or 'pip install py3mf'."
        )
    return _three_mf


def _get_wrapper(mf):
    if hasattr(mf, "get_wrapper"):
        return mf.get_wrapper()
    if hasattr(mf, "Wrapper"):
        return mf.Wrapper()
    return mf


def _make_position(mf, x: float, y: float, z: float):
    if hasattr(mf, "Position"):
        try:
            return mf.Position(float(x), float(y), float(z))
        except Exception:
            try:
                pos = mf.Position()
                if hasattr(pos, "Coordinates"):
                    pos.Coordinates = (float(x), float(y), float(z))
                    return pos
            except Exception:
                pass
    pos = getattr(mf, "Position", object)()
    if hasattr(pos, "X") and hasattr(pos, "Y") and hasattr(pos, "Z"):
        pos.X = float(x)
        pos.Y = float(y)
        pos.Z = float(z)
        return pos
    return (float(x), float(y), float(z))


def _make_triangle(mf, i0: int, i1: int, i2: int):
    if hasattr(mf, "Triangle"):
        try:
            return mf.Triangle(int(i0), int(i1), int(i2))
        except Exception:
            try:
                tri = mf.Triangle()
                if hasattr(tri, "Indices"):
                    tri.Indices = (int(i0), int(i1), int(i2))
                    return tri
            except Exception:
                pass
    tri = getattr(mf, "Triangle", object)()
    if hasattr(tri, "Indices"):
        tri.Indices = (int(i0), int(i1), int(i2))
        return tri
    return (int(i0), int(i1), int(i2))


def _make_color(mf, r: int, g: int, b: int, a: int = 255):
    if hasattr(mf, "Color"):
        try:
            return mf.Color(int(r), int(g), int(b), int(a))
        except Exception:
            pass
        try:
            return mf.Color(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(a) / 255.0)
        except Exception:
            pass
    packed = ((int(a) & 0xFF) << 24) | ((int(r) & 0xFF) << 16) | ((int(g) & 0xFF) << 8) | (int(b) & 0xFF)
    return packed


def _set_mesh_geometry(mf, mesh_obj, vertices, triangles):
    if hasattr(mesh_obj, "SetGeometry"):
        return mesh_obj.SetGeometry(vertices, triangles)
    if hasattr(mesh_obj, "SetGeometry2"):
        return mesh_obj.SetGeometry2(vertices, triangles)
    if hasattr(mesh_obj, "Vertices"):
        mesh_obj.Vertices = vertices
    if hasattr(mesh_obj, "Triangles"):
        mesh_obj.Triangles = triangles


def _add_color_and_get_index(color_group, color):
    if hasattr(color_group, "AddColor"):
        idx = color_group.AddColor(color)
    elif hasattr(color_group, "AddColorRGBa"):
        idx = color_group.AddColorRGBa(color)
    else:
        if not hasattr(color_group, "_colors"):
            color_group._colors = []
        color_group._colors.append(color)
        idx = len(color_group._colors) - 1
    rid = color_group.GetResourceID() if hasattr(color_group, "GetResourceID") else getattr(color_group, "ResourceID", 0)
    return rid, idx


def _set_triangle_color(mf, mesh_obj, tri_index: int, resource_id: int, color_index: int):
    try:
        props = getattr(mf, "TriangleProperties", None)
        if props is not None:
            tp = props()
            if hasattr(tp, "m_ResourceID"):
                tp.m_ResourceID = resource_id
            elif hasattr(tp, "ResourceID"):
                tp.ResourceID = resource_id
            if hasattr(tp, "m_Colors"):
                tp.m_Colors = [color_index, color_index, color_index]
            elif hasattr(tp, "Colors"):
                tp.Colors = [color_index, color_index, color_index]
            elif hasattr(tp, "m_Properties"):
                for k in range(3):
                    if hasattr(tp.m_Properties[k], "m_ColorIndex"):
                        tp.m_Properties[k].m_ColorIndex = color_index
            if hasattr(mesh_obj, "SetTriangleProperties"):
                mesh_obj.SetTriangleProperties(int(tri_index), tp)
                return
    except Exception:
        pass

    try:
        if hasattr(mesh_obj, "SetTriangleProperties"):
            mesh_obj.SetTriangleProperties(int(tri_index), int(resource_id), [int(color_index)] * 3)
            return
    except Exception:
        pass

    try:
        if hasattr(mesh_obj, "SetTriangleProperties"):
            mesh_obj.SetTriangleProperties(int(tri_index), [int(color_index)] * 3, int(resource_id))
            return
    except Exception:
        pass


def load_png(image_bytes: bytes, target_size: Optional[int] = None) -> Image.Image:
    """
    Load PNG image preserving exact color values.
    Uses NEAREST neighbor resizing to prevent color interpolation.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if target_size is not None:
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.NEAREST)
    elif img.size != (75, 75):
        img = img.resize((75, 75), Image.NEAREST)
    
    # Debug: Check if image contains the expected 4-color palette
    arr = np.asarray(img, dtype=np.uint8)
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    print(f"📸 Loaded image: {img.size}, {len(unique_colors)} unique colors")
    
    # Check if we have the expected 4-color palette
    expected_colors = set(tuple(c) for c in FOUR_COLORS_RGB)
    actual_colors = set(tuple(c) for c in unique_colors)
    matching = expected_colors.intersection(actual_colors)
    if len(matching) < len(expected_colors):
        print(f"⚠️  Image has {len(matching)}/{len(expected_colors)} expected colors")
        missing = expected_colors - matching
        print(f"   Missing: {missing}")
        # Show sample of actual colors
        print(f"   Sample actual colors: {list(actual_colors)[:10]}")
    
    return img


def get_png_as_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to RGB numpy array"""
    arr = np.asarray(img.convert('RGB'), dtype=np.uint8)
    return arr


def load_stl_vertices_faces(stl_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type='stl', force='mesh', process=False)
    if mesh.is_empty:
        raise ValueError("Failed to load STL mesh or mesh is empty.")
    
    # Check and repair mesh to ensure it's watertight (manifold)
    print(f"📊 Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"   Watertight: {mesh.is_watertight}, Volume: {mesh.is_volume}")
    
    if not mesh.is_watertight:
        print("⚠️  Mesh is not watertight - attempting repair...")
        
        try:
            # Fill holes in the mesh
            trimesh.repair.fill_holes(mesh)
        except Exception as e:
            print(f"   ⚠️  Could not fill holes: {e}")
        
        try:
            # Fix normals (ensure they all point outward)
            trimesh.repair.fix_normals(mesh)
        except Exception as e:
            print(f"   ⚠️  Could not fix normals: {e}")
        
        try:
            # Remove duplicate faces using correct trimesh API
            mesh.update_faces(mesh.unique_faces())
        except Exception as e:
            print(f"   ⚠️  Could not remove duplicate faces: {e}")
        
        try:
            # Remove degenerate faces (zero area triangles)
            # Filter out faces where all three vertices are the same or form zero area
            original_face_count = len(mesh.faces)
            valid_faces = []
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                # Check if triangle has non-zero area
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                if area > 1e-10:  # Only keep faces with meaningful area
                    valid_faces.append(face)
            if len(valid_faces) < original_face_count:
                mesh.faces = np.array(valid_faces)
                print(f"   ✓ Removed {original_face_count - len(valid_faces)} degenerate faces")
        except Exception as e:
            print(f"   ⚠️  Could not remove degenerate faces: {e}")
        
        try:
            # Merge duplicate vertices
            mesh.merge_vertices()
        except Exception as e:
            print(f"   ⚠️  Could not merge vertices: {e}")
        
        print(f"✅ After repair: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"   Watertight: {mesh.is_watertight}, Volume: {mesh.is_volume}")
        
        if not mesh.is_watertight:
            print("⚠️  Warning: Mesh still has issues after repair, but continuing anyway...")
    else:
        print("✅ Mesh is already watertight!")
    
    # Trimesh already loads as triangles; STL files are triangle meshes
    # Each cube in your grid is made of 12 triangles (2 per face)
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)
    return vertices, faces


def get_triangle_colors_from_image(vertices: np.ndarray, faces: np.ndarray, img_rgb: np.ndarray, grid_size: int = 75) -> np.ndarray:
    """
    Map each triangle to its corresponding pixel color in the image.
    Uses EXACT same logic as frontend applyColorsToMesh function.
    Returns shape: (num_triangles, 3) with RGB values
    """
    # Get actual image dimensions
    img_height, img_width = img_rgb.shape[:2]
    
    # Get triangle vertices
    tri_verts = vertices[faces]
    
    # Compute XY bounds based ONLY on near-horizontal (top) faces (matches frontend exactly)
    minX = np.inf
    minY = np.inf
    maxX = -np.inf
    maxY = -np.inf
    
    # Calculate face normals to find top faces
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    
    ab = v1 - v0
    ac = v2 - v0
    normals = np.cross(ab, ac)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths = np.where(norm_lengths == 0, 1, norm_lengths)
    normals_unit = normals / norm_lengths
    nz_unit = np.abs(normals_unit[:, 2])
    
    # Only use top-ish faces (nz > 0.5) to determine bounds
    top_mask = nz_unit > 0.5
    
    if np.any(top_mask):
        top_tri_verts = tri_verts[top_mask]
        for tri in top_tri_verts:
            minX = min(minX, tri[0, 0], tri[1, 0], tri[2, 0])
            minY = min(minY, tri[0, 1], tri[1, 1], tri[2, 1])
            maxX = max(maxX, tri[0, 0], tri[1, 0], tri[2, 0])
            maxY = max(maxY, tri[0, 1], tri[1, 1], tri[2, 1])
    
    if not (np.isfinite(minX) and np.isfinite(minY)):
        # Fallback to bbox if no top faces detected
        minX = vertices[:, 0].min()
        minY = vertices[:, 1].min()
        maxX = vertices[:, 0].max()
        maxY = vertices[:, 1].max()
    
    sizeX = max(1e-9, maxX - minX)
    sizeY = max(1e-9, maxY - minY)
    
    # NOTE: Frontend does NOT enforce square aspect ratio - it uses sizeX and sizeY directly
    # Matching frontend behavior exactly - do NOT square the aspect ratio
    # This ensures pixel mapping matches exactly what the user sees in the 3D viewer
    
    # Get triangle centroids
    centroids = tri_verts.mean(axis=1)
    
    # Map each triangle centroid to image pixel
    num_faces = len(faces)
    triangle_colors = np.zeros((num_faces, 3), dtype=np.uint8)
    
    # Debug: Log bounds and image dimensions
    print(f"🔍 Color mapping debug:")
    print(f"   Bounds: X=[{minX:.3f}, {maxX:.3f}], Y=[{minY:.3f}, {maxY:.3f}]")
    print(f"   Size: X={sizeX:.3f}, Y={sizeY:.3f}")
    print(f"   Image dimensions: {img_width}×{img_height}")
    print(f"   Grid size: {grid_size}")
    print(f"   Number of triangles: {num_faces}")
    
    # For Normal mode (48), use continuous mapping - each triangle maps individually (no pixel grouping)
    # For other modes, use grid-based mapping with pixel grouping
    is_normal_mode = (grid_size == 48)
    
    if is_normal_mode:
        # Pixel-perfect mapping: each triangle maps to the exact pixel it covers
        # Group triangles by pixel to ensure all triangles mapping to same pixel get exact same color
        from collections import defaultdict
        pixel_triangles = defaultdict(list)
        
        # Map each triangle to its pixel location (pixel-perfect)
        for face_idx in range(num_faces):
            cx = centroids[face_idx, 0]
            cy = centroids[face_idx, 1]
            
            # Normalize to [0, 1] range using square bounding box
            u = max(0.0, min(0.999999, (cx - minX) / sizeX))
            v = max(0.0, min(0.999999, (cy - minY) / sizeY))
            
            # Map to image coordinates - use nearest pixel (pixel-perfect mapping)
            img_x = u * (img_width - 1)
            img_y = (1.0 - v) * (img_height - 1)  # Flip Y: v=0 (top) maps to bottom
            
            # Round to nearest pixel for pixel-perfect mapping
            px = int(np.round(img_x))
            py = int(np.round(img_y))
            
            # Clamp to valid pixel range
            px = max(0, min(img_width - 1, px))
            py = max(0, min(img_height - 1, py))
            
            # Group triangles by pixel
            pixel_triangles[(py, px)].append(face_idx)
        
        # Assign colors: all triangles mapping to the same pixel get the exact same color
        for (py, px), tri_indices in pixel_triangles.items():
            pixel_color = img_rgb[py, px, :]  # Get exact pixel color
            for tri_idx in tri_indices:
                triangle_colors[tri_idx] = pixel_color
        
        # Debug: Log color distribution before quantization
        unique_colors_before = len(set(tuple(c) for c in triangle_colors))
        print(f"   Normal mode: {len(pixel_triangles)} unique pixels mapped, {unique_colors_before} unique colors before quantization")
    else:
        # Grid-based mapping for pixelated modes - group triangles by pixel
        from collections import defaultdict
        pixel_triangles = defaultdict(list)
        
        grid = float(grid_size)
        
        # DIRECT 1:1 MAPPING OVERRIDE - Force perfect grid cell to pixel mapping
        # When grid_size matches image dimensions, directly map grid cells to pixels
        # This bypasses all the complex normalization that causes misalignment
        for face_idx in range(num_faces):
            cx = centroids[face_idx, 0]
            cy = centroids[face_idx, 1]
            
            # Normalize to [0, 1] range using square bounding box
            u = max(0.0, min(0.999999, (cx - minX) / sizeX))
            v = max(0.0, min(0.999999, (cy - minY) / sizeY))
            
            # DIRECT MAPPING: Map normalized coords directly to grid cell index
            # This ensures perfect 1:1 mapping between grid cells and pixels
            cell_x = int(np.floor(u * grid))
            cell_y = int(np.floor(v * grid))
            
            # Clamp to valid grid range
            cell_x = max(0, min(int(grid) - 1, cell_x))
            cell_y = max(0, min(int(grid) - 1, cell_y))
            
            # OVERRIDE: Direct cell-to-pixel mapping (no complex calculations)
            # When grid_size == img_width, grid cell N maps directly to pixel N
            if grid_size == img_width:
                px = cell_x
                py = int(grid) - 1 - cell_y  # Flip Y axis
            else:
                # Fallback: Scale cell index to pixel index
                px = int((cell_x * img_width) / grid)
                py = int(((int(grid) - 1 - cell_y) * img_height) / grid)
            
            # Final clamp to valid pixel range
            px = max(0, min(img_width - 1, px))
            py = max(0, min(img_height - 1, py))
            
            # Group triangles by pixel
            pixel_triangles[(py, px)].append(face_idx)
        
        # Assign colors: all triangles in the same pixel get the exact same color from the image
        # This is the only color assignment - no additional edge checks that could cause duplication
        for (py, px), tri_indices in pixel_triangles.items():
            pixel_color = img_rgb[py, px, :]  # Get color directly from image
            for tri_idx in tri_indices:
                triangle_colors[tri_idx] = pixel_color
        
        # Debug: Log color distribution before quantization
        unique_colors_before = len(set(tuple(c) for c in triangle_colors))
        print(f"   Grid mode: {len(pixel_triangles)} unique pixels mapped, {unique_colors_before} unique colors before quantization")
    
    # Debug: Show sample of assigned colors
    if num_faces > 0:
        sample_colors = triangle_colors[:min(10, num_faces)]
        print(f"   Sample colors (first 10 triangles): {sample_colors.tolist()}")
    
    return triangle_colors


def write_3mf_with_texture(vertices: np.ndarray, faces: np.ndarray, img_rgb: np.ndarray, png_bytes: bytes) -> bytes:
    """
    Write 3MF file with texture mapping (for Normal mode 48x48)
    Uses UV coordinates and embedded PNG texture
    """
    import io
    from PIL import Image
    
    mf = require_three_mf()
    wrapper = _get_wrapper(mf)
    model = wrapper.CreateModel() if hasattr(wrapper, "CreateModel") else mf.CreateModel()

    mesh_obj = model.AddMeshObject()
    if hasattr(mesh_obj, "SetName"):
        mesh_obj.SetName("Textured Mesh")

    # Build geometry
    verts_list = [_make_position(mf, float(x), float(y), float(z)) for x, y, z in vertices.tolist()]
    tris_list = [_make_triangle(mf, int(a), int(b), int(c)) for a, b, c in faces.tolist()]
    _set_mesh_geometry(mf, mesh_obj, verts_list, tris_list)

    # Generate UV coordinates (same logic as OBJ)
    img_height, img_width = img_rgb.shape[:2]
    
    # Compute XY bounds based ONLY on near-horizontal (top) faces
    tri_verts = vertices[faces]
    minX = np.inf
    minY = np.inf
    maxX = -np.inf
    maxY = -np.inf
    
    # Calculate face normals to find top faces
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    
    ab = v1 - v0
    ac = v2 - v0
    normals = np.cross(ab, ac)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths = np.where(norm_lengths == 0, 1, norm_lengths)
    normals_unit = normals / norm_lengths
    nz_unit = np.abs(normals_unit[:, 2])
    
    # Only use top-ish faces (nz > 0.5) to determine bounds
    top_mask = nz_unit > 0.5
    
    if np.any(top_mask):
        top_tri_verts = tri_verts[top_mask]
        for tri in top_tri_verts:
            minX = min(minX, tri[0, 0], tri[1, 0], tri[2, 0])
            minY = min(minY, tri[0, 1], tri[1, 1], tri[2, 1])
            maxX = max(maxX, tri[0, 0], tri[1, 0], tri[2, 0])
            maxY = max(maxY, tri[0, 1], tri[1, 1], tri[2, 1])
    
    if not (np.isfinite(minX) and np.isfinite(minY)):
        # Fallback to bbox if no top faces detected
        minX = vertices[:, 0].min()
        minY = vertices[:, 1].min()
        maxX = vertices[:, 0].max()
        maxY = vertices[:, 1].max()
    
    sizeX = max(1e-9, maxX - minX)
    sizeY = max(1e-9, maxY - minY)
    
    # Ensure square aspect ratio
    maxSize = max(sizeX, sizeY)
    sizeX = maxSize
    sizeY = maxSize
    
    # Generate UV coordinates
    uvs = []
    for v in vertices:
        u = (v[0] - minX) / sizeX
        v_coord = 1.0 - (v[1] - minY) / sizeY  # Flip Y
        uvs.append((u, v_coord))
    
    # Add texture2d resource and bake image to texture
    try:
        # Save texture to temp file first (3MF needs file path for texture)
        texture_path = "/tmp/texture_temp.png"
        with open(texture_path, "wb") as f:
            f.write(png_bytes)
        
        # Add texture2d resource
        texture2d = None
        if hasattr(model, "AddTexture2D"):
            texture2d = model.AddTexture2D()
        elif hasattr(model, "CreateTexture2D"):
            texture2d = model.CreateTexture2D()
        else:
            raise AttributeError("No texture2d method found")
        
        # Set texture path/content
        if hasattr(texture2d, "SetPath"):
            texture2d.SetPath(texture_path)
        elif hasattr(texture2d, "SetContentPath"):
            texture2d.SetContentPath(texture_path)
        elif hasattr(texture2d, "SetAttachmentPath"):
            texture2d.SetAttachmentPath(texture_path)
        elif hasattr(texture2d, "SetContent"):
            # Try to set content directly
            texture2d.SetContent(png_bytes)
        
        texture_id = texture2d.GetResourceID() if hasattr(texture2d, "GetResourceID") else getattr(texture2d, "ResourceID", 0)
        print(f"✅ Created texture2d resource (ID: {texture_id})")
        
        # Set UV coordinates on mesh vertices
        # 3MF requires UV coordinates per vertex, matching vertex order
        uv_coords = []
        for u, v_coord in uvs:
            uv_coords.append((float(u), float(v_coord)))
        
        # Try different methods to set UV coordinates
        if hasattr(mesh_obj, "SetUVCoordinates"):
            mesh_obj.SetUVCoordinates(uv_coords)
        elif hasattr(mesh_obj, "SetTextureCoordinates"):
            mesh_obj.SetTextureCoordinates(uv_coords)
        elif hasattr(mesh_obj, "SetVertexUVCoordinates"):
            mesh_obj.SetVertexUVCoordinates(uv_coords)
        
        # Create material with texture
        if hasattr(model, "AddBaseMaterialGroup"):
            material_group = model.AddBaseMaterialGroup()
            material_id = material_group.GetResourceID()
            
            # Add a material that uses the texture
            if hasattr(material_group, "AddMaterial"):
                mat_idx = material_group.AddMaterial("TexturedMaterial", None)
                # Try to set texture on material
                if hasattr(material_group, "SetMaterialTexture"):
                    material_group.SetMaterialTexture(mat_idx, texture_id)
                elif hasattr(material_group, "SetMaterialTexture2D"):
                    material_group.SetMaterialTexture2D(mat_idx, texture_id)
        else:
            material_id = 0
        
        # Link texture to mesh
        if hasattr(mesh_obj, "SetTexture2D"):
            mesh_obj.SetTexture2D(texture_id)
        elif hasattr(mesh_obj, "SetTexture"):
            mesh_obj.SetTexture(texture_id)
        elif hasattr(mesh_obj, "SetMaterial"):
            mesh_obj.SetMaterial(material_id)
        
        # Set texture coordinates per triangle
        # 3MF might need texture coordinates per triangle vertex
        for tri_idx, face in enumerate(faces):
            # Get UV coordinates for this triangle's vertices
            uv0 = uvs[face[0]]
            uv1 = uvs[face[1]]
            uv2 = uvs[face[2]]
            
            # Try to set triangle texture coordinates
            if hasattr(mesh_obj, "SetTriangleTextureCoordinates"):
                try:
                    mesh_obj.SetTriangleTextureCoordinates(tri_idx, uv0, uv1, uv2)
                except:
                    pass
        
        print(f"✅ Baked texture to 3MF with UV coordinates")
    except Exception as e:
        import traceback
        print(f"⚠️  Warning: Could not add texture to 3MF: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print(f"   Falling back to OBJ with UV texture mapping...")
        # Fallback: Use OBJ with UV texture (more reliable)
        obj_bytes, mtl_bytes = write_obj_with_uv_texture(vertices, faces, img_rgb)
        # Convert OBJ to 3MF? No, just raise error to use OBJ instead
        raise RuntimeError(f"3MF texture not supported, use OBJ format instead: {e}")

    # Add to build
    identity = None
    if hasattr(wrapper, "GetIdentityTransform"):
        identity = wrapper.GetIdentityTransform()
    elif hasattr(mf, "Transform"):
        identity = mf.Transform()
    model.AddBuildItem(mesh_obj, identity)

    # Write to temporary file then read back
    tmp_path = "/tmp/output_temp.3mf"
    try:
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        # Fallback: try direct model write
        try:
            model.WriteToFile(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            raise RuntimeError(f"Failed to write 3MF file: {e}")


def write_3mf_vertex_colors(vertices: np.ndarray, faces: np.ndarray, triangle_colors: np.ndarray) -> bytes:
    """
    Write 3MF file with vertex colors (single object, colorgroup).
    Bakes triangle colors to vertices - each vertex gets the color of its face.
    Creates ONE unified mesh with vertex colors (no basematerials, no multiple objects).
    """
    import zipfile
    import tempfile
    
    print(f"✅ Writing 3MF with vertex colors (single object, {len(vertices)} vertices, {len(faces)} triangles)")
    
    # Step 1: Collect unique triangle colors and create color index mapping
    from collections import OrderedDict
    
    # Get unique colors from triangles (as tuples for hashing)
    unique_colors = OrderedDict()
    triangle_color_indices = []
    
    for tri_idx, face in enumerate(faces):
        r, g, b = int(triangle_colors[tri_idx][0]), int(triangle_colors[tri_idx][1]), int(triangle_colors[tri_idx][2])
        color_tuple = (r, g, b)
        
        if color_tuple not in unique_colors:
            unique_colors[color_tuple] = len(unique_colors)
        
        triangle_color_indices.append(unique_colors[color_tuple])
    
    print(f"✅ Found {len(unique_colors)} unique colors for {len(faces)} triangles")
    
    # Step 2: Build 3MF XML manually as string (for proper namespace handling)
    # Use color:colorresources extension (Bambu Studio compatible format)
    # This is the color extension namespace method for vertex colors
    
    # Build colorresources XML (placed BEFORE <resources> as per 3MF spec)
    color_xml = []
    color_xml.append('  <color:colorresources id="1">')
    color_xml.append('    <color:colors>')
    
    # Build color list - one color per unique color (this is the color palette)
    # Colors use 0-255 range (not normalized) for colorresources
    for color_tuple in unique_colors.keys():
        r, g, b = color_tuple
        color_xml.append(f'      <color:color r="{r}" g="{g}" b="{b}"/>')
    
    color_xml.append('    </color:colors>')
    color_xml.append('  </color:colorresources>')
    
    # Build vertices XML
    vertices_xml = []
    for i in range(len(vertices)):
        x, y, z = vertices[i]
        vertices_xml.append(f'          <vertex x="{float(x)}" y="{float(y)}" z="{float(z)}"/>')
    
    # Build triangles XML with colorid to reference colorresources
    # colorid references the color index in the colorresources palette
    triangles_xml = []
    for tri_idx, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        color_index = triangle_color_indices[tri_idx]
        # Use colorid to reference color in colorresources (all vertices get same color for triangle)
        triangles_xml.append(f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}" colorid="{color_index}"/>')
    
    # Build complete model XML with Bambu Lab metadata
    # Using 2013/01 namespace with color extension for vertex colors
    model_xml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter"
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2013/01"
       xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
       xmlns:cad="http://schemas.autodesk.com/cad/2018/03/3dmodel"
       xmlns:color="http://schemas.microsoft.com/3dmanufacturing/2013/01/color">
  <metadata name="Producer">Bambu Lab</metadata>
  <metadata name="Application">Bambu Studio</metadata>

{chr(10).join(color_xml)}
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
{chr(10).join(vertices_xml)}
        </vertices>
        <triangles>
{chr(10).join(triangles_xml)}
        </triangles>
      </mesh>
    </object>
  </resources>

  <build>
    <item objectid="1"/>
  </build>
</model>'''
    
    # Step 3: Create 3MF zip package
    with tempfile.NamedTemporaryFile(delete=False, suffix=".3mf") as tmp_file:
        tmp_path = tmp_file.name
    
    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model XML (built as string for proper namespace handling)
        zipf.writestr("3D/3dmodel.model", model_xml_str.encode('utf-8'))
        
        # Add [Content_Types].xml
        content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Override PartName="/3D/3dmodel.model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
        zipf.writestr("[Content_Types].xml", content_types)
        
        # Add _rels/.rels
        rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>'''
        zipf.writestr("_rels/.rels", rels)
    
    # Read and return
    with open(tmp_path, "rb") as f:
        result = f.read()
    
    os.unlink(tmp_path)
    return result


def write_3mf(vertices: np.ndarray, faces: np.ndarray, triangle_colors: np.ndarray) -> bytes:
    """
    Write 3MF file with vertex colors (single object, colorgroup).
    Uses the new vertex color approach instead of multiple objects with basematerials.
    """
    return write_3mf_vertex_colors(vertices, faces, triangle_colors)


def write_obj_with_uv_texture(vertices: np.ndarray, faces: np.ndarray, img_rgb: np.ndarray, grid_size: int = 75) -> Tuple[bytes, bytes]:
    """
    Write OBJ file with UV coordinates and MTL with texture file
    Uses the EXACT same UV mapping logic as the frontend 3D viewer
    Matches frontend applyColorsToMesh function pixel-perfectly
    """
    import io
    from PIL import Image
    
    # Get image dimensions
    img_height, img_width = img_rgb.shape[:2]
    
    # Compute XY bounds based ONLY on near-horizontal (top) faces (matches frontend exactly)
    tri_verts = vertices[faces]
    minX = np.inf
    minY = np.inf
    maxX = -np.inf
    maxY = -np.inf
    
    # Calculate face normals to find top faces
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    
    ab = v1 - v0
    ac = v2 - v0
    normals = np.cross(ab, ac)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths = np.where(norm_lengths == 0, 1, norm_lengths)
    normals_unit = normals / norm_lengths
    nz_unit = np.abs(normals_unit[:, 2])
    
    # Only use top-ish faces (nz > 0.5) to determine bounds
    top_mask = nz_unit > 0.5
    
    if np.any(top_mask):
        top_tri_verts = tri_verts[top_mask]
        for tri in top_tri_verts:
            minX = min(minX, tri[0, 0], tri[1, 0], tri[2, 0])
            minY = min(minY, tri[0, 1], tri[1, 1], tri[2, 1])
            maxX = max(maxX, tri[0, 0], tri[1, 0], tri[2, 0])
            maxY = max(maxY, tri[0, 1], tri[1, 1], tri[2, 1])
    
    if not (np.isfinite(minX) and np.isfinite(minY)):
        # Fallback to bbox if no top faces detected
        minX = vertices[:, 0].min()
        minY = vertices[:, 1].min()
        maxX = vertices[:, 0].max()
        maxY = vertices[:, 1].max()
    
    sizeX = max(1e-9, maxX - minX)
    sizeY = max(1e-9, maxY - minY)
    
    # NOTE: Frontend does NOT enforce square aspect ratio - it uses sizeX and sizeY directly
    # Matching frontend behavior exactly - do NOT square the aspect ratio
    # This ensures UV mapping matches exactly what the user sees in the 3D viewer
    
    # Create OBJ content
    obj_content = []
    obj_content.append("# Colored Album Cover Model with UV Texture")
    obj_content.append("# Texture mapping for pixel-perfect image display")
    obj_content.append("# UV coordinates match frontend 3D viewer exactly")
    obj_content.append("")
    
    # Write vertices
    for v in vertices:
        obj_content.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    obj_content.append("")
    
    # Generate UV coordinates using EXACT frontend logic
    # For Normal mode (48): Use continuous mapping (no grid snapping)
    # For other modes: Use grid snapping (cellU, cellV, snappedU, snappedV)
    is_normal_mode = (grid_size == 48)
    uvs = []
    
    # Compute UV for each vertex using the same logic as frontend color mapping
    # Frontend maps triangle centroids, but for UV we map vertex positions directly
    # and apply the same grid snapping logic
    for vertex in vertices:
        # Normalize XY coordinates to [0, 1] range (matches frontend line 3505-3506)
        u = max(0.0, min(0.999999, (vertex[0] - minX) / sizeX))
        v = max(0.0, min(0.999999, (vertex[1] - minY) / sizeY))
        
        if is_normal_mode:
            # Normal mode: continuous mapping (no grid snapping)
            # Map directly to UV coordinates - pixel-perfect mapping
            # Flip Y: v=0 (top) maps to top of image in UV space
            final_u = u
            final_v = 1.0 - v  # Flip Y: top of STL maps to top of image
        else:
            # Grid-based mapping for pixelated modes (EXACT frontend logic lines 3508-3514)
            grid = float(grid_size)
            
            # Frontend: cellU = Math.floor(u * grid) + 0.5
            cellU = np.floor(u * grid) + 0.5
            cellV = np.floor(v * grid) + 0.5
            
            # Frontend: snappedU = cellU / grid; snappedV = cellV / grid
            snappedU = cellU / grid
            snappedV = cellV / grid
            
            # Use snapped coordinates for UV (this matches what frontend does for color mapping)
            final_u = snappedU
            final_v = 1.0 - snappedV  # Flip Y: top of STL maps to top of image
        
        uvs.append((final_u, final_v))
    
    # Write UV coordinates
    for u, v_coord in uvs:
        obj_content.append(f"vt {u:.6f} {v_coord:.6f}")
    obj_content.append("")
    
    # CRITICAL: Add MTL reference so Bambu Studio loads the texture
    obj_content.append("mtllib output.mtl")
    obj_content.append("usemtl texture_material")
    obj_content.append("")
    
    # Write faces with UV indices (1-indexed)
    for face in faces:
        # OBJ format: f v1/vt1 v2/vt2 v3/vt3
        obj_content.append(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}")
    
    obj_bytes = "\n".join(obj_content).encode('utf-8')
    
    # Create MTL file with texture reference
    mtl_content = []
    mtl_content.append("# Material file for textured model")
    mtl_content.append("newmtl texture_material")
    mtl_content.append("Ka 1.000 1.000 1.000")  # Ambient color
    mtl_content.append("Kd 1.000 1.000 1.000")  # Diffuse color
    mtl_content.append("Ks 0.000 0.000 0.000")  # Specular color
    mtl_content.append("d 1.0")  # Dissolve (opacity)
    mtl_content.append("illum 1")  # Illumination model
    mtl_content.append("map_Kd texture.png")  # Diffuse texture map
    
    mtl_bytes = "\n".join(mtl_content).encode('utf-8')
    
    return obj_bytes, mtl_bytes


def write_obj_with_colors(vertices: np.ndarray, faces: np.ndarray, triangle_colors: np.ndarray, is_normal_mode: bool = False, img_rgb: np.ndarray = None) -> Tuple[bytes, bytes]:
    """
    Write OBJ file with colors separated into material groups for exact color matching.
    Groups triangles by their exact color and creates separate materials for each color.
    This ensures colors match exactly what's shown in the 3D viewer.
    Returns: (obj_bytes, mtl_bytes)
    
    NOTE: This is a fallback method using MTL materials. The primary method is
    write_obj_with_vertex_colors() which uses 'vc' commands for better Bambu Studio compatibility.
    
    Args:
        is_normal_mode: If True (48x48), preserves all colors. If False, ensures 4-color palette.
        img_rgb: Not used anymore - kept for compatibility
    """
    from collections import defaultdict
    
    if is_normal_mode:
        print(f"✅ Creating OBJ with material groups (Normal mode - all colors preserved)")
    else:
        print(f"✅ Creating OBJ with material groups (4-color palette)")
    
    # Group triangles by their exact color
    color_groups = defaultdict(list)
    for face_idx, face in enumerate(faces):
        color_tuple = tuple(triangle_colors[face_idx])
        color_groups[color_tuple].append((face_idx, face))
    
    unique_colors = list(color_groups.keys())
    print(f"📊 Colors in OBJ: {len(unique_colors)} unique colors")
    
    # Create OBJ file with materials
    obj_content = []
    obj_content.append("# Colored Album Cover Model")
    obj_content.append("# Colors preserved from original PNG")
    obj_content.append("# Materials separated by color for exact color matching")
    obj_content.append("# Vertices duplicated per triangle to prevent color interpolation")
    obj_content.append("")
    obj_content.append("mtllib output.mtl")
    obj_content.append("")
    
    # Build all vertices and faces, grouped by material
    all_vertices = []
    all_faces = []
    material_face_ranges = {}  # Track which faces belong to which material
    
    # Process each color group separately
    for color_idx, (color_tuple, triangle_list) in enumerate(color_groups.items()):
        r, g, b = color_tuple[0], color_tuple[1], color_tuple[2]
        material_name = f"color_{r}_{g}_{b}"
        material_face_ranges[material_name] = []
        
        # Start index for this material's faces
        face_start_idx = len(all_faces)
        
        # Process all triangles with this color
        for face_idx, face in triangle_list:
            # Get the three vertices for this triangle
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Add vertices (without color in OBJ - color comes from material)
            vertex_base_idx = len(all_vertices)
            all_vertices.append(v0)
            all_vertices.append(v1)
            all_vertices.append(v2)
            
            # Create new face indices (1-indexed for OBJ)
            all_faces.append((vertex_base_idx + 1, vertex_base_idx + 2, vertex_base_idx + 3, material_name))
        
        # End index for this material's faces
        face_end_idx = len(all_faces)
        material_face_ranges[material_name] = (face_start_idx, face_end_idx)
    
    # Write all vertices (without colors - colors come from materials)
    for v in all_vertices:
        obj_content.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    obj_content.append("")
    
    # Write faces grouped by material
    current_material = None
    for face_idx, (v1, v2, v3, material_name) in enumerate(all_faces):
        # Switch material when needed
        if material_name != current_material:
            obj_content.append(f"usemtl {material_name}")
            current_material = material_name
        
        # Write face
        obj_content.append(f"f {v1} {v2} {v3}")
    
    obj_bytes = "\n".join(obj_content).encode('utf-8')
    
    # Create MTL file with one material per color
    mtl_content = []
    mtl_content.append("# Material file for colored model")
    mtl_content.append("# Each color gets its own material for exact color matching")
    mtl_content.append("")
    
    # For pixelated modes (not Normal mode), ensure all 4 colors have materials
    if not is_normal_mode:
        color_names = ["Black", "Dark Gray", "Light Gray", "White"]
        for i, color in enumerate(FOUR_COLORS_RGB):
            color_tuple = tuple(color)
            if color_tuple in unique_colors:
                print(f"   ✓ {color_names[i]}: {color}")
            else:
                print(f"   ✗ {color_names[i]}: {color} (MISSING - will add material anyway)")
        
        # Ensure all 4 colors have materials (even if not used)
        for i, color in enumerate(FOUR_COLORS_RGB):
            r, g, b = color[0], color[1], color[2]
            material_name = f"color_{r}_{g}_{b}"
            
            # Add material definition
            mtl_content.append(f"newmtl {material_name}")
            mtl_content.append(f"Ka {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")  # Ambient color
            mtl_content.append(f"Kd {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")  # Diffuse color
            mtl_content.append(f"Ks 0.000 0.000 0.000")  # Specular color (black = no specular)
            mtl_content.append(f"d 1.0")  # Dissolve (opacity)
            mtl_content.append(f"illum 1")  # Illumination model
            mtl_content.append("")
        
        # Add materials for any additional colors (shouldn't happen in 4-color mode, but handle it)
        for color_tuple in unique_colors:
            r, g, b = color_tuple[0], color_tuple[1], color_tuple[2]
            if tuple(color_tuple) not in FOUR_COLORS_RGB:
                material_name = f"color_{r}_{g}_{b}"
                mtl_content.append(f"newmtl {material_name}")
                mtl_content.append(f"Ka {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")
                mtl_content.append(f"Kd {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")
                mtl_content.append(f"Ks 0.000 0.000 0.000")
                mtl_content.append(f"d 1.0")
                mtl_content.append(f"illum 1")
                mtl_content.append("")
    else:
        # Normal mode: create material for each unique color
        for color_tuple in sorted(unique_colors):
            r, g, b = color_tuple[0], color_tuple[1], color_tuple[2]
            material_name = f"color_{r}_{g}_{b}"
            
            mtl_content.append(f"newmtl {material_name}")
            mtl_content.append(f"Ka {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")  # Ambient color
            mtl_content.append(f"Kd {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}")  # Diffuse color
            mtl_content.append(f"Ks 0.000 0.000 0.000")  # Specular color
            mtl_content.append(f"d 1.0")  # Dissolve (opacity)
            mtl_content.append(f"illum 1")  # Illumination model
            mtl_content.append("")
    
    mtl_bytes = "\n".join(mtl_content).encode('utf-8')
    
    print(f"✅ Created OBJ with {len(unique_colors)} material groups")
    return obj_bytes, mtl_bytes


def write_obj_with_vertex_colors(vertices: np.ndarray, faces: np.ndarray, triangle_colors: np.ndarray, is_normal_mode: bool = False) -> bytes:
    """
    Write OBJ file with vertex colors embedded directly in vertex lines.
    Format: v x y z r g b (extended OBJ format widely supported by 3D software)
    This is more compatible with various 3D applications than MTL materials.
    Returns: obj_bytes (no MTL file needed - colors are embedded in OBJ)
    
    Args:
        vertices: Array of vertex positions
        faces: Array of face indices
        triangle_colors: Array of RGB colors for each triangle (0-255 range)
        is_normal_mode: If True (48x48), preserves all colors. If False, ensures 4-color palette.
    """
    if is_normal_mode:
        print(f"✅ Creating OBJ with vertex colors (Normal mode - all colors preserved)")
    else:
        print(f"✅ Creating OBJ with vertex colors (4-color palette)")
    
    # Create OBJ file with vertex colors
    obj_content = []
    obj_content.append("# Colored Album Cover Model")
    obj_content.append("# Colors preserved from original PNG")
    obj_content.append("# Vertex colors embedded directly: v x y z r g b")
    obj_content.append("# Vertices duplicated per triangle to prevent color interpolation")
    obj_content.append("")
    
    # Build all vertices with their colors
    # Duplicate vertices per triangle to prevent color interpolation at edges
    all_vertices = []
    all_vertex_colors = []
    all_faces = []
    
    for face_idx, face in enumerate(faces):
        # Get the three vertices for this triangle
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Get triangle color (normalize to 0.0-1.0 for OBJ vertex colors)
        r, g, b = triangle_colors[face_idx]
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # Add vertices and their colors
        vertex_base_idx = len(all_vertices)
        all_vertices.append(v0)
        all_vertex_colors.append((r_norm, g_norm, b_norm))
        all_vertices.append(v1)
        all_vertex_colors.append((r_norm, g_norm, b_norm))
        all_vertices.append(v2)
        all_vertex_colors.append((r_norm, g_norm, b_norm))
        
        # Create new face indices (1-indexed for OBJ)
        all_faces.append((vertex_base_idx + 1, vertex_base_idx + 2, vertex_base_idx + 3))
    
    # Write vertices with colors embedded directly: v x y z r g b
    # This is the extended OBJ format that's widely supported
    for i, v in enumerate(all_vertices):
        r, g, b = all_vertex_colors[i]
        obj_content.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.6f} {g:.6f} {b:.6f}")
    
    obj_content.append("")
    
    # Write faces
    for v1, v2, v3 in all_faces:
        obj_content.append(f"f {v1} {v2} {v3}")
    
    obj_bytes = "\n".join(obj_content).encode('utf-8')
    
    unique_colors = len(set(tuple(c) for c in triangle_colors))
    print(f"✅ Created OBJ with vertex colors ({unique_colors} unique colors, {len(all_vertices)} vertices)")
    return obj_bytes


def generate_3mf_from_inputs(stl_bytes: bytes, png_bytes: bytes, grid_size: int = 75) -> bytes:
    """
    Generate 3MF file with per-triangle colors that match frontend exactly
    Uses the same color mapping logic as the frontend 3D viewer
    Falls back to OBJ if 3MF library is not available
    """
    # Check if 3MF is available
    if _three_mf is None:
        raise RuntimeError("3MF library not available. Please install: pip install lib3mf")
    
    # For Normal mode (48), keep full resolution; otherwise resize to grid_size
    is_normal_mode = (grid_size == 48)
    if is_normal_mode:
        img = load_png(png_bytes, target_size=None)  # Keep full resolution
    else:
        img = load_png(png_bytes, grid_size)
    img_array = get_png_as_array(img)
    vertices, faces = load_stl_vertices_faces(stl_bytes)
    
    # For Normal mode, use actual image dimensions; otherwise use grid_size
    mapping_grid_size = img_array.shape[0] if is_normal_mode else grid_size
    triangle_colors = get_triangle_colors_from_image(vertices, faces, img_array, mapping_grid_size)
    
    # For Normal mode: use exact colors from image (no quantization)
    # For pixelated modes: quantize to 4 colors
    if not is_normal_mode:
        triangle_colors = quantize_to_four_colors(triangle_colors)
        print(f"✅ Generating 3MF with per-triangle colors (4-color palette)")
    else:
        print(f"✅ Generating 3MF with per-triangle colors (exact colors from image)")
    
    try:
        three_mf_bytes = write_3mf(vertices, faces, triangle_colors)
        return three_mf_bytes
    except Exception as e:
        error_msg = str(e)
        if "Lib3MFException" in error_msg or "COULDNOTLOADLIBRARY" in error_msg or ".dylib" in error_msg:
            raise RuntimeError(
                f"3MF library not properly installed. Error: {error_msg}\n"
                f"Please install lib3mf: pip install lib3mf\n"
                f"Or use OBJ format instead (which doesn't require lib3mf)."
            )
        raise


def generate_obj_from_inputs(stl_bytes: bytes, png_bytes: bytes, grid_size: int = 75) -> Tuple[bytes, bytes, bytes]:
    """
    Generate OBJ file with vertex colors (vc commands) for Bambu Studio compatibility.
    Uses vertex colors as primary method (more compatible than MTL materials).
    Returns: (obj_bytes, empty bytes, None)
    """
    # For Normal mode (48), keep full resolution; otherwise resize to grid_size
    is_normal_mode = (grid_size == 48)
    if is_normal_mode:
        img = load_png(png_bytes, target_size=None)  # Keep full resolution
    else:
        img = load_png(png_bytes, grid_size)
    img_array = get_png_as_array(img)
    vertices, faces = load_stl_vertices_faces(stl_bytes)
    
    # For Normal mode, use actual image dimensions; otherwise use grid_size
    mapping_grid_size = img_array.shape[0] if is_normal_mode else grid_size
    triangle_colors = get_triangle_colors_from_image(vertices, faces, img_array, mapping_grid_size)
    
    # For Normal mode: use exact colors from image (no quantization)
    # For pixelated modes: quantize to 4 colors
    if not is_normal_mode:
        triangle_colors = quantize_to_four_colors(triangle_colors)
        print(f"⚙️  Generating OBJ with vertex colors (4-color palette)")
    else:
        print(f"⚙️  Generating OBJ with vertex colors (exact colors from image)")
    
    # Generate OBJ with vertex colors (primary method for Bambu Studio)
    obj_bytes = write_obj_with_vertex_colors(vertices, faces, triangle_colors, is_normal_mode)
    
    # Return OBJ bytes, empty MTL bytes, and None for texture
    return obj_bytes, b"", None


# Flask app
app = Flask(__name__)

# Base directory - use absolute path to server.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📂 Base directory: {BASE_DIR}")

# Enable CORS for frontend access - allow all origins for admin panel and Hostinger domain
# Allow requests from Hostinger domain and localhost for development
CORS(app, resources={
    r"/admin/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Cache-Control", "Authorization"]},
    r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Cache-Control"]},
    r"/product_images/*": {"origins": "*", "methods": ["GET", "OPTIONS"], "allow_headers": ["Content-Type"]},
    r"/get-stl/*": {"origins": "*", "methods": ["GET", "OPTIONS"], "allow_headers": ["Content-Type", "Cache-Control"]}  # Allow STL file downloads from any origin
})

# Add after_request handler to ensure CORS headers are ALWAYS set on ALL responses
@app.after_request
def after_request(response):
    """Add CORS headers to all responses - this ensures preflight requests work"""
    # Always set CORS headers regardless of what flask-cors did
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Cache-Control, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'false'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response


@app.route('/')
def index():
    """Serve mobile or desktop version based on user agent"""
    user_agent = request.headers.get('User-Agent', '').lower()
    
    # Check if it's a mobile device
    is_mobile = any(device in user_agent for device in [
        'mobile', 'android', 'iphone', 'ipad', 'ipod', 
        'blackberry', 'windows phone', 'opera mini'
    ])
    
    # Also check screen width via query parameter (for testing)
    force_mobile = request.args.get('mobile') == 'true'
    force_desktop = request.args.get('desktop') == 'true'
    
    if force_mobile:
        mobile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobile', 'index.html')
        if not os.path.exists(mobile_path):
            print(f"❌ Mobile file not found at: {mobile_path}")
            return jsonify({'error': f'Mobile file not found: {mobile_path}'}), 404
        response = send_file(mobile_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    elif force_desktop:
        desktop_path = os.path.join(os.path.dirname(__file__), 'desktop.html')
        if not os.path.exists(desktop_path):
            print(f"❌ Desktop file not found at: {desktop_path}")
            return jsonify({'error': f'Desktop file not found: {desktop_path}'}), 404
        response = send_file(desktop_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    elif is_mobile:
        # Serve mobile version
        mobile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobile', 'index.html')
        if not os.path.exists(mobile_path):
            print(f"❌ Mobile file not found at: {mobile_path}")
            return jsonify({'error': f'Mobile file not found: {mobile_path}'}), 404
        response = send_file(mobile_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    else:
        # Serve desktop version
        desktop_path = os.path.join(os.path.dirname(__file__), 'desktop.html')
        if not os.path.exists(desktop_path):
            print(f"❌ Desktop file not found at: {desktop_path}")
            return jsonify({'error': f'Desktop file not found: {desktop_path}'}), 404
        response = send_file(desktop_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response

@app.route('/mobile')
def mobile():
    """Force mobile version"""
    # Use BASE_DIR for absolute path
    mobile_path = os.path.join(BASE_DIR, 'mobile', 'index.html')
    
    # Debug: Print path information
    print(f"🔍 Mobile route called")
    print(f"   __file__: {__file__}")
    print(f"   BASE_DIR: {BASE_DIR}")
    print(f"   mobile_path: {mobile_path}")
    print(f"   exists: {os.path.exists(mobile_path)}")
    print(f"   cwd: {os.getcwd()}")
    
    if not os.path.exists(mobile_path):
        # Try alternative path
        alt_path = os.path.join(os.getcwd(), 'mobile', 'index.html')
        print(f"   Trying alternative: {alt_path}")
        if os.path.exists(alt_path):
            mobile_path = alt_path
        else:
            return jsonify({
                'error': f'Mobile file not found',
                'tried': [mobile_path, alt_path],
                'BASE_DIR': BASE_DIR,
                'cwd': os.getcwd()
            }), 404
    
    response = send_file(mobile_path)
    # Force no-cache to prevent browser caching issues
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/mobile/fresh')
@app.route('/mobile/<int:timestamp>')
def mobile_fresh(timestamp=None):
    """Mobile version with cache busting"""
    mobile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobile', 'index.html')
    if not os.path.exists(mobile_path):
        print(f"❌ Mobile file not found at: {mobile_path}")
        return jsonify({'error': f'Mobile file not found: {mobile_path}'}), 404
    response = send_file(mobile_path)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/desktop')
@app.route('/desktop/mobile')
def desktop():
    """Force desktop version (accessible from mobile via /desktop or /desktop/mobile)"""
    desktop_path = os.path.join(os.path.dirname(__file__), 'desktop.html')
    if not os.path.exists(desktop_path):
        print(f"❌ Desktop file not found at: {desktop_path}")
        return jsonify({'error': f'Desktop file not found: {desktop_path}'}), 404
    response = send_file(desktop_path)
    # Force no-cache to prevent browser caching issues
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/generate', methods=['POST'])
def generate():
    """
    Accepts multipart/form-data with:
    - stl: STL file
    - png: PNG file
    Returns: OBJ file with vertex colors (Bambu Studio compatible)
    """
    try:
        if 'stl' not in request.files or 'png' not in request.files:
            return jsonify({'error': 'Missing stl or png file'}), 400

        stl_file = request.files['stl']
        png_file = request.files['png']

        stl_bytes = stl_file.read()
        png_bytes = png_file.read()

        grid_size = int(request.form.get('grid_size', 75))

        # Generate OBJ with vertex colors (primary method for Bambu Studio)
        obj_bytes, mtl_bytes, texture_bytes = generate_obj_from_inputs(stl_bytes, png_bytes, grid_size)

        # Return OBJ file directly (no MTL needed - colors are embedded via vc commands)
        return send_file(
            io.BytesIO(obj_bytes),
            mimetype='model/obj',
            as_attachment=True,
            download_name='output.obj'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-obj', methods=['POST'])
def generate_obj_route():
    """
    Accepts multipart/form-data with:
    - stl: STL file
    - png: PNG file
    Returns: ZIP file containing OBJ + MTL files
    """
    import zipfile
    import traceback
    import gc
    
    # Memory limits (in bytes)
    MAX_STL_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PNG_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        print("🎨 OBJ generation request received")
        
        if 'stl' not in request.files or 'png' not in request.files:
            print("❌ Missing files in request")
            return jsonify({'error': 'Missing stl or png file'}), 400

        stl_file = request.files['stl']
        png_file = request.files['png']
        
        print(f"📦 STL file: {stl_file.filename}")
        print(f"🖼️  PNG file: {png_file.filename}")

        # Check file sizes before reading
        stl_file.seek(0, 2)
        stl_size = stl_file.tell()
        stl_file.seek(0)
        
        png_file.seek(0, 2)
        png_size = png_file.tell()
        png_file.seek(0)
        
        if stl_size > MAX_STL_SIZE:
            return jsonify({'error': f'STL file too large. Maximum size: {MAX_STL_SIZE / 1024 / 1024:.0f}MB'}), 400
        if png_size > MAX_PNG_SIZE:
            return jsonify({'error': f'PNG file too large. Maximum size: {MAX_PNG_SIZE / 1024 / 1024:.0f}MB'}), 400

        stl_bytes = stl_file.read()
        png_bytes = png_file.read()
        
        # Clear file handles immediately
        del stl_file, png_file
        
        # Get grid size from form data (default to 75 if not provided)
        grid_size = int(request.form.get('grid_size', 75))
        
        print(f"📊 STL size: {len(stl_bytes)} bytes")
        print(f"📊 PNG size: {len(png_bytes)} bytes")
        print(f"📊 Grid size: {grid_size}x{grid_size}")

        # Generate OBJ with vertex colors (primary method for Bambu Studio)
        obj_bytes, mtl_bytes, texture_png_bytes = generate_obj_from_inputs(stl_bytes, png_bytes, grid_size)
        
        # OBJ with vertex colors doesn't need MTL file (colors are embedded via vc commands)
        # Return OBJ file directly
        print(f"✅ OBJ size: {len(obj_bytes)} bytes")
        
        # Clean up memory
        del stl_bytes, png_bytes
        if mtl_bytes:
            del mtl_bytes
        if texture_png_bytes:
            del texture_png_bytes
        gc.collect()
        print("📤 Sending OBJ file with vertex colors to client...")
        return send_file(
            io.BytesIO(obj_bytes),
            mimetype='model/obj',
            as_attachment=True,
            download_name='colored_model.obj'
        )

    except Exception as e:
        print(f"❌ Error generating OBJ: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get-stl/<int:size>', methods=['GET', 'OPTIONS'])
def get_stl(size):
    """
    Serves the pre-uploaded STL file for the specified grid size.
    Tries multiple filename patterns: {size}x{size}_grid.stl, {size}x{size}.stl
    """
    # Handle OPTIONS preflight request - MUST return 200 with CORS headers
    if request.method == 'OPTIONS':
        print(f"✅ Handling OPTIONS preflight request for /get-stl/{size}")
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Cache-Control, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
        return response, 200
    
    try:
        if size not in [48, 75, 96]:
            error_msg = f'Invalid grid size: {size}. Must be 48, 75, or 96.'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Try multiple filename patterns
        possible_paths = [
            os.path.join('stl_files', f'{size}x{size}_grid.stl'),
            os.path.join('stl_files', f'{size}x{size}.stl'),
            os.path.join('shopify-version', 'stl_files', f'{size}x{size}_grid.stl'),
            os.path.join('shopify-version', 'stl_files', f'{size}x{size}.stl'),
        ]
        
        stl_path = None
        for path in possible_paths:
            if os.path.exists(path):
                stl_path = path
                print(f"✅ Found STL file at: {stl_path}")
                break
        
        if not stl_path:
            error_msg = f'STL file not found for {size}×{size} grid. Tried: {", ".join(possible_paths)}. Please upload via admin panel.'
            print(f"❌ {error_msg}")
            response = jsonify({'error': error_msg, 'size': size, 'tried_paths': possible_paths})
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 404
        
        print(f"📦 Serving STL file: {stl_path} (size: {os.path.getsize(stl_path)} bytes)")
        response = send_file(
            stl_path,
            mimetype='application/octet-stream',
            as_attachment=False
        )
        # Ensure CORS headers are set
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error serving STL for size {size}: {error_details}")
        response = jsonify({'error': str(e), 'size': size, 'details': error_details})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500


@app.route('/upload-for-checkout', methods=['POST'])
def upload_for_checkout():
    """
    Accepts multipart/form-data with:
    - stl: STL file
    - png: PNG file
    Generates OBJ+MTL, saves them with a unique order ID, and returns the order ID
    """
    import zipfile
    import traceback
    import gc
    
    # Memory limits (in bytes)
    MAX_STL_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PNG_SIZE = 10 * 1024 * 1024  # 10MB
    
    try:
        print("🛒 Checkout upload request received")
        
        if 'stl' not in request.files or 'png' not in request.files:
            print("❌ Missing files in request")
            return jsonify({'error': 'Missing stl or png file'}), 400

        stl_file = request.files['stl']
        png_file = request.files['png']
        
        print(f"📦 STL file: {stl_file.filename}")
        print(f"🖼️  PNG file: {png_file.filename}")

        # Check file sizes before reading
        stl_file.seek(0, 2)  # Seek to end
        stl_size = stl_file.tell()
        stl_file.seek(0)  # Reset to beginning
        
        png_file.seek(0, 2)
        png_size = png_file.tell()
        png_file.seek(0)
        
        if stl_size > MAX_STL_SIZE:
            return jsonify({'error': f'STL file too large. Maximum size: {MAX_STL_SIZE / 1024 / 1024:.0f}MB'}), 400
        if png_size > MAX_PNG_SIZE:
            return jsonify({'error': f'PNG file too large. Maximum size: {MAX_PNG_SIZE / 1024 / 1024:.0f}MB'}), 400
        
        print(f"📊 File sizes: STL={stl_size / 1024:.1f}KB, PNG={png_size / 1024:.1f}KB")

        stl_bytes = stl_file.read()
        png_bytes = png_file.read()
        
        # Clear file handles immediately
        del stl_file, png_file
        
        # Get grid size from form data (default to 75 if not provided)
        grid_size = int(request.form.get('grid_size', 75))
        
        # Get order details from form data
        stand_selected = request.form.get('stand_selected', 'true').lower() == 'true'
        mounting_selected = request.form.get('mounting_selected', 'false').lower() == 'true'
        total_price = float(request.form.get('total_price', 0.0))
        
        # Get base price from Shopify (for validation/logging only - actual price comes from total_price parameter)
        base_price = 0.0
        try:
            if SHOPIFY_API_AVAILABLE:
                shopify_api = get_shopify_api()
                if shopify_api.is_configured():
                    # Get variant ID for this grid size
                    variant_id_map = {
                        48: os.getenv('SHOPIFY_VARIANT_48', '10470738559281'),
                        75: os.getenv('SHOPIFY_VARIANT_75', '10470738952497'),
                        96: os.getenv('SHOPIFY_VARIANT_96', '10470739312945')
                    }
                    variant_id = variant_id_map.get(grid_size)
                    if variant_id:
                        variant_ids = {f'{grid_size}x{grid_size}': variant_id}
                        prices = shopify_api.get_variant_prices(variant_ids)
                        if prices:
                            base_price = prices.get(f'{grid_size}x{grid_size}', 0.0)
        except Exception as e:
            print(f"⚠️  Warning: Could not load price from Shopify: {e}")
            # Continue anyway - actual price comes from total_price parameter
        
        # Use OBJ format with vertex colors (Bambu Studio compatible)
        obj_bytes, mtl_bytes, texture_bytes = generate_obj_from_inputs(stl_bytes, png_bytes, grid_size)
        
        # Create unique order ID
        order_id = str(uuid.uuid4())
        print(f"🆔 Generated order ID: {order_id}")
        print(f"💰 Total Price: ${total_price:.2f}")
        
        # Create orders directory if it doesn't exist (use absolute path)
        orders_dir = os.path.join(BASE_DIR, 'orders')
        os.makedirs(orders_dir, exist_ok=True)
        
        # Save files with order ID
        order_dir = os.path.join(orders_dir, order_id)
        os.makedirs(order_dir, exist_ok=True)
        
        # Save OBJ file with vertex colors (Bambu Studio compatible)
        model_path = os.path.join(order_dir, 'model.obj')
        with open(model_path, 'wb') as f:
            f.write(obj_bytes)
        print(f"✅ Saved model.obj to {order_dir}")
        
        # Also save original PNG for reference
        png_path = os.path.join(order_dir, 'original.png')
        with open(png_path, 'wb') as f:
            f.write(png_bytes)
        
        # Save STL file for easy access
        stl_path = os.path.join(order_dir, 'model.stl')
        with open(stl_path, 'wb') as f:
            f.write(stl_bytes)
        
        # Create order metadata
        from datetime import datetime, timezone
        order_data = {
            'order_id': order_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'grid_size': grid_size,
            'dimensions': f"{grid_size}×{grid_size}",
            'base_price': base_price,
            'stand_selected': stand_selected,
            'mounting_selected': mounting_selected,
            'total_price': total_price,
            'addons': [],
            'completed': False
        }
        
        if stand_selected:
            order_data['addons'].append('Stand')
        if mounting_selected:
            order_data['addons'].append('Nano Wall Mounting Dots')
        
        # Save order metadata to orders.json (use absolute path)
        orders_file = os.path.join(BASE_DIR, 'orders.json')
        orders = []
        if os.path.exists(orders_file):
            try:
                with open(orders_file, 'r') as f:
                    orders = json.load(f)
            except:
                orders = []
        
        orders.append(order_data)
        
        with open(orders_file, 'w') as f:
            json.dump(orders, f, indent=2)
        
        print(f"✅ Files saved to {order_dir}")
        print(f"📋 Order saved: {order_id}")
        
        # Clean up memory
        del stl_bytes, png_bytes, obj_bytes
        if mtl_bytes:
            del mtl_bytes
        if texture_bytes:
            del texture_bytes
        gc.collect()  # Force garbage collection
        
        # Return order ID and price
        return jsonify({
            'order_id': order_id,
            'price': total_price,
            'grid_size': grid_size,
            'message': 'Order prepared successfully'
        })
        
    except Exception as e:
        print(f"❌ Error in checkout upload: {e}")
        traceback.print_exc()
        # Clean up on error
        gc.collect()
        return jsonify({'error': str(e)}), 500


@app.route('/api/shopify/variants', methods=['GET'])
def get_shopify_variants():
    """Return Shopify variant IDs from environment variables"""
    try:
        variants = {
            'variant_48': os.getenv('SHOPIFY_VARIANT_48', '10470738559281'),  # Default from image
            'variant_75': os.getenv('SHOPIFY_VARIANT_75', '10470738952497'),  # Default from image
            'variant_96': os.getenv('SHOPIFY_VARIANT_96', '10470739312945'),  # Default from image
            'variant_stand': os.getenv('SHOPIFY_VARIANT_STAND', '10470741901617'),  # Default from image
            'variant_mounting': os.getenv('SHOPIFY_VARIANT_MOUNTING', '10470742655281'),  # Default from image
            'store_url': os.getenv('SHOPIFY_STORE_URL')
        }
        
        # Filter out None values
        variants = {k: v for k, v in variants.items() if v}
        
        return jsonify(variants)
    except Exception as e:
        print(f"❌ Error getting Shopify variants: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/create-cart', methods=['POST'])
def create_shopify_cart():
    """
    Create a Shopify cart via Storefront API and return checkout URL
    
    Expected request body:
    {
        "size": 48|75|96,
        "quantity": 1,
        "customization_id": "cst_abc123",
        "addons": {
            "stand": true|false,
            "mounting_dots": true|false
        }
    }
    
    Returns:
    {
        "checkout_url": "https://...",
        "cart_id": "gid://shopify/Cart/...",
        "customization_id": "cst_abc123"
    }
    """
    if not SHOPIFY_STOREFRONT_API_AVAILABLE:
        return jsonify({'error': 'Shopify Storefront API not available'}), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        size = data.get('size')
        quantity = data.get('quantity', 1)
        customization_id = data.get('customization_id')
        addons = data.get('addons', {})
        
        # Validate required fields
        if not size or size not in [48, 75, 96]:
            return jsonify({'error': 'Invalid size. Must be 48, 75, or 96'}), 400
        
        if not customization_id:
            return jsonify({'error': 'customization_id is required'}), 400
        
        # Get Storefront API instance
        storefront_api = get_storefront_api()
        if not storefront_api.is_configured():
            return jsonify({'error': 'Shopify Storefront API not configured'}), 503
        
        # Determine variant GID based on size (server-side validation)
        # Map size to variant GID from environment variables
        variant_env_key = f'SHOPIFY_VARIANT_{size}'
        variant_gid = os.getenv(variant_env_key)
        
        if not variant_gid:
            return jsonify({
                'error': f'Variant for size {size}x{size} not configured',
                'help': f'Set {variant_env_key} environment variable'
            }), 500
        
        print(f"🛒 Creating cart for {size}x{size} (variant: {variant_gid})")
        print(f"   Customization ID: {customization_id}")
        print(f"   Addons: {addons}")
        
        # Build line attributes
        attributes = [
            {'key': 'size', 'value': f'{size}x{size}'},
            {'key': 'customization_id', 'value': customization_id}
        ]
        
        # Add addon flags as attributes
        if addons.get('stand'):
            attributes.append({'key': 'addon_stand', 'value': 'true'})
        if addons.get('mounting_dots'):
            attributes.append({'key': 'addon_mounting_dots', 'value': 'true'})
        
        # Create cart with main product variant
        cart_result = storefront_api.create_cart(
            variant_gid=variant_gid,
            quantity=quantity,
            customization_id=customization_id,
            attributes=attributes
        )
        
        if not cart_result:
            return jsonify({'error': 'Failed to create cart'}), 500
        
        cart_id = cart_result['cart_id']
        checkout_url = cart_result['checkout_url']
        
        # Add addon items to cart if selected
        if addons.get('stand'):
            stand_variant_gid = os.getenv('SHOPIFY_VARIANT_STAND')
            if stand_variant_gid:
                print(f"   Adding stand addon (variant: {stand_variant_gid})")
                storefront_api.add_cart_lines(
                    cart_id=cart_id,
                    variant_gid=stand_variant_gid,
                    quantity=1,
                    attributes=[{'key': 'addon_type', 'value': 'stand'}]
                )
            else:
                print(f"⚠️  Stand addon requested but SHOPIFY_VARIANT_STAND not configured")
        
        if addons.get('mounting_dots'):
            mounting_variant_gid = os.getenv('SHOPIFY_VARIANT_MOUNTING')
            if mounting_variant_gid:
                print(f"   Adding mounting dots addon (variant: {mounting_variant_gid})")
                storefront_api.add_cart_lines(
                    cart_id=cart_id,
                    variant_gid=mounting_variant_gid,
                    quantity=1,
                    attributes=[{'key': 'addon_type', 'value': 'mounting_dots'}]
                )
            else:
                print(f"⚠️  Mounting dots addon requested but SHOPIFY_VARIANT_MOUNTING not configured")
        
        print(f"✅ Cart created successfully")
        print(f"   Cart ID: {cart_id}")
        print(f"   Checkout URL: {checkout_url}")
        
        # Return checkout URL and cart info
        return jsonify({
            'checkout_url': checkout_url,
            'cart_id': cart_id,
            'customization_id': customization_id,
            'success': True
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error creating cart: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/shopify/orders', methods=['GET'])
def get_shopify_orders():
    """Get orders from Shopify Admin API"""
    if not SHOPIFY_API_AVAILABLE:
        return jsonify({'error': 'Shopify API not available'}), 503
    
    try:
        shopify_api = get_shopify_api()
        if not shopify_api.is_configured():
            return jsonify({'error': 'Shopify API not configured'}), 503
        
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status', None)
        
        orders = shopify_api.get_orders(limit=limit, status=status)
        
        # Convert to JSON-serializable format
        orders_data = []
        for order in orders:
            orders_data.append({
                'id': str(order.id),
                'name': order.name,
                'email': order.email if hasattr(order, 'email') else None,
                'total_price': float(order.total_price) if hasattr(order, 'total_price') else 0,
                'financial_status': order.financial_status if hasattr(order, 'financial_status') else None,
                'fulfillment_status': order.fulfillment_status if hasattr(order, 'fulfillment_status') else None,
                'created_at': order.created_at if hasattr(order, 'created_at') else None,
                'line_items': [
                    {
                        'id': str(item.id),
                        'title': item.title,
                        'quantity': item.quantity,
                        'price': float(item.price) if hasattr(item, 'price') else 0,
                        'properties': [
                            {'name': prop.name, 'value': prop.value}
                            for prop in (item.properties if hasattr(item, 'properties') else [])
                        ]
                    }
                    for item in (order.line_items if hasattr(order, 'line_items') else [])
                ]
            })
        
        return jsonify({'orders': orders_data})
    except Exception as e:
        print(f"❌ Error getting Shopify orders: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/shopify/order/<order_id>', methods=['GET'])
def get_shopify_order(order_id):
    """Get a specific order from Shopify"""
    if not SHOPIFY_API_AVAILABLE:
        return jsonify({'error': 'Shopify API not available'}), 503
    
    try:
        shopify_api = get_shopify_api()
        if not shopify_api.is_configured():
            return jsonify({'error': 'Shopify API not configured'}), 503
        
        order = shopify_api.get_order(order_id)
        if not order:
            return jsonify({'error': 'Order not found'}), 404
        
        # Convert to JSON-serializable format
        order_data = {
            'id': str(order.id),
            'name': order.name,
            'email': order.email if hasattr(order, 'email') else None,
            'total_price': float(order.total_price) if hasattr(order, 'total_price') else 0,
            'financial_status': order.financial_status if hasattr(order, 'financial_status') else None,
            'fulfillment_status': order.fulfillment_status if hasattr(order, 'fulfillment_status') else None,
            'created_at': order.created_at if hasattr(order, 'created_at') else None,
            'line_items': [
                {
                    'id': str(item.id),
                    'title': item.title,
                    'quantity': item.quantity,
                    'price': float(item.price) if hasattr(item, 'price') else 0,
                    'properties': [
                        {'name': prop.name, 'value': prop.value}
                        for prop in (item.properties if hasattr(item, 'properties') else [])
                    ]
                }
                for item in (order.line_items if hasattr(order, 'line_items') else [])
            ]
        }
        
        return jsonify(order_data)
    except Exception as e:
        print(f"❌ Error getting Shopify order: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/webhooks/orders/create', methods=['POST'])
def webhook_order_create():
    """Handle Shopify order creation webhook"""
    if not WEBHOOK_HANDLERS_AVAILABLE:
        return jsonify({'error': 'Webhook handlers not available'}), 503
    
    try:
        # Verify webhook signature
        hmac_header = request.headers.get('X-Shopify-Hmac-Sha256')
        if hmac_header:
            shopify_api = get_shopify_api_for_webhook()
            if not shopify_api.verify_webhook_signature(request.data, hmac_header):
                print("❌ Invalid webhook signature")
                return jsonify({'error': 'Invalid signature'}), 401
        
        # Get order data
        order_data = request.get_json()
        
        # Process webhook
        success = handle_order_create(order_data)
        
        if success:
            return '', 200
        else:
            return jsonify({'error': 'Failed to process webhook'}), 500
            
    except Exception as e:
        print(f"❌ Error processing order create webhook: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/webhooks/orders/paid', methods=['POST'])
def webhook_order_paid():
    """Handle Shopify order paid webhook"""
    if not WEBHOOK_HANDLERS_AVAILABLE:
        return jsonify({'error': 'Webhook handlers not available'}), 503
    
    try:
        # Verify webhook signature
        hmac_header = request.headers.get('X-Shopify-Hmac-Sha256')
        if hmac_header:
            shopify_api = get_shopify_api_for_webhook()
            if not shopify_api.verify_webhook_signature(request.data, hmac_header):
                print("❌ Invalid webhook signature")
                return jsonify({'error': 'Invalid signature'}), 401
        
        # Get order data
        order_data = request.get_json()
        
        # Process webhook
        success = handle_order_paid(order_data)
        
        if success:
            return '', 200
        else:
            return jsonify({'error': 'Failed to process webhook'}), 500
            
    except Exception as e:
        print(f"❌ Error processing order paid webhook: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin')
@app.route('/admin/')
@app.route('/admin/prices')
def admin_prices_page():
    """Serve admin price editor HTML page"""
    try:
        # Use BASE_DIR for absolute path
        admin_path = os.path.join(BASE_DIR, 'admin.html')
        
        # Check if file exists
        if not os.path.exists(admin_path):
            # Try current working directory
            admin_path = os.path.join(os.getcwd(), 'admin.html')
            if not os.path.exists(admin_path):
                # Return helpful error with debug info
                current_dir = os.getcwd()
                base_dir_files = os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
                current_dir_files = os.listdir(current_dir) if os.path.exists(current_dir) else []
                
                return jsonify({
                    'error': 'Admin file not found',
                    'message': 'admin.html file is missing',
                    'debug': {
                        'BASE_DIR': BASE_DIR,
                        'current_dir': current_dir,
                        'base_dir_files': base_dir_files[:20],  # Limit to first 20
                        'current_dir_files': current_dir_files[:20],
                        'admin_path_attempted': os.path.join(BASE_DIR, 'admin.html')
                    },
                    'help': 'Make sure admin.html is in the same directory as server.py'
                }), 404
        
        return send_file(admin_path, mimetype='text/html')
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/admin/prices/api', methods=['GET', 'POST'])
def admin_prices_api():
    """Admin API to get prices from Shopify (read-only, prices cannot be edited here)"""
    
    if request.method == 'GET':
        # Return current prices (try Shopify first, then fallback to prices.json)
        try:
            # Try Shopify first if available
            if SHOPIFY_API_AVAILABLE:
                variant_ids = {
                    '48x48': os.getenv('SHOPIFY_VARIANT_48', '10470738559281'),
                    '75x75': os.getenv('SHOPIFY_VARIANT_75', '10470738952497'),
                    '96x96': os.getenv('SHOPIFY_VARIANT_96', '10470739312945'),
                    'stand': os.getenv('SHOPIFY_VARIANT_STAND', '10470741901617'),
                    'wall_mounting_dots': os.getenv('SHOPIFY_VARIANT_MOUNTING', '10470742655281')
                }
                
                # Filter out None values
                variant_ids = {k: v for k, v in variant_ids.items() if v}
                
                try:
                    shopify_api = get_shopify_api()
                    if shopify_api.is_configured():
                        prices = shopify_api.get_variant_prices(variant_ids)
                        if prices:
                            result = {
                                '48x48': prices.get('48x48', 0),
                                '75x75': prices.get('75x75', 0),
                                '96x96': prices.get('96x96', 0),
                                'stand': prices.get('stand', 0),
                                'wall_mounting_dots': prices.get('wall_mounting_dots', 0)
                            }
                            return jsonify(result)
                except Exception as e:
                    print(f"⚠️ Shopify API failed in admin, falling back to prices.json: {e}")
            
            # Fallback to local prices.json file
            prices_file = os.path.join(BASE_DIR, 'prices.json')
            if os.path.exists(prices_file):
                with open(prices_file, 'r') as f:
                    prices = json.load(f)
                return jsonify(prices)
            
            # If no prices available at all
            return jsonify({'error': 'No prices configured'}), 503
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Save prices to local prices.json file
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Validate price data
            required_keys = ['48x48', '75x75', '96x96', 'stand', 'wall_mounting_dots']
            for key in required_keys:
                if key not in data:
                    return jsonify({'error': f'Missing required price: {key}'}), 400
                try:
                    float(data[key])
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid price value for {key}'}), 400
            
            # Save to prices.json
            prices_file = os.path.join(BASE_DIR, 'prices.json')
            with open(prices_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return jsonify({'success': True, 'message': 'Prices saved successfully'})
        except Exception as e:
            print(f"❌ Error saving prices: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Public API to get current prices from Shopify (read-only for customers)"""
    try:
        # Try Shopify first if available
        if SHOPIFY_API_AVAILABLE:
            # Get variant IDs from environment or use defaults from image
            variant_ids = {
                '48x48': os.getenv('SHOPIFY_VARIANT_48', '10470738559281'),
                '75x75': os.getenv('SHOPIFY_VARIANT_75', '10470738952497'),
                '96x96': os.getenv('SHOPIFY_VARIANT_96', '10470739312945'),
                'stand': os.getenv('SHOPIFY_VARIANT_STAND', '10470741901617'),
                'wall_mounting_dots': os.getenv('SHOPIFY_VARIANT_MOUNTING', '10470742655281')
            }
            
            # Filter out None values
            variant_ids = {k: v for k, v in variant_ids.items() if v}
            
            try:
                shopify_api = get_shopify_api()
                if shopify_api.is_configured():
                    prices = shopify_api.get_variant_prices(variant_ids)
                    if prices:
                        return jsonify(prices)
            except Exception as e:
                print(f"⚠️ Shopify API failed, falling back to prices.json: {e}")
        
        # Fallback to local prices.json file
        prices_file = os.path.join(BASE_DIR, 'prices.json')
        if os.path.exists(prices_file):
            with open(prices_file, 'r') as f:
                prices = json.load(f)
            return jsonify(prices)
        
        # If no prices available at all
        return jsonify({'error': 'No prices configured'}), 503
    except Exception as e:
        print(f"❌ Error in get_prices: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/admin/images/api', methods=['GET', 'POST'])
def admin_images_api():
    """Admin API to get/upload images for products"""
    images_file = 'images.json'
    images_dir = 'product_images'
    os.makedirs(images_dir, exist_ok=True)
    
    if request.method == 'GET':
        # Return current images as JSON
        try:
            if os.path.exists(images_file):
                try:
                    with open(images_file, 'r') as f:
                        images = json.load(f)
                except json.JSONDecodeError as e:
                    # If JSON is corrupted, backup and return empty dict
                    import shutil
                    backup_file = images_file + '.backup.' + str(int(time.time()))
                    try:
                        shutil.copy2(images_file, backup_file)
                        print(f"⚠️ images.json is corrupted, backed up to {backup_file}, returning empty")
                    except:
                        print(f"⚠️ images.json is corrupted, returning empty (backup failed)")
                    images = {}
            else:
                images = {}
            response = jsonify(images)
            # Prevent caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        except Exception as e:
            import traceback
            print(f"❌ Error reading images: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Upload image
        try:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
            file = request.files['image']
            key = request.form.get('key')
            
            if not key:
                return jsonify({'success': False, 'error': 'No key provided'}), 400
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            # Generate filename
            filename = f"{key}_{uuid.uuid4().hex[:8]}.{file.filename.rsplit('.', 1)[1].lower()}"
            filepath = os.path.join(images_dir, filename)
            file.save(filepath)
            
            # Update images.json
            if os.path.exists(images_file):
                try:
                    with open(images_file, 'r') as f:
                        images = json.load(f)
                except json.JSONDecodeError as e:
                    # If JSON is corrupted, backup the file and start fresh
                    import shutil
                    backup_file = images_file + '.backup.' + str(int(time.time()))
                    try:
                        shutil.copy2(images_file, backup_file)
                        print(f"⚠️ images.json is corrupted, backed up to {backup_file}, starting fresh")
                    except:
                        print(f"⚠️ images.json is corrupted, starting fresh (backup failed)")
                    images = {}
            else:
                images = {}
            
            images[key] = f'/product_images/{filename}'
            
            with open(images_file, 'w') as f:
                json.dump(images, f, indent=2)
            
            return jsonify({'success': True, 'imageUrl': images[key]})
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error uploading image: {error_details}")
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/product_images/<filename>', methods=['GET', 'OPTIONS'])
def serve_product_image(filename):
    """Serve product images"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response, 200
    
    images_dir = 'product_images'
    filepath = os.path.join(images_dir, filename)
    if os.path.exists(filepath):
        response = send_file(filepath)
        # Prevent caching of images
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # CORS headers already set by after_request, but ensure they're there
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        response = jsonify({'error': 'Image not found'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 404


@app.route('/api/images', methods=['GET'])
def get_images():
    """Public API to get product images (read-only)"""
    images_file = 'images.json'
    try:
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                images = json.load(f)
        else:
            images = {}
        response = jsonify(images)
        # Prevent caching so images update immediately
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/stl/api', methods=['GET', 'POST'])
def admin_stl_api():
    """Admin API to upload/get STL files"""
    stl_dir = 'stl_files'
    os.makedirs(stl_dir, exist_ok=True)
    
    if request.method == 'GET':
        # Return status of STL files
        statuses = {}
        for size in [48, 75, 96]:
            filename = f'{size}x{size}_grid.stl'
            filepath = os.path.join(stl_dir, filename)
            statuses[f'{size}x{size}'] = os.path.exists(filepath)
        return jsonify(statuses)
    
    elif request.method == 'POST':
        # Upload STL file
        try:
            if 'stl' not in request.files:
                return jsonify({'success': False, 'error': 'No STL file provided'}), 400
            
            file = request.files['stl']
            size = request.form.get('size')
            
            if not size:
                return jsonify({'success': False, 'error': 'No size provided'}), 400
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            # Save STL file
            filename = f'{size}x{size}_grid.stl'
            filepath = os.path.join(stl_dir, filename)
            file.save(filepath)
            
            return jsonify({'success': True, 'filename': filename})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/admin/content/api', methods=['GET', 'POST'])
def admin_content_api():
    """Admin API to get/edit text content"""
    content_file = 'content.json'
    
    # Default content - always use this as base
    default_content = {
        'title': '3D Album Cover Mosaic Builder',
        'price_subtitle': 'Create colorized 3D prints',
        'upload_image_text': 'Choose image file...',
        'upload_subtext': 'Will be resized to 75×75 pixels',
        'stl_upload_text': 'Choose STL file...',
        'stl_subtext': 'Or auto-load from server',
        'info_title': 'Custom Brick Mosaic Designer',
        'info_description': 'Turn your favourite photos into stunning brick art—made by you!',
        'info_additional': 'Bring your memories to life, one brick at a time. With our Custom Brick Mosaic Designer you can transform any image into a beautiful 3D printable mosaic.',
        'howto_title': 'How to Use',
        'howto_content': '1. Upload Your Image\n2. Select Grid Size\n3. Adjust Image\n4. View in 3D',
        'desktop_info': 'Each pixel in your PNG maps to one cube in the STL grid. Colors are preserved exactly as-is.',
        'desktop_output': 'OBJ file with MTL colors - Import to Bambu Studio and export as 3MF for printing.',
        # Desktop orange text labels
        'panel_title': 'Edit Your Photo',
        'canvas_label': 'Processed (Posterized)',
        'section_upload': '1. Upload Color Image',
        'section_grid': '2. Select Grid Size',
        'section_adjustments': 'Image Adjustments',
        'section_painting': 'Painting',
        # Additional editable text fields
        'grid_btn_48': '48 × 48',
        'grid_btn_75': '75 × 75',
        'grid_btn_96': '96 × 96',
        'slider_contrast_label': 'Contrast',
        'slider_brightness_label': 'Brightness',
        'slider_tones_label': 'Tones',
        'label_dimensions': 'Dimensions:',
        'label_addons': 'Addons:',
        'label_48x48': '48×48:',
        'label_75x75': '75×75:',
        'label_96x96': '96×96:',
        'stand_name': 'Stand',
        'stand_upload_btn': 'Upload Image',
        'mounting_name': 'Nano Wall Mounting Dots (Pack of 8)',
        'mounting_upload_btn': 'Upload Image',
        'color_black_title': 'Black',
        'color_darkgray_title': 'Dark Gray',
        'color_lightgray_title': 'Light Gray',
        'color_white_title': 'White',
        'size_guide_title': 'Size Guide',
        'size_guide_button_text': 'Size Guide',
        'size_guide': {
            'square': [
                {'bricks': '32 × 32 bricks', 'cm': '28.6 × 28.6 cm'},
                {'bricks': '32 × 32 bricks*', 'cm': '25.6 × 25.6 cm'},
                {'bricks': '48 × 48 bricks', 'cm': '41.4 × 41.4 cm'},
                {'bricks': '48 × 48 bricks*', 'cm': '38.4 × 38.4 cm'},
                {'bricks': '64 × 64 bricks', 'cm': '54.2 × 54.2 cm'},
                {'bricks': '96 × 96 bricks', 'cm': '79.8 × 79.8 cm'}
            ],
            'portrait': [],
            'landscape': [],
            'note': '* Unframed baseplate. All other measurements shown are for the framed baseplate.'
        }
    }
    
    if request.method == 'GET':
        # Return current content merged with defaults
        try:
            content = default_content.copy()  # Start with defaults
            
            if os.path.exists(content_file):
                with open(content_file, 'r') as f:
                    saved_content = json.load(f)
                    # Deep merge saved content into defaults (saved content overrides defaults)
                    def deep_merge(default, saved):
                        result = default.copy()
                        for key, value in saved.items():
                            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                                # Recursively merge nested dictionaries
                                result[key] = deep_merge(result[key], value)
                            else:
                                # Overwrite with saved value
                                result[key] = value
                        return result
                    content = deep_merge(content, saved_content)
            
            return jsonify(content)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Update content - merge with defaults to ensure all fields are present
        try:
            new_content = request.get_json()
            
            # Deep merge function for nested dictionaries
            def deep_merge(default, new):
                result = default.copy()
                for key, value in new.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        # Recursively merge nested dictionaries
                        result[key] = deep_merge(result[key], value)
                    else:
                        # Overwrite with new value
                        result[key] = value
                return result
            
            # Merge with defaults to ensure all fields are present
            merged_content = deep_merge(default_content, new_content)
            
            # Save the merged content
            with open(content_file, 'w') as f:
                json.dump(merged_content, f, indent=2)
            
            return jsonify({'success': True, 'content': merged_content})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/content', methods=['GET'])
def get_content():
    """Public API to get text content (read-only)"""
    content_file = 'content.json'
    
    # Default content - always use this as base
    default_content = {
        'title': '3D Album Cover Mosaic Builder',
        'price_subtitle': 'Create colorized 3D prints',
        'upload_image_text': 'Choose image file...',
        'upload_subtext': 'Will be resized to 75×75 pixels',
        'stl_upload_text': 'Choose STL file...',
        'stl_subtext': 'Or auto-load from server',
        'info_title': 'Custom Brick Mosaic Designer',
        'info_description': 'Turn your favourite photos into stunning brick art—made by you!',
        'info_additional': 'Bring your memories to life, one brick at a time. With our Custom Brick Mosaic Designer you can transform any image into a beautiful 3D printable mosaic.',
        'howto_title': 'How to Use',
        'howto_content': '1. Upload Your Image\n2. Select Grid Size\n3. Adjust Image\n4. View in 3D',
        'desktop_info': 'Each pixel in your PNG maps to one cube in the STL grid. Colors are preserved exactly as-is.',
        'desktop_output': 'OBJ file with MTL colors - Import to Bambu Studio and export as 3MF for printing.',
        # Desktop orange text labels
        'panel_title': 'Edit Your Photo',
        'canvas_label': 'Processed (Posterized)',
        'section_upload': '1. Upload Color Image',
        'section_grid': '2. Select Grid Size',
        'section_adjustments': 'Image Adjustments',
        'section_painting': 'Painting',
        # Additional editable text fields
        'grid_btn_48': '48 × 48',
        'grid_btn_75': '75 × 75',
        'grid_btn_96': '96 × 96',
        'slider_contrast_label': 'Contrast',
        'slider_brightness_label': 'Brightness',
        'slider_tones_label': 'Tones',
        'label_dimensions': 'Dimensions:',
        'label_addons': 'Addons:',
        'label_48x48': '48×48:',
        'label_75x75': '75×75:',
        'label_96x96': '96×96:',
        'stand_name': 'Stand',
        'stand_upload_btn': 'Upload Image',
        'mounting_name': 'Nano Wall Mounting Dots (Pack of 8)',
        'mounting_upload_btn': 'Upload Image',
        'color_black_title': 'Black',
        'color_darkgray_title': 'Dark Gray',
        'color_lightgray_title': 'Light Gray',
        'color_white_title': 'White',
        'size_guide_title': 'Size Guide',
        'size_guide_button_text': 'Size Guide',
        'size_guide': {
            'square': [
                {'bricks': '32 × 32 bricks', 'cm': '28.6 × 28.6 cm'},
                {'bricks': '32 × 32 bricks*', 'cm': '25.6 × 25.6 cm'},
                {'bricks': '48 × 48 bricks', 'cm': '41.4 × 41.4 cm'},
                {'bricks': '48 × 48 bricks*', 'cm': '38.4 × 38.4 cm'},
                {'bricks': '64 × 64 bricks', 'cm': '54.2 × 54.2 cm'},
                {'bricks': '96 × 96 bricks', 'cm': '79.8 × 79.8 cm'}
            ],
            'portrait': [],
            'landscape': [],
            'note': '* Unframed baseplate. All other measurements shown are for the framed baseplate.'
        }
    }
    
    try:
        content = default_content.copy()  # Start with defaults
        
        if os.path.exists(content_file):
            with open(content_file, 'r') as f:
                saved_content = json.load(f)
                # Deep merge saved content into defaults (saved content overrides defaults)
                def deep_merge(default, saved):
                    result = default.copy()
                    for key, value in saved.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            # Recursively merge nested dictionaries
                            result[key] = deep_merge(result[key], value)
                        else:
                            # Overwrite with saved value
                            result[key] = value
                    return result
                content = deep_merge(content, saved_content)
        
        response = jsonify(content)
        # Prevent caching so admin updates are seen immediately
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert-image', methods=['POST'])
def convert_image():
    """Convert HEIC/HEIF or other formats to JPEG"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file into memory
        file_data = file.read()
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        
        # Try to open image with PIL (supports HEIC if pillow-heif is installed)
        try:
            img = Image.open(io.BytesIO(file_data))
            
            # Convert RGBA to RGB if necessary (removes alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[3] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG to BytesIO
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=92)
            output.seek(0)
            
            # Return the converted image
            return send_file(
                output,
                mimetype='image/jpeg',
                as_attachment=False
            )
            
        except Exception as img_error:
            error_msg = f"Failed to convert image: {str(img_error)}"
            print(f"❌ Image conversion error: {error_msg}")
            
            # Provide helpful error message based on format
            if file_extension in ('heic', 'heif'):
                if not HEIF_SUPPORT_AVAILABLE:
                    error_msg = "HEIC/HEIF conversion not available. Please install pillow-heif library on the server or convert the image to JPEG format."
            else:
                error_msg = f"Unable to process {file_extension.upper() if file_extension else 'image'} format. Please convert to JPEG format."
            
            return jsonify({'error': error_msg}), 400
            
    except Exception as e:
        print(f"❌ Error in convert-image endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/admin/orders/api', methods=['GET', 'POST', 'DELETE'])
def admin_orders_api():
    """Admin API to get, update, or delete orders"""
    orders_file = os.path.join(BASE_DIR, 'orders.json')
    
    if request.method == 'GET':
        if not os.path.exists(orders_file):
            return jsonify([])
        
        try:
            with open(orders_file, 'r') as f:
                orders = json.load(f)
            # Return orders in reverse chronological order (newest first)
            orders.reverse()
            return jsonify(orders)
        except Exception as e:
            print(f"❌ Error loading orders: {e}")
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Update order (mark as completed/uncompleted)
        try:
            data = request.get_json()
            order_id = data.get('order_id')
            completed = data.get('completed', False)
            
            if not os.path.exists(orders_file):
                return jsonify({'error': 'No orders file found'}), 404
            
            with open(orders_file, 'r') as f:
                orders = json.load(f)
            
            # Find and update the order
            for order in orders:
                if order['order_id'] == order_id:
                    order['completed'] = completed
                    break
            
            with open(orders_file, 'w') as f:
                json.dump(orders, f, indent=2)
            
            return jsonify({'success': True})
        except Exception as e:
            print(f"❌ Error updating order: {e}")
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'DELETE':
        # Delete order
        try:
            order_id = request.args.get('order_id')
            
            if not order_id:
                return jsonify({'error': 'order_id required'}), 400
            
            if not os.path.exists(orders_file):
                return jsonify({'error': 'No orders file found'}), 404
            
            with open(orders_file, 'r') as f:
                orders = json.load(f)
            
            # Remove the order
            orders = [order for order in orders if order['order_id'] != order_id]
            
            with open(orders_file, 'w') as f:
                json.dump(orders, f, indent=2)
            
            # Also delete the order directory (use absolute path)
            import shutil
            order_dir = os.path.join(BASE_DIR, 'orders', order_id)
            if os.path.exists(order_dir):
                shutil.rmtree(order_dir)
            
            return jsonify({'success': True})
        except Exception as e:
            print(f"❌ Error deleting order: {e}")
            return jsonify({'error': str(e)}), 500


@app.route('/admin/orders/download/<order_id>/<filename>', methods=['GET', 'OPTIONS'])
def download_order_file(order_id, filename):
    """Download order files (STL, OBJ, PNG)"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response, 200
    
    try:
        # Use absolute path based on BASE_DIR
        order_dir = os.path.join(BASE_DIR, 'orders', order_id)
        original_filename = filename
        
        print(f"🔍 Download request: order_id={order_id}, filename={original_filename}")
        print(f"   BASE_DIR: {BASE_DIR}")
        print(f"   Order directory: {order_dir}")
        print(f"   Order dir exists: {os.path.exists(order_dir)}")
        
        # Check if order directory exists
        if not os.path.exists(order_dir):
            error_msg = f'Order directory not found: {order_id}'
            print(f"❌ {error_msg}")
            response = jsonify({
                'error': error_msg,
                'order_id': order_id,
                'order_dir': order_dir
            })
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 404
        
        # Special handling for model.3mf - generate on demand if needed
        if filename == 'model.3mf':
            file_path = os.path.join(order_dir, 'model.3mf')
            
            # If 3MF file doesn't exist, generate it
            if not os.path.exists(file_path):
                print(f"📦 3MF file not found, generating on-demand...")
                
                # Load order metadata to get grid_size
                orders_file = os.path.join(BASE_DIR, 'orders.json')
                grid_size = 75  # default
                
                if os.path.exists(orders_file):
                    try:
                        with open(orders_file, 'r') as f:
                            orders = json.load(f)
                            for order in orders:
                                if order.get('order_id') == order_id:
                                    grid_size = order.get('grid_size', 75)
                                    print(f"   Found grid_size: {grid_size}")
                                    break
                    except Exception as e:
                        print(f"⚠️  Could not load order metadata: {e}, using default grid_size=75")
                
                # Read STL and PNG files
                stl_path = os.path.join(order_dir, 'model.stl')
                png_path = os.path.join(order_dir, 'original.png')
                
                if not os.path.exists(stl_path):
                    error_msg = f'STL file not found in order: model.stl'
                    print(f"❌ {error_msg}")
                    response = jsonify({
                        'error': error_msg,
                        'order_id': order_id
                    })
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    return response, 404
                
                if not os.path.exists(png_path):
                    error_msg = f'PNG file not found in order: original.png'
                    print(f"❌ {error_msg}")
                    response = jsonify({
                        'error': error_msg,
                        'order_id': order_id
                    })
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    return response, 404
                
                # Read files
                with open(stl_path, 'rb') as f:
                    stl_bytes = f.read()
                with open(png_path, 'rb') as f:
                    png_bytes = f.read()
                
                print(f"   Read STL: {len(stl_bytes)} bytes, PNG: {len(png_bytes)} bytes")
                
                # Generate 3MF file
                try:
                    print(f"   Generating 3MF with grid_size={grid_size}...")
                    three_mf_bytes = generate_3mf_from_inputs(stl_bytes, png_bytes, grid_size)
                    
                    # Save the generated 3MF file for future requests
                    with open(file_path, 'wb') as f:
                        f.write(three_mf_bytes)
                    print(f"✅ Generated and saved 3MF file: {len(three_mf_bytes)} bytes")
                except Exception as gen_error:
                    import traceback
                    error_details = traceback.format_exc()
                    error_msg = f'Error generating 3MF file: {str(gen_error)}'
                    print(f"❌ {error_msg}")
                    print(f"   Traceback: {error_details}")
                    response = jsonify({
                        'error': error_msg,
                        'order_id': order_id,
                        'details': str(gen_error)
                    })
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    return response, 500
        
        file_path = os.path.join(order_dir, filename)
        print(f"   File path: {file_path}")
        print(f"   File exists: {os.path.exists(file_path)}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            # List what files actually exist in the order directory
            try:
                existing_files = os.listdir(order_dir)
                print(f"   Available files in order directory: {existing_files}")
            except Exception as list_error:
                print(f"   Error listing directory: {list_error}")
                existing_files = []
            
            error_msg = f'File not found: {filename}'
            print(f"❌ {error_msg}")
            response = jsonify({
                'error': error_msg,
                'order_id': order_id,
                'requested_file': original_filename,
                'resolved_file': filename,
                'file_path': file_path,
                'available_files': existing_files
            })
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 404
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(file_path):
            error_msg = f'Path is not a file: {filename}'
            print(f"❌ {error_msg}")
            response = jsonify({
                'error': error_msg,
                'order_id': order_id,
                'file_path': file_path
            })
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 400
        
        # Determine MIME type based on file extension
        mimetype = None
        if filename.endswith('.obj'):
            mimetype = 'model/obj'
        elif filename.endswith('.mtl'):
            mimetype = 'text/plain'
        elif filename.endswith('.stl'):
            mimetype = 'application/octet-stream'
        elif filename.endswith('.3mf'):
            mimetype = 'application/vnd.ms-package.3dmanufacturing-3dmodel+xml'
        elif filename.endswith('.png'):
            mimetype = 'image/png'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            mimetype = 'image/jpeg'
        
        print(f"✅ Serving file: {file_path} (size: {os.path.getsize(file_path)} bytes, mimetype: {mimetype})")
        
        # Send the file
        response = send_file(
            file_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        # CORS headers already set by after_request, but ensure they're there
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f'Error serving file: {str(e)}'
        print(f"❌ {error_msg}")
        print(f"   Traceback: {error_details}")
        response = jsonify({
            'error': error_msg,
            'order_id': order_id,
            'filename': filename,
            'details': error_details
        })
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500


def validate_shopify_credentials():
    """Validate Shopify API credentials are configured"""
    store_url = os.getenv('SHOPIFY_STORE_URL')
    api_key = os.getenv('SHOPIFY_API_KEY')
    api_secret = os.getenv('SHOPIFY_API_SECRET')
    
    if not store_url or not api_key or not api_secret:
        print("⚠️  Shopify credentials not configured")
        print("   Set SHOPIFY_STORE_URL, SHOPIFY_API_KEY, and SHOPIFY_API_SECRET in .env")
        print("   See SHOPIFY_SETUP_GUIDE.md for instructions")
        return False
    
    # Check if values are still placeholders
    if 'your-store' in store_url or 'your_api_key' in api_key or 'your_api_secret' in api_secret:
        print("⚠️  Shopify credentials appear to be placeholders")
        print("   Please update .env with your actual credentials")
        return False
    
    print("✅ Shopify credentials configured")
    return True


if __name__ == '__main__':
    print("🚀 Starting Album Cover 3D Color Mapper server...")
    
    # Validate Shopify credentials if attempting to use Shopify features
    validate_shopify_credentials()
    
    # Use PORT environment variable if available (for production), otherwise default to 5001
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_ENV') != 'production'
    
    print(f"📂 Open http://localhost:{port} in your browser")
    print(f"🔧 Admin price editor: http://localhost:{port}/admin/prices")
    
    app.run(debug=debug, port=port, host='0.0.0.0')

