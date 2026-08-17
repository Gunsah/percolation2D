import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
import json
from collections import deque

def load_tiff_3d(path):
    """Loads a multi-page TIFF and returns a 3D numpy array (z, y, x)."""
    with tifffile.TiffFile(path) as tif:
        images = tif.asarray()
    return images

def save_tiff_3d(arr, path, dtype=np.int32):
    """Saves a 3D numpy array as a multi-page TIFF."""
    tifffile.imwrite(path, arr.astype(dtype), compression=None)


# (use the fast C++ version if available)
try:
    from bfs_propagate import propagate_3d
    print("Using accelerated C++ version of propagate_3d")
except ImportError:
    print("C++ module not found, using slow Python version of propagate_3d")
    def propagate_3d(binary_img_3d, stim_rect):
        """BFS propagation in 3D (Python)."""
        zmax, ymax, xmax = binary_img_3d.shape
        lat = np.zeros_like(binary_img_3d, dtype=int)
        q = deque()
        xmin, xmax_rect, ymin, ymax_rect, zmin, zmax_rect = stim_rect
        for z in range(zmin, zmax_rect+1):
            for y in range(ymin, ymax_rect+1):
                for x in range(xmin, xmax_rect+1):
                    if 0 <= z < zmax and 0 <= y < ymax and 0 <= x < xmax and binary_img_3d[z, y, x] == 255:
                        lat[z, y, x] = 1
                        q.append((x, y, z))
        while q:
            x, y, z = q.popleft()
            step = lat[z, y, x]
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < xmax and 0 <= ny < ymax and 0 <= nz < zmax:
                    if binary_img_3d[nz, ny, nx] == 255 and lat[nz, ny, nx] == 0:
                        lat[nz, ny, nx] = step + 1
                        q.append((nx, ny, nz))
        return lat


# ---------- 3D blurring ----------
def blur_3d(binary_img_3d, sigma):
    """
    Applies 3D Gaussian blur to a binary image.
    """
    img_norm = binary_img_3d.astype(np.float32) / 255.0
    
    p = np.empty_like(img_norm)
    
    gaussian_filter(img_norm, sigma=sigma, output=p, mode='reflect')
    
    np.clip(p, 0, 1, out=p)
    return p


# ---------- Noise generation based on blurred probability ----------
def generate_noisy_image_3d(p_3d, original_shape=None):
    """
    Generates a 3D binary image (0 and 255) from a probability map p_3d.
    """
    if original_shape is not None:
        from scipy.ndimage import zoom
        factors = (original_shape[0] / p_3d.shape[0],
                   original_shape[1] / p_3d.shape[1],
                   original_shape[2] / p_3d.shape[2])
        prob = zoom(p_3d, factors, order=1, mode='reflect')
        np.clip(prob, 0, 1, out=prob)
    else:
        prob = p_3d
        original_shape = p_3d.shape
    
    random_vals = np.random.random(original_shape)
    noisy = (random_vals < prob).astype(np.uint8) * 255
    return noisy


# ---------- Remaining functions (unchanged) ----------
def transform_p_3d(p_3d, x0, x1, y1, smooth=False):
    """
    Applies the p -> p' transformation to each element of a 3D array.
    """
    p_prime = np.zeros_like(p_3d)
    mask1 = p_3d <= x0
    if x0 > 0:
        p_prime[mask1] = (0.5972 / x0) * p_3d[mask1]
    else:
        p_prime[mask1] = 0.0
    
    if smooth:
        mask_rest = p_3d > x0
        x_pts = np.array([x0, x1, 1.0])
        y_pts = np.array([0.5972, y1, 1.0])
        interp = PchipInterpolator(x_pts, y_pts)
        p_prime[mask_rest] = interp(p_3d[mask_rest])
    else:
        mask2 = (p_3d > x0) & (p_3d <= x1)
        if x1 > x0:
            slope = (y1 - 0.5972) / (x1 - x0)
            p_prime[mask2] = 0.5972 + slope * (p_3d[mask2] - x0)
        else:
            p_prime[mask2] = 0.5972
        
        mask3 = p_3d > x1
        if 1 - x1 > 0:
            slope = (1 - y1) / (1 - x1)
            p_prime[mask3] = y1 + slope * (p_3d[mask3] - x1)
        else:
            p_prime[mask3] = y1
    
    np.clip(p_prime, 0, 1, out=p_prime)
    return p_prime


def compute_loss_3d(lat_orig, lat_noisy_list, points, radius=1):
    """Computes the total loss by comparing mean LAT values in patches around keypoints."""
    z_dim, y_dim, x_dim = lat_orig.shape
    total_loss = 0.0
    for (x, y, z) in points:
        x0 = max(0, x - radius)
        x1 = min(x_dim, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(y_dim, y + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(z_dim, z + radius + 1)
        
        patch_orig = lat_orig[z0:z1, y0:y1, x0:x1]
        nonzero_orig = patch_orig[patch_orig > 0]
        mean_orig = np.mean(nonzero_orig) if len(nonzero_orig) else 0.0
        
        means_noisy = []
        for lat_noisy in lat_noisy_list:
            patch_noisy = lat_noisy[z0:z1, y0:y1, x0:x1]
            nonzero_noisy = patch_noisy[patch_noisy > 0]
            mean_noisy = np.mean(nonzero_noisy) if len(nonzero_noisy) else 0.0
            means_noisy.append(mean_noisy)
        mean_noisy_avg = np.mean(means_noisy) if means_noisy else 0.0
        
        total_loss += abs(mean_orig - mean_noisy_avg)
    return total_loss


def save_config(config_path, rect, points):
    """Saves the rectangle and points to a JSON configuration file."""
    data = {'rect': rect, 'points': points}
    with open(config_path, 'w') as f:
        json.dump(data, f)

def load_config(config_path):
    """Loads the rectangle and points from a JSON configuration file."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    rect = tuple(data['rect'])
    points = [tuple(p) for p in data['points']]
    return rect, points