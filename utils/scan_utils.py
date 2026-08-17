import numpy as np
import os

def scalar_transform_p(val, x0, x1, y1, smooth=False):
    """Applies the p -> p' transformation to a scalar value val."""
    if val <= x0:
        if x0 > 0:
            return (0.5972 / x0) * val
        else:
            return 0.0
    if smooth:
        from scipy.interpolate import PchipInterpolator
        x_pts = np.array([x0, x1, 1.0])
        y_pts = np.array([0.5972, y1, 1.0])
        interp = PchipInterpolator(x_pts, y_pts)
        return interp(val)
    else:
        if val <= x1:
            if x1 > x0:
                slope = (y1 - 0.5972) / (x1 - x0)
                return 0.5972 + slope * (val - x0)
            else:
                return 0.5972
        else:
            if 1 - x1 > 0:
                slope = (1 - y1) / (1 - x1)
                return y1 + slope * (val - x1)
            else:
                return y1

def generate_points_in_triangle(x0, S):
    """Generates S random points (x1, y1) uniformly distributed inside the triangle defined by x0."""
    points = []
    A = np.array([x0, 0.5972])
    B = np.array([1.0, 0.5972])
    C = np.array([x0, 1.0])
    for _ in range(S):
        u, v = np.random.rand(2)
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        point = A + u * (B - A) + v * (C - A)
        x1, y1 = point[0], point[1]
        if x1 <= x0 or y1 <= 0.5972 or y1 >= 1.0 or x1 >= 1.0:
            continue
        points.append((x1, y1))
    while len(points) < S:
        u, v = np.random.rand(2)
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        point = A + u * (B - A) + v * (C - A)
        x1, y1 = point[0], point[1]
        if x1 > x0 and y1 > 0.5972 and y1 < 1.0 and x1 < 1.0:
            points.append((x1, y1))
    return points

def mean_in_patch(lat, point, radius=1):
    """Computes the mean of non-zero values in a 3D patch centered at the given point."""
    x, y, z = point
    z_dim, y_dim, x_dim = lat.shape
    x0 = max(0, x - radius)
    x1 = min(x_dim, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(y_dim, y + radius + 1)
    z0 = max(0, z - radius)
    z1 = min(z_dim, z + radius + 1)
    patch = lat[z0:z1, y0:y1, x0:x1]
    nonzero = patch[patch > 0]
    return np.mean(nonzero) if len(nonzero) else 0.0