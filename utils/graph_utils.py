import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d


def create_cv_interpolator(kind='linear'):
    """
    Creates an interpolation function for CV(p').
    kind: 'linear' or 'cubic' (for smoother CV interpolation).
    """
    p_vals = np.array(sorted(CV_TABLE.keys()))
    cv_vals = np.array([CV_TABLE[p] for p in p_vals])
    if kind == 'cubic':
        # Cubic interpolation (requires scipy)
        from scipy.interpolate import CubicSpline
        interp = CubicSpline(p_vals, cv_vals, extrapolate=False)
        # For p' < 0.5972 return 0
        def wrapped_cv(p_prime):
            result = interp(p_prime)
            result[p_prime < 0.5972] = 0.0
            return result
        return wrapped_cv
    else:
        interp = interp1d(p_vals, cv_vals, kind='linear', bounds_error=False, fill_value=0.0)
        return interp

def transform_p(p_val, x0, x1, y1, smooth=True):
    """Transform p -> p' (scalar version)."""
    if p_val <= x0:
        if x0 > 0:
            return (0.5972 / x0) * p_val
        else:
            return 0.0

    if smooth:
        x_pts = np.array([x0, x1, 1.0])
        y_pts = np.array([0.5972, y1, 1.0])
        interp = PchipInterpolator(x_pts, y_pts)
        return interp(p_val)
    else:
        if p_val <= x1:
            if x1 > x0:
                slope = (y1 - 0.5972) / (x1 - x0)
                return 0.5972 + slope * (p_val - x0)
            else:
                return 0.5972
        else:
            if 1 - x1 > 0:
                slope = (1 - y1) / (1 - x1)
                return y1 + slope * (p_val - x1)
            else:
                return y1

def parse_line(line):
    """Extracts x0, x1, y1 from a string."""
    match = re.search(r'x0=([0-9.]+),\s*x1=([0-9.]+),\s*y1=([0-9.]+)', line)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None

def plot_pprime_curves(params_list, p_vals, smooth_pprime, alpha, output_file, dpi):
    """Plots and saves the p'(p) graph."""
    plt.figure(figsize=(7, 6))
    for x0, x1, y1 in params_list:
        p_prime_vals = np.array([transform_p(p, x0, x1, y1, smooth=smooth_pprime) for p in p_vals])
        plt.plot(p_vals, p_prime_vals, color='gray', alpha=alpha, linewidth=0.8)
    # Ideal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label="p' = p")
    plt.xlabel('p')
    plt.ylabel("p'")
    plt.title(f"p'(p) curves for {len(params_list)} parameter sets\nsmooth={smooth_pprime}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    print(f"p'(p) plot saved to {output_file}")
    plt.close()

def plot_cv_curves(params_list, p_vals, smooth_pprime, cv_interp_kind, alpha, output_file, dpi):
    """Plots and saves the CV(p) graph."""
    cv_interp = create_cv_interpolator(kind=cv_interp_kind)
    plt.figure(figsize=(7, 6))
    for x0, x1, y1 in params_list:
        p_prime_vals = np.array([transform_p(p, x0, x1, y1, smooth=smooth_pprime) for p in p_vals])
        cv_vals = cv_interp(p_prime_vals)
        plt.plot(p_vals, cv_vals, color='gray', alpha=alpha, linewidth=0.8)
    # Curve for p' = p
    cv_identity = cv_interp(p_vals)
    plt.plot(p_vals, cv_identity, 'k--', linewidth=1.5, label="CV(p) for p' = p")
    plt.xlabel('p')
    plt.ylabel('CV')
    plt.title(f"CV(p) curves for {len(params_list)} parameter sets\n"
              f"smooth(p'→p)={smooth_pprime}, CV interpolation={cv_interp_kind}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    print(f"CV(p) plot saved to {output_file}")
    plt.close()