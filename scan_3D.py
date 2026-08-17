import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from tqdm import tqdm
from utils.three_d_utils import (load_tiff_3d, save_tiff_3d, propagate_3d,
                      blur_3d, transform_p_3d,
                      generate_noisy_image_3d, compute_loss_3d,
                      save_config, load_config)
from utils.scan_utils import scalar_transform_p, generate_points_in_triangle, mean_in_patch

def main():
    parser = argparse.ArgumentParser(description='3D scanning by parameters with keypoints')
    parser.add_argument('input', help='Path to the source 3D TIFF')
    parser.add_argument('--config', required=True, help='Path to the configuration file (rectangle and points)')
    parser.add_argument('--sigma', type=float, required=True, help='Gaussian blur parameter')
    parser.add_argument('--scan-repeats', type=int, default=1, help='Number of repetitions for each set')
    
    parser.add_argument('--x0-min', type=float, default=0.0, help='Minimum x0')
    parser.add_argument('--x0-max', type=float, default=0.5972, help='Maximum x0')
    parser.add_argument('--x0-steps', type=int, default=10, help='Number of steps for x0')
    
    parser.add_argument('--S', type=int, default=50, help='Number of random points (x1,y1) for each x0')
    
    parser.add_argument('--output', help='Base name for output files (without extension)')
    parser.add_argument('--smooth', action='store_true', help='Use smoothed p(p) curve')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    # ---------- Loading and dimension normalization ----------
    img = load_tiff_3d(args.input)
    print(f"Image loaded, original dimensions: {img.shape}")

    # Convert to 3D (add Z axis if needed)
    if img.ndim == 2:
        print("2D image detected: adding Z axis (converting to 3D with a single layer).")
        img = img[np.newaxis, :, :]  # (1, y, x)
    
    elif img.ndim != 3:
        raise ValueError(f"Image must be 2D or 3D, but has {img.ndim} dimensions")

    

    z_dim, y_dim, x_dim = img.shape
    print(f"Processing size: {img.shape} (z, y, x)")

    if y_dim != x_dim:
        raise ValueError("Image must be square in XY")

    # ---------- Loading configuration ----------
    stim_rect, points = load_config(args.config)
    print(f"Loaded rectangle: {stim_rect}")
    print(f"Loaded points: {len(points)}")

    # Adjustment for 2D (z_dim == 1)
    if z_dim == 1:
        xmin, xmax, ymin, ymax, zmin, zmax = stim_rect
        if zmin != 0 or zmax != 0:
            print(f"Warning: for a 2D image, the rectangle's z-coordinates are ignored, using zmin=0, zmax=0")
            stim_rect = (xmin, xmax, ymin, ymax, 0, 0)
        # Adjust points
        new_points = []
        for (x, y, z) in points:
            if z != 0:
                print(f"Warning: point ({x},{y},{z}) has z={z}, but the image is 2D; using z=0")
                new_points.append((x, y, 0))
            else:
                new_points.append((x, y, z))
        points = new_points

    if len(points) == 0:
        raise ValueError("No keypoints in the configuration.")

    # ---------- Original propagation ----------
    lat_orig = propagate_3d(img, stim_rect)
    save_tiff_3d(lat_orig, 'data/lat_original.tiff')
    print("Original LAT map saved to data/lat_original.tiff")

    # ---------- Blurring ----------
    p = blur_3d(img, args.sigma)

    # ---------- Generating parameter combinations ----------
    x0_vals = np.linspace(args.x0_min, args.x0_max, args.x0_steps)
    all_combinations = []
    for x0 in x0_vals:
        points_xy = generate_points_in_triangle(x0, args.S)
        for (x1, y1) in points_xy:
            all_combinations.append((x0, x1, y1))

    total = len(all_combinations)
    print(f"Total combinations: {total}")

    results = []  # (x0, x1, y1, loss)

    # ---------- Main loop ----------
    for (x0, x1, y1) in tqdm(all_combinations, desc="Scanning"):
        p_prime = transform_p_3d(p, x0, x1, y1, smooth=args.smooth)
        lat_noisy_list = []
        for rep in range(args.scan_repeats):
            noisy_img = generate_noisy_image_3d(p_prime, img.shape)
            lat_noisy = propagate_3d(noisy_img, stim_rect)
            lat_noisy_list.append(lat_noisy)
        loss = compute_loss_3d(lat_orig, lat_noisy_list, points)
        results.append((x0, x1, y1, loss))

    results = np.array(results, dtype=[('x0', float), ('x1', float), ('y1', float), ('loss', float)])
    results.sort(order='loss')

    if args.output:
        np.savez(args.output + '_all.npz',
                 x0=results['x0'], x1=results['x1'], y1=results['y1'], loss=results['loss'])
        print(f"All results saved to {args.output}_all.npz")

    n_best = min(5, len(results))
    best = results[:n_best]
    print("\nTop 5 combinations:")
    for idx, (x0, x1, y1, loss) in enumerate(best):
        print(f"{idx+1}: x0={x0:.4f}, x1={x1:.4f}, y1={y1:.4f}, loss={loss:.4f}")

    # ---------- Saving LAT for the best ones ----------
    for idx, (x0, x1, y1, loss) in enumerate(best):
        p_prime = transform_p_3d(p, x0, x1, y1, smooth=args.smooth)
        lat_noisy_list = []
        for rep in range(args.scan_repeats):
            noisy_img = generate_noisy_image_3d(p_prime, img.shape)
            lat_noisy = propagate_3d(noisy_img, stim_rect)
            lat_noisy_list.append(lat_noisy)
        lat_noise_max = np.max(np.array(lat_noisy_list), axis=0)
        save_tiff_3d(lat_noise_max, f'data/lat_noise_best{idx+1}.tiff')

    # ---------- MAIN PLOT ----------
    print("\nPlotting all p'(p) curves...")
    p_vals = np.linspace(0, 1, 200)
    plt.figure(figsize=(6, 6))
    for (x0, x1, y1, loss) in results:
        p_prime = transform_p_3d(p_vals, x0, x1, y1, smooth=args.smooth)
        plt.plot(p_vals, p_prime, color='gray', alpha=0.05, linewidth=0.5)
    for (x0, x1, y1, loss) in best:
        p_prime = transform_p_3d(p_vals, x0, x1, y1, smooth=args.smooth)
        plt.plot(p_vals, p_prime, color='red', alpha=0.5, linewidth=1.0)
    plt.xlabel('p')
    plt.ylabel("p'")
    plt.title(f"All p'(p) curves and top 5\nsmooth={args.smooth}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/all_pprime_curves.png')
    plt.show()
    print("Plot saved to data/all_pprime_curves.png")

if __name__ == "__main__":
    main()