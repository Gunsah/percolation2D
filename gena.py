import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate config.json with random points and blurred intensities.')
    parser.add_argument('image1', help='Path to 8-bit TIFF stack')
    parser.add_argument('image2', help='Path to 32-bit TIFF stack')
    parser.add_argument('rect', nargs=6, type=int, help='xmin xmax ymin ymax zmin zmax (half-open intervals)')
    parser.add_argument('N', type=int, help='Number of points (interpretation depends on mode)')
    parser.add_argument('sigma', nargs=3, type=float, help='Gaussian sigma for x y z axes')
    parser.add_argument('--output', default='config.json', help='Output JSON file (default: config.json)')
    parser.add_argument('--intensities', default='intensities.json', help='Output file for blurred intensities (default: intensities.json)')
    parser.add_argument('--plot_proj', help='Output image file for color-coded projection with points (e.g., projection.png)')
    parser.add_argument('--plot_hist', help='Output image file for histogram of blurred intensities (e.g., histogram.png)')
    parser.add_argument('--only_nonzero', action='store_true', help='Generate only points where both images are nonzero (cond1)')
    parser.add_argument('--min_half', action='store_true', help='Keep only half of points with smallest blurred intensities (works with --only_nonzero)')
    parser.add_argument('--max_half', action='store_true', help='Keep only half of points with largest blurred intensities (works with --only_nonzero)')
    args = parser.parse_args()

    # Check mutual exclusivity
    if args.min_half and args.max_half:
        raise ValueError("Cannot specify both --min_half and --max_half")
    if (args.min_half or args.max_half) and not args.only_nonzero:
        raise ValueError("--min_half/--max_half require --only_nonzero mode")
    if (args.min_half or args.max_half) and args.N < 2:
        raise ValueError("N must be at least 2 when using --min_half/--max_half")

    # Load images
    img1 = tifffile.imread(args.image1)
    img2 = tifffile.imread(args.image2)

    # Convert both images to 3D shape (depth, height, width)
    def to_3d(img):
        if img.ndim == 2:
            # (height, width) -> (1, height, width)
            return img[np.newaxis, ...]
        elif img.ndim == 3:
            return img
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}D")

    img1 = to_3d(img1)
    img2 = to_3d(img2)

    # Now compare shapes
    if img1.shape != img2.shape:
        print(f"Warning: image shapes differ after 3D conversion: img1 {img1.shape}, img2 {img2.shape}")
        # Determine minimum sizes along each dimension
        min_depth = min(img1.shape[0], img2.shape[0])
        min_height = min(img1.shape[1], img2.shape[1])
        min_width = min(img1.shape[2], img2.shape[2])
        # Crop both to the minimum sizes (taking the beginning)
        img1 = img1[:min_depth, :min_height, :min_width]
        img2 = img2[:min_depth, :min_height, :min_width]
        print(f"Cropped both images to common shape: {img1.shape}")

    depth, height, width = img1.shape

    # Rectangle bounds (half-open intervals)
    xmin, xmax, ymin, ymax, zmin, zmax = args.rect

    # Create rectangle mask
    x_inside = (np.arange(width) >= xmin) & (np.arange(width) < xmax)
    y_inside = (np.arange(height) >= ymin) & (np.arange(height) < ymax)
    z_inside = (np.arange(depth) >= zmin) & (np.arange(depth) < zmax)
    inside_mask = np.zeros((depth, height, width), dtype=bool)
    inside_mask[np.ix_(z_inside, y_inside, x_inside)] = True

    # Category masks
    cond1 = (img1 != 0) & (img2 != 0) & ~inside_mask
    cond2 = (img1 != 0) & (img2 == 0) & ~inside_mask
    cond3 = (img1 == 0) & (img2 == 0) & ~inside_mask

    # Blur img1
    sigma_reorder = (args.sigma[2], args.sigma[1], args.sigma[0])
    blurred = gaussian_filter(img1.astype(np.float32), sigma=sigma_reorder)

    # Helper function to select random points
    def select_random_points(mask, n):
        indices = np.flatnonzero(mask)
        chosen = np.random.choice(indices, size=n, replace=False)
        z, y, x = np.unravel_index(chosen, mask.shape)
        return [[int(x[i]), int(y[i]), int(z[i])] for i in range(n)]

    # Generate points
    if args.only_nonzero:
        if np.sum(cond1) < args.N:
            raise ValueError(f"cond1 has only {np.sum(cond1)} points, need {args.N}")
        candidate_points = select_random_points(cond1, args.N)
        candidate_intensities = [float(blurred[z, y, x]) / 255.0 for x, y, z in candidate_points]

        paired = list(zip(candidate_intensities, candidate_points))
        paired.sort(key=lambda p: p[0])

        if args.min_half:
            half = args.N // 2
            selected_pairs = paired[:half]
        elif args.max_half:
            half = args.N // 2
            selected_pairs = paired[-half:] if half > 0 else []
        else:
            selected_pairs = paired

        pos_map = {tuple(pt): i for i, pt in enumerate(candidate_points)}
        selected_pairs.sort(key=lambda p: pos_map[tuple(p[1])])

        intensities = [p[0] for p in selected_pairs]
        points = [p[1] for p in selected_pairs]
    else:
        for i, cond in enumerate([cond1, cond2, cond3], 1):
            if np.sum(cond) < args.N:
                raise ValueError(f"Condition {i} has only {np.sum(cond)} points, need {args.N}")
        points = select_random_points(cond1, args.N) + \
                 select_random_points(cond2, args.N) + \
                 select_random_points(cond3, args.N)
        intensities = [float(blurred[z, y, x]) / 255.0 for x, y, z in points]

    # Save config.json
    config = {"rect": args.rect, "points": points}
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)

    # Save intensities
    with open(args.intensities, 'w') as f:
        json.dump(intensities, f, indent=2)

    # Optional visualization
    if args.plot_proj or args.plot_hist:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

    if args.plot_proj:
        depth_map = np.full((height, width), -1, dtype=int)
        for z in range(depth):
            layer = img1[z] != 0
            depth_map[layer] = z
        cmap = plt.cm.jet
        norm = Normalize(vmin=0, vmax=max(1, depth-1))
        colored = np.zeros((height, width, 3))
        mask_valid = depth_map != -1
        if depth == 1:
            colored[mask_valid] = cmap(norm(0))[:3]
        else:
            colored[mask_valid] = cmap(norm(depth_map[mask_valid]))[:, :3]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(colored)
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, marker='x', color='red', s=30, linewidths=1)
        ax.set_title(f'Color-coded depth projection (z=0..{depth-1}) with {len(points)} points')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(args.plot_proj, dpi=150)
        plt.close()

    if args.plot_hist and intensities:
        plt.figure(figsize=(8, 5))
        plt.hist(intensities, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Blurred intensity (normalized)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of blurred intensities (N={len(intensities)})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.plot_hist, dpi=150)
        plt.close()
    elif args.plot_hist and not intensities:
        print("Warning: no intensities to plot, skipping histogram.")

if __name__ == '__main__':
    main()