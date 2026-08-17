"""
Plot p'(p) and CV(p) curves based on x0, x1, y1 parameters from a text file.
Supports curve smoothing by increasing the number of points (--points).
Saves two plots: p'(p) and CV(p).
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d

# CV(p') dependency table
CV_TABLE = {
    1.0: 1.043296818,
    0.9: 0.9843023462,
    0.8: 0.9134572347,
    0.7: 0.8083688814,
    0.65: 0.7314020905,
    0.64: 0.7028557733,
    0.63: 0.6717937969,
    0.62: 0.6351779324,
    0.61: 0.5755612658,
    0.6: 0.4929969779,
    0.5972: 0.0
}


def main():
    parser = argparse.ArgumentParser(description='Plot p\'(p) and CV(p) curves from a parameter file.')
    parser.add_argument('input', help='Path to the text file with lines in the format "x0=..., x1=..., y1=..."')
    parser.add_argument('--output-pp', default='pprime_curves.png',
                        help='Output file for the p\'(p) plot (default: pprime_curves.png)')
    parser.add_argument('--output-cv', default='cv_curves.png',
                        help='Output file for the CV(p) plot (default: cv_curves.png)')
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='Use smoothed interpolation for p\'(p) (PCHIP).')
    parser.add_argument('--no-smooth', dest='smooth', action='store_false',
                        help='Disable smoothing for p\'(p), use piecewise linear.')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Curve transparency (0-1).')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution of the saved plots.')
    parser.add_argument('--points', type=int, default=500,
                        help='Number of points along p to plot the curves (more = smoother).')
    parser.add_argument('--cv-interp', choices=['linear', 'cubic'], default='linear',
                        help='Interpolation type for CV(p\'): linear (default) or cubic (smoother).')
    args = parser.parse_args()

    # Read parameters
    params_list = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            params = parse_line(line)
            if params:
                params_list.append(params)
            else:
                print(f"Warning: unrecognized line: {line}")

    if not params_list:
        print("Error: no valid parameter lines found.")
        return

    print(f"Found {len(params_list)} parameter sets.")
    p_vals = np.linspace(0, 1, args.points)

    # Plot and save both graphs
    plot_pprime_curves(params_list, p_vals, args.smooth, args.alpha, args.output_pp, args.dpi)
    plot_cv_curves(params_list, p_vals, args.smooth, args.cv_interp, args.alpha, args.output_cv, args.dpi)

    # Optional: show plots on screen
    show = input("\nShow plots? (y/n): ").strip().lower()
    if show == 'y':
        # Rebuild for display
        p_vals_show = np.linspace(0, 1, 300)  # slightly fewer for speed
        plot_pprime_curves(params_list, p_vals_show, args.smooth, args.alpha, 'temp_pp.png', 72)
        plot_cv_curves(params_list, p_vals_show, args.smooth, args.cv_interp, args.alpha, 'temp_cv.png', 72)
        # Show both
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # p'(p)
        for x0, x1, y1 in params_list:
            p_prime_vals = np.array([transform_p(p, x0, x1, y1, smooth=args.smooth) for p in p_vals_show])
            ax1.plot(p_vals_show, p_prime_vals, color='gray', alpha=args.alpha, linewidth=0.8)
        ax1.plot([0,1], [0,1], 'k--', linewidth=1.5, label="p' = p")
        ax1.set_xlabel('p'); ax1.set_ylabel("p'"); ax1.set_title("p'(p)")
        ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3); ax1.legend()
        # CV(p)
        cv_interp = create_cv_interpolator(kind=args.cv_interp)
        for x0, x1, y1 in params_list:
            p_prime_vals = np.array([transform_p(p, x0, x1, y1, smooth=args.smooth) for p in p_vals_show])
            cv_vals = cv_interp(p_prime_vals)
            ax2.plot(p_vals_show, cv_vals, color='gray', alpha=args.alpha, linewidth=0.8)
        cv_identity = cv_interp(p_vals_show)
        ax2.plot(p_vals_show, cv_identity, 'k--', linewidth=1.5, label="CV(p) for p'=p")
        ax2.set_xlabel('p'); ax2.set_ylabel('CV'); ax2.set_title("CV(p)")
        ax2.grid(True, alpha=0.3); ax2.legend()
        plt.tight_layout()
        plt.show()
        # Remove temporary files
        import os
        os.remove('temp_pp.png')
        os.remove('temp_cv.png')

if __name__ == "__main__":
    main()