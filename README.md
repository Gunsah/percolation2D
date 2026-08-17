# Percolation
A computational pipeline for simulating wave propagation in 2D/3D tissues using percolation theory

## Prerequisites & Installation

### 1. Install Conda
Ensure you have a Conda distribution installed (e.g., [Miniforge](https://conda-forge.org/download/), Miniconda, or Anaconda). 

* **For Apple Silicon (ARM) Mac users:** It is highly recommended to use Miniforge. Download the `Miniforge3-MacOSX-arm64.sh` script, install it by running `bash Miniforge3-MacOSX-arm64.sh` in your terminal, and then restart your terminal.

*(Optional) Update conda to the latest version:*
```bash
conda update -n base -c conda-forge conda
```

### 2. Clone the Repository
Clone the project from GitHub or download the source files. Ensure the following required files and directories are present in your project folder:
- `/in` (directory with an example input file)
- `/utils`
- `bfs_propagate.cpp`
- `scan_3D.py`
- `setup.py`
- `gena.py`
- `graph.py`
- `config.json`
- `intensities.json`
- `environment.yml`
- `README.md`

### 3. Set Up the Environment
All necessary dependencies are already managed within the `environment.yml` file. To automatically install all required packages and create the environment, run:

```bash
conda env create -f environment.yml
conda activate bfs-propagate
```

## Usage

### 1. Build C++ Extensions
To significantly improve computational performance, compile the C++ extensions using `setup.py`. While the pipeline will execute without this step, the simulation will be substantially slower.

```bash
python setup.py build_ext --inplace
```

### 2. Run the Simulation
The main analysis is executed via `scan_3D.py`. Ensure your conda environment is activated before running the script.

**Example Command:**
```bash
conda activate bfs-propagate
python scan_3D.py in/Image1.tif --config config.json --sigma 15 --x0-min 0.20 --x0-max 0.30 --x0-steps 5 --S 100 --scan-repeats 1 --smooth --output data/scan
```

### 3. Command-Line Arguments

| Argument | Description |
| :--- | :--- |
| `image_path` | Path to the input image. Must be an 8-bit binary TIFF (pixel values strictly 0 and 255). |
| `--config` | Path to the configuration file (JSON). Contains ROI coordinates and reference points used to compare the stochastic activation map with the original. |
| `--sigma` | Gaussian blur standard deviation. A value of `15` to `30` is empirically tuned for images of ~1275×1275 µm. |
| `--x0-min`, `--x0-max` | Search bounds for the percolation threshold ($x_0$). Empirical values typically fall between `0.24` and `0.26`, but may approach `0.60` for images with a high matrix fraction. |
| `--x0-steps` | Number of intermediate points in the $x_0$ search grid. Keep this relatively low to optimize search time. |
| `--S` | Number of random points sampled to fit the $p(p')$ curve. The curve is anchored at $(x_0, p_c)$ (where $p_c \approx 0.5927$ is the theoretical site percolation threshold for a 2D square lattice without diagonal connections) and $(1, 1)$. Increasing `S` (recommended: `30`–`100`) improves the curve fit. |
| `--scan-repeats` | Number of stochastic realizations (noise generations) per point. `1` is usually sufficient if `S` is large. Increase to `3`–`5` for higher stochastic robustness. |
| `--smooth` | Flag to apply smoothing to the generated trajectories. |
| `--output` | Directory path for saving the output data and plots. |

### 4. Output and Validation
Upon completion, the script outputs the optimal percolation threshold ($x_0$) to the terminal and generates the following files in the specified output directory:
* **`all_pprime_curves.png`**: Visualizes the explored $p(p')$ curves. Gray lines represent all sampled curves, while the top 5 best-fitting curves are highlighted in red.
* **Wave Trajectories**: Saved as image files/stacks.
* **lat_original.tiff**: Original LAT map

**Validation:** 
To validate the results, visually compare the `LAT_original` and `LAT_noise` trajectories. For convenience, you can align them using Fiji/ImageJ (e.g., via `Image > Stacks > Images to Stack`) to ensure the simulated wave propagation adequately matches the original experimental data. Iteratively tune the $x_0$ bounds and `S` parameter until optimal trajectory alignment is achieved.

## Pre-computation and Full Parameter Scan

For correct execution of the full parameter scan, it is strictly required to generate the baseline activation map using `gena.py` before running the exhaustive search.

### 1. Initial Estimation (Fast Run)
First, perform a fast run to compute the baseline original local activation time (`LAT_original`). Parameters `--S 1` and `--x0-steps 1` are used here solely for computational speed:
```bash
python scan_3D.py in/Image1.tif --config config.json --sigma 15 --x0-min 0.20 --x0-max 0.30 --x0-steps 1 --S 1 --scan-repeats 1 --smooth --output data/scan
```

### 2. Generate Baseline Activation Map (`gena.py`)
Next, run `gena.py` using the original image and the generated `LAT_original` file:
```bash
python gena.py in/Image1.tif data/lat_original.tiff 0 1024 0 100 1 1 100 15 15 15 --only_nonzero
```

**Positional Arguments Explanation:**
* `0 1024 0 100 1 1`: Coordinates defining the wave propagation source. **Important:** These values must exactly match the coordinates specified in `config.json`. If you need to modify them, update `config.json` first and re-run Step 1.
* `100`: Number of sampled points. Do not change this value.
* `15 15 15`: Smoothing standard deviations. These must match the `--sigma` value used in `scan_3D.py`.

### 3. Full Parameter Scan
Once the baseline map is generated, run the full parameter scan with higher resolution:
```bash
python scan_3D.py in/Image1.tif --config config.json --sigma 15 --x0-min 0.20 --x0-max 0.30 --x0-steps 5 --S 100 --scan-repeats 1 --smooth --output data/scan
```

### 4. Statistical Repetition
For robust statistical analysis, it is highly recommended to repeat Steps 2 and 3 (`gena.py` and the full `scan_3D.py` run) **9–10 times** for each sample, saving all resulting outputs independently.

### Dataset Description (`/in` directory)
The `/in` directory contains 6 pre-processed sample images. 
* **Original physical size:** 1416 µm.
* **Preprocessing:** Images were segmented using Color Thresholding, downscaled to 1024×1024 pixels (without averaging or interpolation to preserve binary integrity), and exported as 8-bit TIFF files. These files have been validated and are ready for the pipeline.