# Quality Control 8: Cosmic Ray efficiency simulation

This project simulates the detection efficiency of the ME0 GEM detector stack using cosmic ray muons.

It generates straight tracks with a $\cos^2(\theta)$ angular distribution, applies a top–bottom scintillator coincidence, and reports efficiencies per layer and per η region. It also gives a detailed “5-of-6 vs 6-of-6” study with strict per-layer and X/Y direction miss classifications.

## Features
- ME0 geometry, trapezoid-looking layers with configurable stack positions (high/low).
- Scintillators aligned over and under the stack.
- Coincidence requirement for the top and bottom scintillators.
- Calculation of per-layer and per-η region acceptance.
- 5-of-6 vs 6-of-6 efficiency:
    - 5/6: Means that the trajectory hits exactly 5 out of 6 layers
    - 6/6: Means that the trajectory hits exactly 6 out of 6 layers
    - Strict per-layer: Compares 5/6 that miss a specific layer vs 6/6 that hit that layer in target η.
    - Axis split (x/y): Classify 5/6 misses in x or y direction.

## Layout
- `Classes/`
  - `geometry.py` — ME0_Geometry (detector + scintillator layout)
  - `simulation.py` — GEMTrajectorySimulator (+ `tally_hits_by_eta` helper)
  - `plots.py` — Plots 
  - `__init__.py` — exports the public API
- `run.py` — Main script to run the simulation
- `Outputs` - Folder with all plots and summary text

## Requirements
- Used Python 3.9
- Packages: `pip install numpy matplotlib`

## Quickstart
Run with default settings (stack positioned high, number of muons N = 500.000, interactive plot save off):
```bash
python run.py 
```

Run with user-choosen stack position and number of muons
```bash
python run.py --position low --N 1000000
```

Enable interactive 3D/geometry windows (press s in the window to save):
```bash
python run.py --interactive
```

### Outputs:
- `stack_geometry.png` — Four-view geometry with η bands.
- `efficiency_hist.png` — Layer acceptance histogram (relative to coincidence).
- `Hit_maps.png` — Hit maps per layer (with η overlays).
- `3D_trajectories.png` — Sampled 3D tracks through the stack.
- `3D_trajectories_5of6.png` — 3D tracks for exactly 5/6 hits (global).
- `3D_trajectories_5of6_eta1.png` — 3D tracks for 5/6 with ≥1 hit in η=1.
- `3D_trajectories_5of6_eta8.png` — 3D tracks for 5/6 with ≥1 hit in η=8.
- `eta_eff_by_layer.png` — Per-η acceptance per layer.
- `x_occupancy_eta_layer*.png` — X-occupancy per η region and layer.
- `summary.txt` — Summary report:
  - Per-layer hits/acceptance
  - Per-η counts/acceptance
  - Global 5-of-6 summary 5-of-6 miss directions (X-left/right, Y-below/above) + per-η breakdown
  - Strict per-layer 5 vs 6 in η=1 and η=8, split by X/Y miss (efficiencies and counts)

#### What the “strict per-layer 5 vs 6” means

For a chosen η band and a specific layer (TOP or BOTTOM):
- 6-of-6 (n6): tracks that hit all layers and whose hit on the constrained layer lies in the target η.
- 5-of-6 (n5): tracks that hit exactly 5 layers, miss exactly the constrained layer, and still have ≥1 hit in the target η on other layers.
- Efficiency reported is n5/(n5+n6) for that η/layer selection.

We also split the 5-of-6 misses by axis:
- X-only: the constrained-layer intersection lies left/right of the GEM polygon at that x (X-left/X-right).
- Y-only: the constrained-layer intersection lies above/below the GEM y-span (Y-above/Y-below).

This allows you to compare edge-loss behavior in X versus Y.
  
#### Optional outputs:
In run.py (main), there is an interactive option for the stack geometry and the 3D trajectory plots.
These should be out-commented if used.

### Example output
<img width="543" height="580" alt="3D_trajectories" src="https://github.com/user-attachments/assets/528315f6-4aed-43f2-bd1b-6c4e12b04d9d" />

# Physics Notes - Cosmic Ray Angular Distribution

## Cosmic rays

For cosmic rays, we use a $cos^2(\theta)$ distribution for downward-going muons.


$\theta$ is the zenith angle ($\theta$ = 0 is straight down, and $\theta$ increases towards the horizon).

## Changing variables
To simplify, we use $\mu = cos(\theta)$, where

$\theta = 0 \Rightarrow \mu = 1$

$\theta = 90 \Rightarrow \mu = 0$

Then $d\mu = - sin(\theta) d\theta$

So the probability density function, $p(\theta) \propto cos^2(\theta) \cdot sin(\theta)$, using $cos(\theta)=\mu$ and $sin(\theta)d\theta=-d\mu$, becomes:

$p(\theta)d\theta \propto \mu^2 \cdot (-d\mu)$

$\Downarrow$

$p(\mu)d\mu \propto \mu^2 d\mu$

*(Minus just flips integration limits... ignored for probabilities)*

So now we have a simple form: 

$p(\mu) \propto \mu^2$, $\mu \in [0,1]$

## Normalizing

The probability density function, $p(\mu)\propto \mu^2$, must satisfy $\int p(x) dx = 1$ when integrating over all possible values. Since our relation is proportional to, we are missing a constant C. This constant is found by normalizing:

$\int_0^1 C \mu^2 d\mu = 1$

Where the integral is $C \cdot \frac{\mu^3}{3} |_0^1 = C \cdot \frac{1}{3}$

$\Rightarrow C = 3$

$\Rightarrow p(\mu)=3\mu^2, 0 \leq \mu \leq 1$

## Sampling using Cumulative Distribution Function

The cumulative distribution function $F(\mu)$ gives the probability that the random variable is less than or equal to $\mu$:

$F(\mu) = P (value \leq \mu) = \int_0^\mu p(t)dt$

So we integrate $p(\mu)$:
$F(/mu) = \int_0^\mu 3t^2 dt = [t^3]_0^\mu = \mu^3$

Now we set $F(\mu)=U$, so $U \sim Uniform(0,1)$

Then $\mu^3 = U \Rightarrow \mu = U^{1/3}$
