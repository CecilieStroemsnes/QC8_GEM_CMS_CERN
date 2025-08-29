# simulation.py
from __future__ import annotations
from .geometry import ME0_Geometry

import numpy as np

class GEMTrajectorySimulator:
    """
    Generate straight muon tracks with a cos²(θ) angular distribution.
    Require coincidence between top and bottom scintillators.
    """

    def __init__(self, geometry: ME0_Geometry, seed=None):
        self.geom = geometry
        self.rng = np.random.default_rng(seed)

    def sample_cos2_directions(self, n):
        """
        Sample unit vectors (v_x, v_y, v_z) with cos²(θ) angular distribution.
        """

        # Sample cos(θ) distribution for downward-going particles
        U = self.rng.random(n)
        mu = U ** (1.0/3.0)  # cos(theta) in [0,1]
        phi = 2 * np.pi * self.rng.random(n)

        # Convert to Cartesian coordinates
        sin_theta = np.sqrt(1.0 - mu * mu)
        v_x = sin_theta * np.cos(phi)
        v_y = sin_theta * np.sin(phi)
        v_z = -mu  # downward-going
        return v_x, v_y, v_z
    
    def simulate_efficiency(self, N=100_000, return_hit_xy=False, max_store=5000):
        """
        Simulate N muons starting at the top scintillator.
        Returns efficiencies per GEM layer and optional hit coordinates.
        """
        # 1) Sample starting positions uniformly on the top scintillator
        x0 = self.rng.uniform(self.geom.scin_xmin, self.geom.scin_xmax, size=N)
        y0 = self.rng.uniform(self.geom.scin_ymin, self.geom.scin_ymax, size=N)
        z0 = np.full(N, self.geom.Z_TOP_SCIN) # Fixed z at top scintillator

        # 2) Sample particle directions
        v_x, v_y, v_z = self.sample_cos2_directions(N)

        # 3) Require bottom scintillator coincidence
        t_bot = (self.geom.Z_BOTTOM_SCIN - z0) / v_z # time to reach bottom scintillator
        valid = t_bot > 0.0 # only forward-going
        x_bot = x0 + v_x * t_bot
        y_bot = y0 + v_y * t_bot
        coinc = valid & self.geom.in_scintillator_xy(x_bot, y_bot) # Check if hits bottom scintillator

        # Keep only coincident events
        x0, y0, z0 = x0[coinc], y0[coinc], z0[coinc]
        v_x, v_y, v_z = v_x[coinc], v_y[coinc], v_z[coinc]
        n_generated = N # Total number of generated particles/events
        n_coinc = x0.size # Number of coincident events

        # Initialize hit counters and storage for hit coordinates
        hits = np.zeros(self.geom.n_layers, dtype=int)
        hit_xy = [None] * self.geom.n_layers if return_hit_xy else None

        # If no coincident events, return early
        if n_coinc == 0:
            return {
                "n_generated": n_generated,
                "n_coinc": 0,
                "layer_hits": hits,
                "efficiency": np.zeros(self.geom.n_layers),
                "hit_xy": hit_xy
            }

        # 4) Intersections with each GEM layer
        for i, z_layer in enumerate(self.geom.layer_z):
            t = (z_layer - z0) / v_z # time to reach current layer
            forward = t > 0.0 # only forward-going intersections
            if not np.any(forward):
                continue

            # Calculate hit positions at the layer
            x_hit = x0[forward] + v_x[forward] * t[forward]
            y_hit = y0[forward] + v_y[forward] * t[forward]

            # Check if hits are inside the GEM trapezoid
            inside = self.geom.gem_path.contains_points(np.column_stack([x_hit, y_hit]))
            hits[i] = np.count_nonzero(inside)

            # Store hit coordinates if requested
            if return_hit_xy:
                sel = np.flatnonzero(inside)
                if sel.size > 0:
                    sub = sel[:min(max_store, sel.size)]
                    hit_xy[i] = (x_hit[sub], y_hit[sub])
                else:
                    hit_xy[i] = (np.array([]), np.array([]))

        # Calculate efficiency per layer
        efficiency = hits / max(n_coinc, 1)
        return {
            "n_generated": n_generated,
            "n_coinc": n_coinc,
            "layer_hits": hits,
            "efficiency": efficiency,
            "hit_xy": hit_xy
        }
    
    def simulate_eta_efficiency(self, N=100_000):
        """
        Per-eta efficiencies for each layer relative to the
        top ∧ bottom coincidence baseline.
        """

        if not hasattr(self.geom, "which_eta"):
            raise RuntimeError("Eta layout not defined. Call geom.enable_default_me0_eta().")

        # 1) Sample starting positions and directions
        x0 = self.rng.uniform(self.geom.scin_xmin, self.geom.scin_xmax, size=N)
        y0 = self.rng.uniform(self.geom.scin_ymin, self.geom.scin_ymax, size=N)
        z0 = np.full(N, self.geom.Z_TOP_SCIN)
        vx, vy, vz = self.sample_cos2_directions(N)

        # 2) require bottom scintillator coincidence
        t_bot = (self.geom.Z_BOTTOM_SCIN - z0) / vz
        valid = t_bot > 0.0
        xb, yb = x0 + vx*t_bot, y0 + vy*t_bot
        coinc = valid & self.geom.in_scintillator_xy(xb, yb)

        # Keep only coincident events
        x0, y0, z0 = x0[coinc], y0[coinc], z0[coinc]
        vx, vy, vz = vx[coinc], vy[coinc], vz[coinc]
        n_coinc = x0.size

        # Initialize totals array for each layer and eta bin
        L, K = self.geom.n_layers, 8 # L = number of layers, K = number of eta regions
        totals = np.zeros((L, K), dtype=int)

        # If no coincident events, return early
        if n_coinc == 0:
            return {"totals": totals, "eff": np.zeros_like(totals, float), "n_coinc": 0}

        # 3) Intersections with each GEM layer and eta counting
        for l, z_layer in enumerate(self.geom.layer_z):
            t = (z_layer - z0) / vz
            xh = x0 + vx*t
            yh = y0 + vy*t

            # Check if hits are inside the GEM trapezoid
            inside = self.geom.gem_path.contains_points(np.column_stack([xh, yh]))
            if not np.any(inside):
                continue

            # Determine eta indices for hits inside the GEM
            eta_idx = self.geom.which_eta(xh[inside], yh[inside])  # 1..8
            for k in range(1, K+1):
                totals[l, k-1] = np.count_nonzero(eta_idx == k)

        # Calculate efficiencies per eta bin
        eff = totals / max(n_coinc, 1)  # <-- denominator is the same for all k
        return {"totals": totals, "eff": eff, "n_coinc": n_coinc}

# Utility: count (x,y) samples per η per layer
def count_hits_by_eta(geom: ME0_Geometry, hit_xy_layer_list):
    """
    hit_xy_layer_list: same structure as result['hit_xy'] (list of (x,y) per layer)
    Returns: dict with 'counts' (shape n_layers x 8) and 'totals' (per layer)
    """
    nL = len(hit_xy_layer_list)
    counts = np.zeros((nL, 8), dtype=int)
    totals = np.zeros(nL, dtype=int)
    for i, (x, y) in enumerate(hit_xy_layer_list):
        if x is None or x.size == 0:
            continue
        eta = geom.which_eta(x, y)
        totals[i] = x.size
        for k in range(1, 9):
            counts[i, k-1] = np.count_nonzero(eta == k)
    return {"counts": counts, "totals": totals}