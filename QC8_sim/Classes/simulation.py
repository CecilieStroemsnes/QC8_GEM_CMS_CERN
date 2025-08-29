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
    
    def simulate_acceptance(self, N=100_000, return_hit_xy=False, max_store=5000):
        """
        Simulate N muons starting at the top scintillator.
        Returns acceptance per GEM layer and optional hit coordinates.
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
                "acceptance": np.zeros(self.geom.n_layers),
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

        # Calculate acceptance per layer
        acceptance = hits / max(n_coinc, 1)
        return {
            "n_generated": n_generated,
            "n_coinc": n_coinc,
            "layer_hits": hits,
            "acceptance": acceptance,
            "hit_xy": hit_xy
        }
    
    def simulate_eta_acceptance(self, N=100_000):
        """
        Per-eta acceptance for each layer relative to the
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
            return {"totals": totals, "acc": np.zeros_like(totals, float), "n_coinc": 0}

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

        # Calculate acceptance per eta bin
        acc = totals / max(n_coinc, 1)  # <-- denominator is the same for all k
        return {"totals": totals, "acc": acc, "n_coinc": n_coinc}
    
    def track_layer_hits(self, N=100_000, seed=None):
        """
        Generate tracks + per-track hit mask across GEM layers for coincident events.
        Returns:
        dict with:
            x0,y0,z0,vx,vy,vz : arrays for coincident tracks (length n_coinc)
            hit_mask           : (n_coinc, L) boolean (True = inside layer trapezoid)
            n_coinc            : int
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        L = self.geom.n_layers

        # start positions at top scintillator
        x0 = rng.uniform(self.geom.scin_xmin, self.geom.scin_xmax, size=N)
        y0 = rng.uniform(self.geom.scin_ymin, self.geom.scin_ymax, size=N)
        z0 = np.full(N, self.geom.Z_TOP_SCIN)

        # directions
        vx, vy, vz = self.sample_cos2_directions(N)

        # bottom coincidence
        t_bot = (self.geom.Z_BOTTOM_SCIN - z0) / vz
        valid = t_bot > 0.0
        xb = x0 + vx*t_bot
        yb = y0 + vy*t_bot
        coinc = valid & self.geom.in_scintillator_xy(xb, yb)

        # keep coincident
        x0, y0, z0 = x0[coinc], y0[coinc], z0[coinc]
        vx, vy, vz = vx[coinc], vy[coinc], vz[coinc]
        n_coinc = x0.size
        if n_coinc == 0:
            return dict(x0=x0,y0=y0,z0=z0,vx=vx,vy=vy,vz=vz,
                        hit_mask=np.zeros((0, L), bool), n_coinc=0)

        # intersections for each layer
        layer_z = self.geom.layer_z
        # shape (n_coinc, L) times to reach each layer
        tL = (layer_z[None, :] - z0[:, None]) / vz[:, None]
        # positions at each layer
        XI = x0[:, None] + vx[:, None] * tL
        YI = y0[:, None] + vy[:, None] * tL
        # forward only
        fwd = tL > 0.0
        # inside polygon for each layer
        inside = self.geom.gem_path.contains_points(
            np.column_stack([XI.ravel(), YI.ravel()])
        ).reshape(n_coinc, L)
        hit_mask = inside & fwd
        return dict(x0=x0,y0=y0,z0=z0,vx=vx,vy=vy,vz=vz, hit_mask=hit_mask, n_coinc=n_coinc)

    def summarize_5of6_from_mask(self, hit_mask: np.ndarray) -> dict:
            """
            Given hit_mask (n_coinc, L) -> stats for tracks that hit exactly L-1 layers.
            Returns dict with:
            n_coinc, n_exactly_Lminus1, frac_exactly_Lminus1, missed_hist, sel_indices
            """
            nC, L = hit_mask.shape if hit_mask.ndim == 2 else (0, 0)
            if nC == 0:
                return {
                    "n_coinc": 0,
                    "n_exactly_Lminus1": 0,
                    "frac_exactly_Lminus1": 0.0,
                    "missed_hist": np.zeros(L, dtype=int),
                    "sel_indices": np.array([], dtype=int),
                }

            hits_per_track = hit_mask.sum(axis=1)                 # (n_coinc,)
            sel = np.flatnonzero(hits_per_track == (L - 1))       # tracks with exactly one miss
            n5 = sel.size
            frac5 = n5 / nC if nC else 0.0

            # which single layer was missed for those tracks
            missed_layer = np.argmax(~hit_mask[sel], axis=1) if n5 else np.array([], dtype=int)
            missed_hist = np.bincount(missed_layer, minlength=L)

            return {
                "n_coinc": int(nC),
                "n_exactly_Lminus1": int(n5),
                "frac_exactly_Lminus1": float(frac5),
                "missed_hist": missed_hist.astype(int),
                "sel_indices": sel.astype(int),
            }

    def compute_5of6_stats(self, N=100_000, seed=None) -> dict:
        """
        One-liner: run track_layer_hits and summarize 5-of-6 stats.
        """
        data = self.track_layer_hits(N=N, seed=seed)
        return self.summarize_5of6_from_mask(data["hit_mask"])

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