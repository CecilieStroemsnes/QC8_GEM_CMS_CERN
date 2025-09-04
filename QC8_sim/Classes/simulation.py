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
        top and bottom coincidence baseline.
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
        L, K = self.geom.n_layers, len(self.geom.eta_y) - 1 # L = number of layers, K = number of eta regions
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

        # intersections
        XI, YI = self._layer_intersections(dict(x0=x0,y0=y0,z0=z0,vx=vx,vy=vy,vz=vz))
        tL = (self.geom.layer_z[None, :] - z0[:, None]) / vz[:, None]
        fwd = tL > 0.0  

        # inside polygon for each layer
        inside = self.geom.gem_path.contains_points(
            np.column_stack([XI.ravel(), YI.ravel()])
        ).reshape(n_coinc, L)
        
        hit_mask = inside & fwd
        return dict(x0=x0,y0=y0,z0=z0,vx=vx,vy=vy,vz=vz, hit_mask=hit_mask, n_coinc=n_coinc)

    # -------------------
    # Helper for (XI,YI)
    # -------------------
    def _layer_intersections(self, data: dict):
        """
        Compute intersections (XI, YI) for all tracks at all GEM layers.
        Args: data dict with x0,y0,z0,vx,vy,vz arrays (same length).
        Returns: XI, YI arrays shaped (n_tracks, n_layers).
        """
        layer_z = self.geom.layer_z
        tL  = (layer_z[None, :] - data["z0"][:, None]) / data["vz"][:, None]
        XI  = data["x0"][:, None] + data["vx"][:, None] * tL
        YI  = data["y0"][:, None] + data["vy"][:, None] * tL
        return XI, YI

    # -------------------
    # Summaries
    # -------------------
    def summarize_5of6_from_mask(self, hit_mask: np.ndarray) -> dict:
        """
        Given hit_mask (n_coinc, L) -> stats for tracks that hit exactly L-1 layers.
        Returns dict with:
        n_coinc, n_exactly_Lminus1, frac_exactly_Lminus1, missed_hist, sel_indices
        """
        n_coinc, L = hit_mask.shape
        hits_per_track = hit_mask.sum(axis=1)
        sel = np.flatnonzero(hits_per_track == (L - 1))

        missed_hist = np.zeros(L, dtype=int)
        for k in sel:
            miss_layers = np.where(~hit_mask[k])[0]
            if miss_layers.size == 1:
                missed_hist[miss_layers[0]] += 1  # exactly one miss

        return dict(
            n_coinc=n_coinc,
            n_exactly_Lminus1=sel.size,
            frac_exactly_Lminus1=(sel.size / max(n_coinc, 1)),
            missed_hist=missed_hist,
            sel_indices=sel,
        )
    
    def summarize_5of6_in_eta(self, data: dict, target_eta: int) -> dict:
        """
        Return stats for tracks that hit exactly L-1 layers (5/6) AND
        have ≥1 hit in the requested eta band on any layer.
        Output:
        {
            n_coinc, n_exactly_Lminus1, frac_exactly_Lminus1,
            missed_hist (per-layer), sel_indices (track indices)
        }
        """
        geom = self.geom
        H = np.asarray(data["hit_mask"], bool)          # (n_coinc, L)
        n_coinc, L = H.shape
        if n_coinc == 0:
            return dict(
                n_coinc=0, n_exactly_Lminus1=0, frac_exactly_Lminus1=0.0,
                missed_hist=np.zeros(L, int), sel_indices=np.array([], dtype=int)
            )

        # per-track hit counts
        hits_per = H.sum(axis=1)
        sel_5 = (hits_per == (L - 1))

        # per-layer intersections → only evaluate eta where there's a hit
        layer_z = geom.layer_z                       # (L,)
        tL = (layer_z[None, :] - data["z0"][:, None]) / data["vz"][:, None]
        XI = data["x0"][:, None] + data["vx"][:, None] * tL
        YI = data["y0"][:, None] + data["vy"][:, None] * tL

        # Build per-layer eta indices only on hit layers (0 where no hit)
        ETA = np.zeros_like(H, dtype=int)
        for ell in range(L):
            sel_hit = H[:, ell]
            if np.any(sel_hit):
                ETA[sel_hit, ell] = geom.which_eta(XI[sel_hit, ell], YI[sel_hit, ell])

        # keep 5/6 tracks that have any-layer hit in target eta
        in_eta_any = (ETA == int(target_eta)).any(axis=1)
        keep = sel_5 & in_eta_any
        sel_indices = np.flatnonzero(keep)

        # per-layer missed histogram (exactly one miss per selected)
        missed_hist = np.zeros(L, dtype=int)
        if sel_indices.size:
            miss_layer = np.argmax(~H[sel_indices], axis=1)   # 0..L-1
            for j in miss_layer:
                missed_hist[j] += 1

        n5 = sel_indices.size
        frac = n5 / max(n_coinc, 1)

        return dict(
            n_coinc=int(n_coinc),
            n_exactly_Lminus1=int(n5),
            frac_exactly_Lminus1=float(frac),
            missed_hist=missed_hist,
            sel_indices=sel_indices,
        )

    def summarize_5_and_6_in_eta(self, data: dict, target_eta: int) -> dict:
        """
        From track_layer_hits output, we compute: 
            n5: number of tracks with 5 hits and >= 1 hit in target_eta
            n6: number of tracks with 6 hits and >= 1 hit in target_eta
            n5or6: n5 + n6
            frac5: n5 / n5or6 (efficiency of 5-of-6 in target_eta)
        """
        geom = self.geom
        H = np.asarray(data["hit_mask"], bool)           # (n_coinc, L)
        n_coinc, L = H.shape if H.size else (0, geom.n_layers)

        # Empty / no-coincidence guard
        if n_coinc == 0 or H.size == 0:
            return dict(target_eta=int(target_eta), n_coinc=int(n_coinc),
                        n5=0, n6=0, n5or6=0, frac5=np.nan,
                        missed_hist_5=np.zeros(L, dtype=int))

        # Per-track hit counts and selections
        hits_per = H.sum(axis=1)
        sel5 = (hits_per == (L - 1))
        sel6 = (hits_per == L)
        sel56 = sel5 | sel6
        if not np.any(sel56):
            return dict(target_eta=int(target_eta), n_coinc=int(n_coinc),
                        n5=0, n6=0, n5or6=0, frac5=np.nan,
                        missed_hist_5=np.zeros(L, dtype=int))

        # Compute intersections once (uses the common helper)
        XI, YI = self._layer_intersections(data)

        # Determine, for each track, whether ANY of its hits is in target_eta
        eta_hit_any = np.zeros(n_coinc, dtype=bool)
        rows, cols = np.where(H)  # only where a hit exists
        if rows.size:
            eta_idx = geom.which_eta(XI[rows, cols], YI[rows, cols])  # 1..8 (0 if outside)
            in_target = (eta_idx == int(target_eta))
            # mark track rows that have at least one target-η hit
            eta_hit_any[rows[in_target]] = True

        # Restrict to tracks that are 5/6 or 6/6 AND have ≥1 hit in target η
        n5 = int(np.count_nonzero(sel5 & eta_hit_any))
        n6 = int(np.count_nonzero(sel6 & eta_hit_any))
        n56 = n5 + n6
        frac5 = (n5 / n56) if n56 > 0 else np.nan

        # For 5-hit tracks in this eta: which single layer was missed?
        missed_hist_5 = np.zeros(L, dtype=int)
        idx5 = np.flatnonzero(sel5 & eta_hit_any)
        if idx5.size:
            # exactly one False per row → the missed layer index
            miss_layer = np.argmax(~H[idx5], axis=1)  # 0..L-1
            for j in miss_layer:
                missed_hist_5[j] += 1

        return dict(target_eta=int(target_eta), n_coinc=int(n_coinc),
                    n5=n5, n6=n6, n5or6=n56, frac5=frac5, missed_hist_5=missed_hist_5) 
    
    def summarize_5_of_6_miss_axis(self, data: dict, tol: float = 1e-9) -> dict:
        geom = self.geom
        H = np.asarray(data["hit_mask"], bool)     # (n_coinc, L)
        n_coinc, L = H.shape
        x0, y0, z0 = data["x0"], data["y0"], data["z0"]
        vx, vy, vz = data["vx"], data["vy"], data["vz"]

        hits_per = H.sum(axis=1)
        idx5 = np.flatnonzero(hits_per == (L - 1))

        cats = ["X-left", "X-right", "Y-below", "Y-above"]
        if idx5.size == 0:
            K = (len(getattr(geom, "eta_y", [])) - 1) if hasattr(geom, "eta_y") else 0
            return {
                "n5": 0, "n_coinc": int(n_coinc),
                "n_classified": 0, "n_unclassified": 0,
                "counts": {c: 0 for c in cats},
                "per_layer": {c: np.zeros(L, dtype=int) for c in cats},
                "per_eta": {c: np.zeros(K, dtype=int) for c in cats},
                "indices": {c: np.array([], dtype=int) for c in cats},
            }

        layer_z = geom.layer_z
        tL = (layer_z[None, :] - z0[:, None]) / vz[:, None]
        XI = x0[:, None] + vx[:, None] * tL
        YI = y0[:, None] + vy[:, None] * tL

        y_min, y_max = geom.y_bounds()
        K = len(geom.eta_y) - 1  # number of η bands

        counts    = {c: 0 for c in cats}
        per_layer = {c: np.zeros(L, int) for c in cats}
        per_eta   = {c: np.zeros(K, int) for c in cats}
        indices   = {c: [] for c in cats}
        n_classified = 0
        n_unclassified = 0

        for k in idx5:
            miss_layers = np.where(~H[k])[0]
            if miss_layers.size != 1:
                n_unclassified += 1
                continue
            m = int(miss_layers[0])

            t = tL[k, m]
            if not np.isfinite(t) or (t <= 0):
                n_unclassified += 1
                continue

            xi = XI[k, m]
            yi = YI[k, m]

            # Decide category
            if yi < y_min - tol:
                cat = "Y-below"
            elif yi > y_max + tol:
                cat = "Y-above"
            else:
                xl, xr = geom.x_limits_at_y(yi)
                if xl is None or xr is None:
                    n_unclassified += 1
                    continue
                if xi < xl - tol:
                    cat = "X-left"
                elif xi > xr + tol:
                    cat = "X-right"
                else:
                    n_unclassified += 1
                    continue

            # Map y → η index (1..K), then tally 0-based bin
            eta_idx = geom.eta_index_from_y(yi)
            if 1 <= eta_idx <= K:
                per_eta[cat][eta_idx - 1] += 1

            counts[cat] += 1
            per_layer[cat][m] += 1
            indices[cat].append(k)
            n_classified += 1

        indices = {c: np.asarray(v, dtype=int) for c, v in indices.items()}

        return {
            "n5": int(idx5.size),
            "n_coinc": int(n_coinc),
            "n_classified": int(n_classified),
            "n_unclassified": int(n_unclassified),
            "counts": counts,
            "per_layer": per_layer,
            "per_eta": per_eta,       # NEW: category -> (K,) counts by η band
            "indices": indices,
        }

    def summarize_5v6_in_eta_strict_layer(self, data: dict, target_eta: int, layer_index: int) -> dict:
        """
        Consistent per-layer η efficiency:
        - n6: 6/6 hits AND layer_index hit is in target_eta
        - n5: 5/6 hits AND the ONLY miss is exactly layer_index,
                AND the track has ≥1 hit in target_eta on any (other) layer.

        Returns:
        {
            "n_coinc": int,
            "layer_index": int,
            "target_eta": int,
            "n5": int,
            "n6": int,
            "ratio_5_over_5p6": float,  # n5 / (n5 + n6)
            "sel5_indices": np.ndarray,
            "sel6_indices": np.ndarray,
        }
        """
        geom = self.geom
        H = np.asarray(data["hit_mask"], bool)          # (n_coinc, L)
        n_coinc, L = H.shape
        hits_per_track = H.sum(axis=1)

        # Recompute per-layer intersections so we can evaluate η only at actual hits
        layer_z = geom.layer_z
        tL  = (layer_z[None, :] - data["z0"][:, None]) / data["vz"][:, None]  # (n_coinc, L)
        XI  = data["x0"][:, None] + data["vx"][:, None] * tL
        YI  = data["y0"][:, None] + data["vy"][:, None] * tL

        # ETA matrix only where there is a hit; 0 elsewhere
        ETA = np.zeros_like(H, dtype=int)
        for ell in range(L):
            sel_hit = H[:, ell]
            if np.any(sel_hit):
                ETA[sel_hit, ell] = geom.which_eta(XI[sel_hit, ell], YI[sel_hit, ell])

        # Tracks with ≥1 hit in the target η (any layer)
        in_eta_any = (ETA == target_eta).any(axis=1)

        # --- n6: all layers hit AND the hit on the requested layer is in target η
        sel6 = (hits_per_track == L) & (ETA[:, layer_index] == target_eta)

        # --- n5: miss exactly that layer, and still have at least one hit in target η
        # exactly one miss overall (5/6) AND that miss is the requested layer
        miss_this_layer = ~H[:, layer_index]
        sel5 = (hits_per_track == (L - 1)) & miss_this_layer & in_eta_any

        n5 = int(np.count_nonzero(sel5))
        n6 = int(np.count_nonzero(sel6))
        denom = n5 + n6
        ratio = (n5 / denom) if denom > 0 else float("nan")

        return {
            "n_coinc": int(n_coinc),
            "layer_index": int(layer_index),
            "target_eta": int(target_eta),
            "n5": n5,
            "n6": n6,
            "ratio_5_over_5p6": ratio,
            "sel5_indices": np.flatnonzero(sel5),
            "sel6_indices": np.flatnonzero(sel6),
        }

    def summarize_5v6_in_eta_top_strict(self, data: dict, target_eta: int):
        return self.summarize_5v6_in_eta_strict_layer(data, target_eta, layer_index=self.geom.n_layers - 1)

    def summarize_5v6_in_eta_bottom_strict(self, data: dict, target_eta: int):
        return self.summarize_5v6_in_eta_strict_layer(data, target_eta, layer_index=0)

    def _classify_miss_reason_at_layer(self, xi: float, yi: float, tol: float = 1e-9):
        """
        Classify why a point (xi, yi) at the constrained layer would miss the GEM:
        - 'X-left'  if xi < x_left(yi) - tol (and yi within [ymin, ymax])
        - 'X-right' if xi > x_right(yi) + tol (and yi within [ymin, ymax])
        - 'Y-below' if yi < ymin - tol
        - 'Y-above' if yi > ymax + tol
        - None      if inside polygon (or numerically ambiguous)
        """
        geom = self.geom
        y_min, y_max = geom.y_bounds()

        if yi < y_min - tol:
            return "Y-below"
        if yi > y_max + tol:
            return "Y-above"

        xl, xr = geom.x_limits_at_y(yi)
        if xl is None or xr is None:
            return None
        if xi < xl - tol:
            return "X-left"
        if xi > xr + tol:
            return "X-right"
        return None


    def summarize_5v6_in_eta_strict_layer_axis(self, data: dict, target_eta: int, layer_index: int, axis: str, tol: float = 1e-9):
        """
        Strict per-layer η efficiency split by miss axis (X or Y):

        n6:  6/6 tracks AND the (layer_index) hit is in target_eta
        n5:  5/6 tracks AND the ONLY miss is exactly (layer_index),
            AND the track has ≥1 hit in target_eta on some other layer,
            AND the miss reason at (layer_index) belongs to the selected axis:
            axis='X' → {X-left, X-right}
            axis='Y' → {Y-below, Y-above}

        Returns:
        {
            "n_coinc": int,
            "layer_index": int,
            "target_eta": int,
            "axis": "X"|"Y",
            "n5": int,
            "n6": int,
            "ratio_5_over_5p6": float,
            "n5_breakdown": dict,  # only categories for that axis
            "sel5_indices": np.ndarray,
            "sel6_indices": np.ndarray,
        }
        """
        geom = self.geom
        H = np.asarray(data["hit_mask"], bool)  # (n_coinc, L)
        if H.size == 0:
            L = geom.n_layers
            return dict(n_coinc=0, layer_index=int(layer_index), target_eta=int(target_eta),
                        axis=str(axis), n5=0, n6=0, ratio_5_over_5p6=float("nan"),
                        n5_breakdown={}, sel5_indices=np.array([], dtype=int), sel6_indices=np.array([], dtype=int))

        n_coinc, L = H.shape
        hits_per_track = H.sum(axis=1)

        # Intersections at each layer
        layer_z = geom.layer_z
        tL = (layer_z[None, :] - data["z0"][:, None]) / data["vz"][:, None]
        XI = data["x0"][:, None] + data["vx"][:, None] * tL
        YI = data["y0"][:, None] + data["vy"][:, None] * tL

        # η only at real hits
        ETA = np.zeros_like(H, dtype=int)
        for ell in range(L):
            sel_hit = H[:, ell]
            if np.any(sel_hit):
                ETA[sel_hit, ell] = geom.which_eta(XI[sel_hit, ell], YI[sel_hit, ell])

        # At least one hit in target η on any layer
        in_eta_any = (ETA == int(target_eta)).any(axis=1)

        # n6: all layers hit AND constrained layer is in target η
        sel6 = (hits_per_track == L) & (ETA[:, layer_index] == int(target_eta))
        sel6_idx = np.flatnonzero(sel6)
        n6 = int(sel6_idx.size)

        # n5 candidates: miss exactly the constrained layer, 5/6 overall, and in_eta_any
        miss_this_layer = ~H[:, layer_index]
        sel5_cand = (hits_per_track == (L - 1)) & miss_this_layer & in_eta_any
        sel5_rows = np.flatnonzero(sel5_cand)

        # Classify miss reason at the constrained layer
        want_x = (str(axis).upper() == "X")
        want_y = (str(axis).upper() == "Y")
        breakdown = {"X-left": 0, "X-right": 0} if want_x else {"Y-below": 0, "Y-above": 0}

        keep5_mask = np.zeros_like(sel5_cand, dtype=bool)
        for r in sel5_rows:
            xi_m, yi_m = XI[r, layer_index], YI[r, layer_index]
            reason = self._classify_miss_reason_at_layer(xi_m, yi_m, tol=tol)
            if want_x and reason in ("X-left", "X-right"):
                breakdown[reason] += 1
                keep5_mask[r] = True
            elif want_y and reason in ("Y-below", "Y-above"):
                breakdown[reason] += 1
                keep5_mask[r] = True

        sel5_idx = np.flatnonzero(keep5_mask)
        n5 = int(sel5_idx.size)

        denom = n5 + n6
        ratio = (n5 / denom) if denom > 0 else float("nan")

        # Trim empty keys from breakdown
        breakdown = {k: v for k, v in breakdown.items() if v > 0}

        return {
            "n_coinc": int(n_coinc),
            "layer_index": int(layer_index),
            "target_eta": int(target_eta),
            "axis": "X" if want_x else "Y",
            "n5": n5,
            "n6": n6,
            "ratio_5_over_5p6": float(ratio),
            "n5_breakdown": breakdown,
            "sel5_indices": sel5_idx,
            "sel6_indices": sel6_idx,
        }


    def summarize_5v6_in_eta_top_strict_axis(self, data: dict, target_eta: int, axis: str, tol: float = 1e-9):
        return self.summarize_5v6_in_eta_strict_layer_axis(
            data, target_eta=int(target_eta), layer_index=self.geom.n_layers - 1, axis=axis, tol=tol
        )


    def summarize_5v6_in_eta_bottom_strict_axis(self, data: dict, target_eta: int, axis: str, tol: float = 1e-9):
        return self.summarize_5v6_in_eta_strict_layer_axis(
            data, target_eta=int(target_eta), layer_index=0, axis=axis, tol=tol
        )


# Count (x,y) samples per η per layer
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