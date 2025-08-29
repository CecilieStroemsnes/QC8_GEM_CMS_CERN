# plots.py
from __future__ import annotations # incase of forward refs in type hints

from pathlib import Path as FsPath  # filesystem paths
from typing import Optional # for type hints (either str or None)

import matplotlib.pyplot as plt # plotting
import numpy as np # numerical operations
from matplotlib.lines import Line2D # for custom legend lines
from matplotlib.patches import Patch # for custom legend patches
from matplotlib.ticker import PercentFormatter # for y-axis percent formatting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 3D polygons
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar placement

from .geometry import ME0_Geometry # ME0 geometry class (from same folder)

class Plots:
    """
    Plotting for ME0 stack:
        - geometry views (with optional η overlays)
        - per-layer efficiency histograms
        - per-layer hit maps (with optional η overlays)
        - 3D trajectories (with optional η overlays)
        - per-η efficiency bar charts
        - per-layer x-occupancy by η
    """

    def __init__(self, geometry, output_dir: str | FsPath | None = None):
        self.geom = geometry
        self.output_dir = FsPath(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True) # ensure directory exists

    # ------------------------------------------
    # ------ Saver -----------------------------
    # ------------------------------------------

    def _save(self, filename: str | None):
        if self.output_dir and filename:
            plt.gcf().savefig(self.output_dir / filename, dpi=200, bbox_inches="tight")

    # ------------------------------------------
    # ------ Interactive save handler ----------
    # ------------------------------------------

    def _interactive_save(self, fig, filename, save_key="s", save_on_close=True, dpi=200):
        """
        Created a interactive figure where the user can rotate the 3D view. 
        Press `save_key` to save a snapshot to `filename`.
        If the window is closed without pressing `save_key`, save once on close if `save_on_close` is True.
        """
        out = (self.output_dir / filename) if self.output_dir else filename

        def do_save():
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            fig._saved_once = True
            print(f"[saved] {out}")

        def on_key(event):
            if event.key == save_key:
                do_save()

        def on_close(event):
            if save_on_close and not getattr(fig, "_saved_once", False):
                do_save()

        fig._saved_once = False
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)

    # ------------------------------------------
    # ------ Add 3D polygon patches ------------
    # ------------------------------------------
    # 
    def add_layer_patch(self, ax, x_poly, y_poly, z_val, fc, ec, alpha=0.25, lw=1.0):
        face = [[(x_poly[i], y_poly[i], z_val) for i in range(len(x_poly))]]
        coll = Poly3DCollection(face, facecolors=fc, edgecolors=ec,
                                linewidths=lw, alpha=alpha)
        ax.add_collection3d(coll)

    # ------------------------------------------
    # ---------- Single‑axis 3D draw -----------
    # ------------------------------------------

    # Works as a 3D snapshot for a chosen view
    def _draw_scene(self, ax, elev, azim,
                    fc_gem='lightskyblue', ec_gem='navy', alpha_gem=0.15,
                    fc_scin='lightcoral',  ec_scin='darkred', alpha_scin=0.20,
                    lw_gem=1.0, lw_scin=1.2, title=None,
                    show_eta=False, y_breaks=None, y_mm=None, mm_bottom=None, mm_top=None):

        # Step 1) Draw GEM layers as flat polygons at their z positions
        for zl in self.geom.layer_z:
            self.add_layer_patch(ax, self.geom.x_top, self.geom.y_top, zl,
                                 fc=fc_gem, ec=ec_gem, alpha=alpha_gem, lw=lw_gem)
            
        # Step 2) Draw scintillator planes as rectangles at their z positions
        scin_x = [-self.geom.scintillator_width/2,  self.geom.scintillator_width/2,
                   self.geom.scintillator_width/2, -self.geom.scintillator_width/2]
        scin_y = [0.0, 0.0, self.geom.scintillator_length, self.geom.scintillator_length]
        self.add_layer_patch(ax, scin_x, scin_y, self.geom.Z_TOP_SCIN,
                             fc=fc_scin, ec=ec_scin, alpha=alpha_scin, lw=lw_scin)
        self.add_layer_patch(ax, scin_x, scin_y, self.geom.Z_BOTTOM_SCIN,
                             fc=fc_scin, ec=ec_scin, alpha=alpha_scin, lw=lw_scin)

        # Step 3) Optional η overlays on all layers
        if show_eta:
            if hasattr(self.geom, "eta_polys"):
                eta_polys = self.geom.eta_polys
            else:
                _, eta_polys = self._eta_polys_from_args(
                    y_breaks=y_breaks, y_mm=y_mm, mm_bottom=mm_bottom, mm_top=mm_top
                )
            if eta_polys:
                self._add_eta_overlays_3d(
                    ax, eta_polys, self.geom.layer_z,
                    edgecolor='deeppink', lw=0.8, alpha=0.9
                )
        
        # Step 4) Cosmetics and labels (add view angles)
        if title:
            ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(0.0, self.geom.scintillator_length)
        ax.set_zlim(self.geom.Z_BOTTOM_SCIN - 0.05, self.geom.Z_TOP_SCIN + 0.05)
        ax.set_box_aspect((0.7, self.geom.scintillator_length,
                           (self.geom.Z_TOP_SCIN - self.geom.Z_BOTTOM_SCIN)))
        ax.view_init(elev=elev, azim=azim)

    # ------------------------------------------
    # ----- Single 3D view (interactive) -------
    # ------------------------------------------

    def plot_3d(
        self, elev=20, azim=-60,
        fc_gem='lightskyblue', ec_gem='navy',
        fc_scin='lightcoral', ec_scin='darkred',
        alpha_gem=0.15, alpha_scin=0.20,
        show_eta=False, y_breaks=None, y_mm=None, mm_bottom=None, mm_top=None,
        filename: str = "stack_geometry.png",
        interactive: bool = False,
        save_key: str = "s",
    ):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        self._draw_scene(
            ax, elev, azim,
            fc_gem=fc_gem, ec_gem=ec_gem, alpha_gem=alpha_gem,
            fc_scin=fc_scin, ec_scin=ec_scin, alpha_scin=alpha_scin,
            show_eta=show_eta, y_breaks=y_breaks, y_mm=y_mm,
            mm_bottom=mm_bottom, mm_top=mm_top
        )

        legend_patches = [
            Patch(facecolor=fc_gem, edgecolor=ec_gem, label="GEM layer", alpha=alpha_gem),
            Patch(facecolor=fc_scin, edgecolor=ec_scin, label="Scintillator", alpha=alpha_scin)
        ]
        if show_eta:
            legend_patches.append(Line2D([0],[0], color='deeppink', lw=1.2, label='η boundaries'))
        fig.legend(handles=legend_patches, loc="upper left")

        plt.tight_layout()

        if interactive:
            # rotate freely; press `save_key` to save; closing also saves if not already saved
            self._interactive_save(fig, filename, save_key=save_key, save_on_close=True)
            plt.show()
            plt.close(fig)
        else:
            self._save(filename)
            plt.close(fig)

    # ------------------------------------------
    # ---------- Four views in one figure ------
    # ------------------------------------------

    def four_views(
        self,
        figsize=(20, 5),
        views=None,
        fc_gem='lightskyblue', ec_gem='navy',
        fc_scin='lightcoral',  ec_scin='darkred',
        alpha_gem=0.15, alpha_scin=0.20,
        show_eta=True, y_breaks=None, y_mm=None, mm_bottom=None, mm_top=None,
        eta_color='deeppink',
        filename: str | None = "Stack Geometry.png",
        interactive: bool = False,
        save_key: str = "s",
    ):
        # Default camera presets
        if views is None:
            views = [("Side", 10, 0), ("Back", 10, 90), ("Top", 90, 0), ("Angled", 20, -60)]
        if len(views) != 4:
            raise ValueError("`views` must have exactly 4 entries of (title, elev, azim).")

        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]

        for ax, (title, elev, azim) in zip(axes, views):
            self._draw_scene(
                ax, elev, azim,
                fc_gem=fc_gem, ec_gem=ec_gem, alpha_gem=alpha_gem,
                fc_scin=fc_scin, ec_scin=ec_scin, alpha_scin=alpha_scin,
                title=title,
                show_eta=show_eta, y_breaks=y_breaks, y_mm=y_mm,
                mm_bottom=mm_bottom, mm_top=mm_top
            )

        handles = [
            Patch(facecolor=fc_gem,  edgecolor=ec_gem,  label="GEM layer",    alpha=alpha_gem),
            Patch(facecolor=fc_scin, edgecolor=ec_scin, label="Scintillator", alpha=alpha_scin),
        ]
        if show_eta:
            handles.append(Line2D([0], [0], color=eta_color, lw=1.2, label='η boundaries'))
        fig.legend(handles=handles, loc="upper left")

        plt.tight_layout()

        # Save / interactive save
        if interactive:
            if filename is None:
                filename = "Stack Geometry.png"
            self._interactive_save(fig, filename, save_key=save_key, save_on_close=True)
            plt.show()
            plt.close(fig)
        else:
            self._save(filename)
            plt.close(fig)

    # ------------------------------------------
    # ---------- Plot efficiency ----------
    # ------------------------------------------
    def plot_efficiency_histogram(
        self,
        result,
        filename: Optional[str] = "efficiency_hist.png",
        ylim: tuple[float, float] | None = (0.9, 1.0),
        as_percent: bool = False,
        annotate: bool = True,
    ):
        layers = np.arange(1, self.geom.n_layers + 1)
        eff = np.asarray(result["efficiency"], float)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        bars = ax.bar(layers, eff if not as_percent else eff*100,
                    color="#4da3ff", edgecolor="navy", alpha=0.85)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Efficiency" + (" (%)" if as_percent else ""), fontsize=12)
        ax.set_title("Detection Efficiency per GEM Layer", fontsize=14, pad=12, weight="bold")

        if ylim is not None:
            lo, hi = (ylim if not as_percent else (ylim[0]*100, ylim[1]*100))
            ax.set_ylim(lo, hi)
        else:
            # auto with a touch of headroom
            yvals = eff if not as_percent else eff*100
            hi = float(np.nanmax(yvals)) if yvals.size else (100 if as_percent else 1)
            ax.set_ylim(0, hi * 1.05)

        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{i}" for i in layers], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle=":")

        if annotate:
            for b in bars:
                height = b.get_height()
                ax.annotate(f"{height*100:.1f}%",
                            xy=(b.get_x() + b.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, color="black")

        fig.tight_layout()
        self._save(filename)
        plt.close(fig) 

    # ------------------------------------------
    # ---------- Per-layer hit maps ----------
    # ------------------------------------------

    def plot_hit_maps(self, result, show_eta=True, filename: str | None = "Hit_maps.png"):
        n = self.geom.n_layers
        fig, axs = plt.subplots(1, n, figsize=(3*n, 5), sharey=True)

        for i, (x, y) in enumerate(result["hit_xy"], start=1):
            ax = axs[i-1]
            # GEM outline
            ax.fill(self.geom.x_top, self.geom.y_top, color='lightblue', alpha=0.3, zorder=0)
            ax.plot(self.geom.x_top, self.geom.y_top, 'b-', lw=1.5)

            # Hits
            if x.size > 0:
                hb = ax.hexbin(x, y, gridsize=60, cmap='copper_r', mincnt=1)
                if i == n:
                    cbar = fig.colorbar(hb, ax=ax)
                    cbar.set_label("Hit density")
            else:
                ax.text(0, 0.5, "No hits", ha="center", va="center", fontsize=10, color="red")

            # Eta overlay
            if show_eta and hasattr(self.geom, "eta_y"):
                # horizontal break lines
                for yb in self.geom.eta_y:
                    # line from (xl(yb), yb) to (xr(yb), yb); we already know those at breakpoints
                    # find index of yb in stored array
                    j = np.where(np.isclose(self.geom.eta_y, yb))[0][0]
                    ax.plot([self.geom.eta_xl[j], self.geom.eta_xr[j]], [yb, yb],
                            color='deeppink', lw=1.2, alpha=0.9)

                # labels at region centers
                for k, poly in enumerate(self.geom.eta_polys, start=1):
                    yc = 0.5*(poly[0,1] + poly[2,1])
                    xc = 0.0  # centered label
                    ax.text(xc, yc, f"Eta {k}", ha='center', va='center',
                            fontsize=9, color='royalblue',
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="royalblue", alpha=0.7))

            ax.set_title(f"Layer {i} Hits")
            ax.set_xlabel("X (m)")
            ax.set_ylim(-0.2, self.geom.scintillator_length + 0.2)
            ax.grid()

        axs[0].set_ylabel("Y (m)")
        plt.tight_layout()
        self._save(filename)

    # ------------------------------------------
    # ---------- 3D Trajectory Plotting ----------
    # ------------------------------------------

    def plot_3d_trajectories(
        self, x0, y0, z0, v_x, v_y, v_z, idx,
        elev: float = 22, azim: float = -35,
        show_eta: bool = True, 
        y_breaks=None, y_mm=None, mm_bottom=None, mm_top=None,
        filename: Optional[str] = "3D_trajectories.png",
        colors: tuple = ((0.1, 0.6, 0.1), (0.8, 0.1, 0.1)),  # (pass, miss) RGB
        interactive: bool = False,          
        save_key: str = "s",
    ):
        """
        Plot 3D trajectories through the stack.
        """
        eps = 1e-12
        z_nodes = np.r_[self.geom.Z_TOP_SCIN, self.geom.layer_z, self.geom.Z_BOTTOM_SCIN]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # planes
        for zl in self.geom.layer_z:
            self.add_layer_patch(ax, self.geom.x_top, self.geom.y_top, zl,
                                fc='lightskyblue', ec='navy', alpha=0.15, lw=1.0)
        scx = [-self.geom.scintillator_width/2,  self.geom.scintillator_width/2,
                self.geom.scintillator_width/2, -self.geom.scintillator_width/2]
        scy = [0.0, 0.0, self.geom.scintillator_length, self.geom.scintillator_length]
        self.add_layer_patch(ax, scx, scy, self.geom.Z_TOP_SCIN,    fc='lightcoral', ec='darkred', alpha=0.20, lw=1.2)
        self.add_layer_patch(ax, scx, scy, self.geom.Z_BOTTOM_SCIN, fc='lightcoral', ec='darkred', alpha=0.20, lw=1.2)

        # line styling (alpha scales with how many we draw)
        n_lines = max(len(idx), 1)
        pass_rgb, miss_rgb = colors
        alpha_pass = float(np.clip(400.0 / n_lines, 0.08, 0.30))
        alpha_miss = float(np.clip(alpha_pass * 2.0, 0.20, 0.50))
        pass_color = (*pass_rgb, alpha_pass)
        miss_color = (*miss_rgb, alpha_miss)

        pass_all_count = 0

        def safe_div(dz, vz):
            vz_safe = np.where(np.abs(vz) < eps, np.copysign(eps, vz), vz)
            return dz / vz_safe

        # draw trajectories
        for k in idx:
            t_nodes = safe_div(z_nodes - z0[k], v_z[k])
            xs = x0[k] + v_x[k] * t_nodes
            ys = y0[k] + v_y[k] * t_nodes

            t_layers = safe_div(self.geom.layer_z - z0[k], v_z[k])
            xi = x0[k] + v_x[k] * t_layers
            yi = y0[k] + v_y[k] * t_layers

            inside = self.geom.gem_path.contains_points(np.column_stack([xi, yi]))
            ok = bool(np.all(inside))
            ax.plot(
                xs, ys, z_nodes,
                color=pass_color if ok else miss_color,
                lw=0.8 if ok else 1.2,
                solid_capstyle="round",  # smoother lines
                zorder=3
            )
            pass_all_count += int(ok)

        # optional eta overlays
        if show_eta:
            if hasattr(self.geom, "eta_polys"):
                eta_polys = self.geom.eta_polys
            else:
                _, eta_polys = self._eta_polys_from_args(y_breaks=y_breaks, y_mm=y_mm,
                                                        mm_bottom=mm_bottom, mm_top=mm_top)
            if eta_polys:
                
                # draw only on top GEM plane for clarity
                z_top_gem = float(np.max(self.geom.layer_z))

                self._add_eta_overlays_3d(ax, eta_polys, [z_top_gem],
                                        edgecolor='deeppink', lw=1.5, alpha=0.9)

        # cosmetics
        ax.set_title("Muon trajectories through GEM stack (3D)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(0.0, self.geom.scintillator_length)
        ax.set_zlim(self.geom.Z_BOTTOM_SCIN - 0.05, self.geom.Z_TOP_SCIN + 0.05)
        ax.set_box_aspect((0.7, self.geom.scintillator_length,
                        (self.geom.Z_TOP_SCIN - self.geom.Z_BOTTOM_SCIN)))
        ax.view_init(elev=elev, azim=azim)

       # legend (single call)
        legend_lines = [
            Line2D([0],[0], color=(*pass_rgb, 1.0), lw=2, label='hits all layers'),
            Line2D([0],[0], color=(*miss_rgb, 1.0), lw=3, label='misses ≥1 layer'),
        ]
        if show_eta:
            legend_lines.append(Line2D([0],[0], color='deeppink', lw=1.2,
                                       label='η boundaries (top layer)'))
        leg = ax.legend(handles=legend_lines, loc='upper left', frameon=True)
        leg.get_frame().set_alpha(0.9)

        # --- Save correctly ---
        if interactive:
            # allow rotation; press save_key to snapshot; close also saves once
            self._interactive_save(fig, filename, save_key=save_key, save_on_close=True)
            plt.show()
            plt.close(fig)
        else:
            fig.tight_layout()
            # save using the local figure, not plt.gcf()
            out = (self.output_dir / filename) if self.output_dir else filename
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)

        frac = pass_all_count / n_lines
        print(f"Fraction of plotted coincident tracks that hit all layers: {frac:.3f}")

    # ------------------------------------------       
    # 3D trajectories (one-liner from simulator) 
    # ------------------------------------------

    def plot_3d_trajectories_from_sim(self, sim, N=60_000, max_plot=600,
                                    elev=22, azim=-35, rng=None, seed=None,
                                    show_eta=False, y_breaks=None, y_mm=None,
                                    mm_bottom=None, mm_top=None):
        # use provided rng, else build from seed, else reuse simulator's
        if rng is None:
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = sim.rng  # ensures consistency with the simulator

        # start positions at top scintillator
        x0 = rng.uniform(self.geom.scin_xmin, self.geom.scin_xmax, size=N)
        y0 = rng.uniform(self.geom.scin_ymin, self.geom.scin_ymax, size=N)
        z0 = np.full(N, self.geom.Z_TOP_SCIN)

        # directions (from the same sim/rng)
        v_x, v_y, v_z = sim.sample_cos2_directions(N)

        # bottom plane intersection & coincidence
        t_bot = (self.geom.Z_BOTTOM_SCIN - z0) / v_z
        valid = t_bot > 0.0
        xb = x0 + v_x * t_bot
        yb = y0 + v_y * t_bot
        coinc = valid & self.geom.in_scintillator_xy(xb, yb)

        idx = np.flatnonzero(coinc)
        if idx.size > max_plot:
            idx = rng.choice(idx, size=max_plot, replace=False)

        return self.plot_3d_trajectories(x0, y0, z0, v_x, v_y, v_z, idx,
                                        elev=elev, azim=azim,
                                        show_eta=show_eta, y_breaks=y_breaks, y_mm=y_mm,
                                        mm_bottom=mm_bottom, mm_top=mm_top)


    # ------------------------------------------
    # ---------- Per-eta efficiency plots ------
    # ------------------------------------------
 
    def _eta_polys_from_args(self, y_breaks=None, y_mm=None, mm_bottom=None, mm_top=None):
        """
        Get eta polygons as list of (4,2) arrays in meters.
        Provide either y_breaks (meters) or y_mm (millimeters).
        """
        if y_breaks is not None:
            yb_m = np.asarray(y_breaks, dtype=float)
            if not np.all(np.diff(yb_m) >= 0):
                yb_m = np.sort(yb_m)
        elif y_mm is not None:
            yb_m = self.geom.eta_breaks_from_mm(y_mm, mm_bottom, mm_top)
        else:
            return None, None  # nothing to draw
        polys = self.geom.eta_polygons(yb_m)
        return yb_m, polys

    # ------------------------------------------
    # ---------- Draw eta overlays in 3D ---------
    # ------------------------------------------

    def _add_eta_overlays_3d(self, ax, eta_polys, z_planes,
                            edgecolor='deepskyblue', lw=0.9, alpha=0.9):
        """
        Draw given eta polygons (top‑view x,y) on each z plane in z_planes.
        Outlines only (no fill) for clarity.
        """
        for z in np.atleast_1d(z_planes):
            for poly in eta_polys:
                # close poly
                P = np.vstack([poly, poly[0]])
                ax.plot(P[:,0], P[:,1], z, color=edgecolor, lw=lw, alpha=alpha)

    # ------------------------------------------
    # ---------- Per-eta efficiency bar charts ------
    # ------------------------------------------

    def plot_eta_efficiency_by_layer(
        self,
        eta_result: dict,
        ylim: Optional[tuple] = None,
        annotate: bool = True,
        filename: Optional[str] = "eta_eff_by_layer.png",
        bar_face: str = "#57c28a",
        bar_edge: str = "#2d7a56",
        show_percent: bool = True,          # NEW
        show_errorbars: bool = False,       # NEW (binomial on conditional eff per-η)
        dpi: int = 150,
    ):
        """
        Per-η efficiencies (relative to coincidence baseline) for each layer.
        If show_errorbars=True and eta_result has 'n_coinc' and 'totals',
        error bars show conditional efficiency σ ≈ sqrt(p*(1-p)/N_eta).
        """
        eff    = np.asarray(eta_result["eff"])               # (L, 8); sums to layer ε
        totals = np.asarray(eta_result.get("totals", np.zeros_like(eff)))  # coincidences per η
        n_coinc = int(eta_result.get("n_coinc", 0))          # total coincidences (baseline)
        L, K = eff.shape

        # choose format (fraction vs percent)
        scale = 100.0 if show_percent else 1.0
        ylab  = "Efficiency (%)" if show_percent else "Efficiency"

        # sensible global y-limit
        max_val = float(np.nanmax(eff)) if eff.size else 1.0
        if ylim is None:
            headroom = 1.12
            ylim = (0.0, min(1.0, max_val * headroom))

        fig, axs = plt.subplots(1, L, figsize=(3.4 * L, 3.6), sharey=True, dpi=dpi)
        if L == 1:
            axs = [axs]

        x = np.arange(1, K + 1)
        xlabels = [f"η{k}" for k in range(1, K + 1)]

        # Precompute error bars if requested
        yerr = None
        if show_errorbars and n_coinc > 0 and totals.size:
            hits = eff * n_coinc                 # hits per (layer, η)
            with np.errstate(divide='ignore', invalid='ignore'):
                p_cond = np.where(totals > 0, hits / totals, np.nan)   # conditional efficiency
                se = np.where(totals > 0, np.sqrt(p_cond*(1-p_cond)/totals), np.nan)
            yerr = se * scale                    # scale to percent if needed

        for li, ax in enumerate(axs, start=1):
            y = eff[li - 1] * scale
            bars = ax.bar(x, y, color=bar_face, edgecolor=bar_edge, alpha=0.9)

            # optional error bars (on top of bars)
            if yerr is not None:
                ax.errorbar(x, y, yerr=yerr[li - 1], fmt='none', ecolor=bar_edge, elinewidth=1, capsize=2, alpha=0.9)

            layer_eps = float(eff[li - 1].sum())
            ax.set_title(f"Layer {li}  (ε = {layer_eps:.3f})", fontsize=12)
            ax.set_xticks(x); ax.set_xticklabels(xlabels)
            ax.set_ylim(ylim[0]*scale, ylim[1]*scale)
            ax.set_xlabel("Eta region")
            ax.grid(axis="y", alpha=0.3, linestyle=":")

            if show_percent:
                ax.yaxis.set_major_formatter(PercentFormatter(100))

            if annotate:
                for k, b in enumerate(bars):
                    n = int(totals[li - 1, k]) if totals.size else 0
                    ax.annotate(f"n={n}",
                                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, clip_on=True)

        axs[0].set_ylabel(ylab)
        fig.suptitle("Per-eta efficiency by layer", fontsize=14, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        # save
        if filename is None:
            filename = "eta_eff_by_layer.png"
        self._save(filename)
        #plt.show()

    # ------------------------------------------
    # ---------- Per-layer x-occupancy by eta ------
    # ------------------------------------------

    def plot_x_occupancy_per_layer(
        self, result, bins=40, layout=(2, 4),
        density=False, facecolor="#4da3ff", ec="white",
        basename: str = "x_occupancy_eta_layer.png",
        share_y: bool = False, y_pad: float = 0.08,
        show: bool = False,
        panel_style: str = "eta-left-count-corner",   # <- NEW
    ):
        if not hasattr(self.geom, "eta_polys"):
            raise RuntimeError("Eta layout not set. Call geom.enable_default_me0_eta() first.")

        n_layers = self.geom.n_layers
        rows, cols = layout
        assert rows * cols == 8, "layout must have 8 panels total (e.g., (2,4) or (4,2))."

        for li in range(n_layers):
            xy = result["hit_xy"][li]
            x, y = (np.array([]), np.array([])) if (xy is None) else xy
            eta_idx = self.geom.which_eta(x, y) if x.size else np.array([], dtype=int)

            fig, axes = plt.subplots(rows, cols, figsize=(cols*3.1, rows*2.6), sharey=share_y)
            axes = np.atleast_1d(axes).ravel()

            for k in range(1, 9):
                ax = axes[k-1]
                sel = (eta_idx == k)
                xk = x[sel]

                # nice x-lims from eta polygon
                poly = self.geom.eta_polys[k-1]
                xmin, xmax = poly[:,0].min(), poly[:,0].max()

                if xk.size:
                    n, _, _ = ax.hist(xk, bins=bins, range=(xmin, xmax),
                                    density=density, color=facecolor, edgecolor=ec)
                    ymax = float(np.max(n)) if n.size else 1.0
                    ax.set_ylim(0, ymax * (1 + y_pad))
                else:
                    ax.text(0.5, 0.5, "no hits", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="crimson")

                ax.set_xlim(xmin, xmax)
                ax.set_xlabel("x (m)")
                if k in (1, 5):
                    ax.set_ylabel("Counts" if not density else "Density")
                ax.grid(alpha=0.25, linestyle=":")

                # ---- cleaner titles ----
                if panel_style == "eta-left-count-corner":
                    ax.set_title(fr"$\eta_{k}$", loc="left", fontsize=12, pad=2)
                    ax.text(0.98, 0.92, f"n={xk.size}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color="0.35")
                elif panel_style == "eta-only":
                    ax.set_title(fr"$\eta_{k}$", fontsize=12)
                elif panel_style == "count-only":
                    ax.set_title(f"n={xk.size}", fontsize=11, color="0.35")
                else:
                    # "none" or any other string → no panel title
                    pass

            # shorter, cleaner suptitle
            fig.suptitle(f"Layer {li+1} · x-occupancy by η", fontsize=14, y=0.99)
            plt.tight_layout()
            self._save(f"{basename}{li+1}.png")
            (plt.show() if show else plt.close(fig))
