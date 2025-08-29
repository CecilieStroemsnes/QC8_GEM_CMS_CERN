# run.py — generate & save all plots + write a text summary (Py3.9 compatible)
import matplotlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import argparse

from Classes import (
    ME0_Geometry,
    GEMTrajectorySimulator,
    Plots,
    count_hits_by_eta,
)

def write_text_report(
    outdir: Path,
    geom,
    res: dict,
    eta_counts: Optional[np.ndarray],
    eta_res: Optional[dict],
    fname: str = "summary.txt",
    stats_5: Optional[dict] = None,
):
    """Summary of simulation """
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / fname

    n_gen   = int(res.get("n_generated", 0))
    n_coinc = int(res.get("n_coinc", 0))
    layer_hits = np.asarray(res.get("layer_hits", []), dtype=int)
    acc = np.asarray(res.get("acceptance", []), dtype=float)

    misses = (n_coinc - layer_hits) if layer_hits.size else np.array([], dtype=int)
    total_hits_layers   = int(layer_hits.sum()) if layer_hits.size else 0
    total_misses_layers = int(misses.sum()) if layer_hits.size else 0

    with fp.open("w", encoding="utf-8") as f:
        f.write("=== GEM Simulation Summary ===\n")
        f.write(f"Generated muons            : {n_gen}\n")
        f.write(f"Scintillator coincidences  : {n_coinc} ({(100*n_coinc/max(n_gen,1)):.2f}% of generated)\n\n")

        if layer_hits.size:
            f.write("Per-layer results (relative to coincidence baseline):\n")
            f.write("Layer  Hits     Misses   Acceptance\n")
            f.write("-----  -------- -------- ----------\n")
            for i, (h, m, e) in enumerate(zip(layer_hits, misses, acc), start=1):
                f.write(f"{i:>5}  {h:>8} {m:>8}  {e:>10.4f}\n")
            f.write("-----  -------- -------- ----------\n")
            f.write(f"TOTAL  {total_hits_layers:>8} {total_misses_layers:>8}\n\n")

        if eta_counts is not None and eta_counts.size:
            L, K = eta_counts.shape
            f.write("Per-η hit counts per layer (per-layer counts, not unique muons):\n")
            header = "Layer " + " ".join([f"η{k:>2}" for k in range(1, K+1)]) + "   |  Sum"
            f.write(header + "\n" + "-" * len(header) + "\n")
            for ell in range(L):
                row_sum = int(eta_counts[ell].sum())
                cols = " ".join([f"{int(c):>3}" for c in eta_counts[ell]])
                f.write(f"{ell+1:>5} {cols}   |  {row_sum:>4}\n")
            f.write("-" * len(header) + "\n")
            per_eta = eta_counts.sum(axis=0)
            f.write(" Sum  " + " ".join([f"{int(x):>3}" for x in per_eta]) + f"   |  {int(per_eta.sum()):>4}\n\n")

        if eta_res is not None and "acc" in eta_res:
            acc_eta = np.asarray(eta_res["acc"])      # (L, K)
            totals  = np.asarray(eta_res.get("totals", np.zeros_like(acc_eta)))
            L, K = acc_eta.shape
            f.write("Per-η acceptances by layer (relative to coincidence baseline):\n")
            header = "Layer " + " ".join([f"η{k:>2}" for k in range(1, K+1)]) + "   |  Layer ε"
            f.write(header + "\n" + "-" * len(header) + "\n")
            for ell in range(L):
                cols = " ".join([f"{acc_eta[ell,k]:>5.3f}" for k in range(K)])
                f.write(f"{ell+1:>5} {cols}   |  {acc_eta[ell].sum():>7.3f}\n")
            f.write("-" * len(header) + "\n")
            f.write("(totals row below are event counts per η across layers)\n")
            f.write("Totals per η (counts): " + " ".join([str(int(x)) for x in totals.sum(axis=0)]) + "\n\n")


        if stats_5 is not None:
            nC   = int(stats_5.get("n_coinc", 0))
            n5   = int(stats_5.get("n_exactly_Lminus1", 0))
            frac = float(stats_5.get("frac_exactly_Lminus1", 0.0))
            missed_hist = np.asarray(stats_5.get("missed_hist", []), dtype=int)

            f.write("\n--- Exactly 5-of-6 layer hits ---\n")
            f.write(f"Coincidences considered      : {nC}\n")
            f.write(f"Tracks with exactly 5/6 hits : {n5} ({100*frac:.2f}%)\n")
            if missed_hist.size:
                by_layer = " ".join(f"L{i+1}:{missed_hist[i]}" for i in range(missed_hist.size))
                f.write(f"Missed-by-layer counts       : {by_layer}\n")

        f.write("\nNotes:\n")
        f.write(' - "Misses" are per-layer: misses = coincidences - hits_on_layer.\n')
        f.write(" - Layer totals count per-layer events (one muon that hits all layers contributes 6 hits).\n")


    print(f"Summary written to: {fp.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", choices=["high", "low"], default="high",
                        help="Choose geometry position (default: high)")
    args = parser.parse_args()
    
    out = Path("Outputs")
    out.mkdir(exist_ok=True)

    # ------------------------------------------
    # -- Geometry ------------------------------
    # ------------------------------------------
    geom = ME0_Geometry(position=args.position)
    if hasattr(geom, "enable_default_me0_eta"):
        geom.enable_default_me0_eta()

    # -------------------------------------------
    # -- Simulation & Plots ---------------------
    # -------------------------------------------

    sim = GEMTrajectorySimulator(geom, seed=42)
    plots = Plots(geom, output_dir=out)  # your Plots class saves internally

    # ------------------------------------------
    # -- 3D Geometry Views ---------------------
    # ------------------------------------------
    if hasattr(plots, "four_views"):
        
        # ------------------------------------------        
        # Static 4 views
        # ------------------------------------------
        plots.four_views(show_eta=True, filename="stack_geometry.png") # -> Stack Geometry.png
        
        # ------------------------------------------
        # Interactive 4 views
        # ------------------------------------------
        #plots.four_views(show_eta=True, interactive=True, filename="Stack Geometry.png") # -> Stack Geometry.png

    # ------------------------------------------
    # Single interactive 3D view
    # ------------------------------------------
    #if hasattr(plots, "plot_3d"):
    #    plots.plot_3d(
    #        elev=20, azim=-60, show_eta=True,
    #        filename="stack_geometry_manual.png",
    #        interactive=True,       # enables rotation + snapshot
    #        save_key="s"            # press s to save; closing also saves once 
    #    ) # -> stack_geometry_manual.png


    # ------------------------------------------
    # -- Acceptance Simulation -----------------
    # ------------------------------------------
    res = sim.simulate_acceptance(N=100_000, return_hit_xy=True)
    print(f"Generated {res.get('n_generated')} muons, {res.get('n_coinc')} with scintillator coincidence.")
    print("Layer hits:", res.get('layer_hits'))
    print("Layer acceptances:", res.get('acceptance'))

    # ------------------------------------------
    # -- Per η hit counts ----------------------
    # ------------------------------------------
    eta_counts = None
    if 'hit_xy' in res and res['hit_xy'] is not None:
        try:
            counts = count_hits_by_eta(geom, res['hit_xy'])
            eta_counts = np.asarray(counts.get('counts'))
        except Exception as e:
            print("count_hits_by_eta failed:", e)


    # ------------------------------------------
    # -- Acceptance plot (per layer) -----------
    # ------------------------------------------
    if hasattr(plots, "plot_acceptance_histogram"):
        plots.plot_acceptance_histogram(res) #-> acceptance_histogram.png
    
    # ------------------------------------------
    # -- Hit maps (per layer) ------------------
    # ------------------------------------------
    if hasattr(plots, "plot_hit_maps"):
        plots.plot_hit_maps(res, show_eta=True) # -> hit_map_1..6.png

    # ------------------------------------------
    # -- 3D trajectory plot --------------------
    # ------------------------------------------        
    if hasattr(plots, "plot_3d_trajectories_from_sim"):
        plots.plot_3d_trajectories_from_sim(
            sim, N=60_000, max_plot=600, elev=22, azim=-35, show_eta=True
        )  # -> 3D_trajectories.png

    # ------------------------------------------
    # -- #3D interactive trajectory plot -------
    # ------------------------------------------
    #if hasattr(plots, "plot_3d_trajectories_from_sim"):
    #    plots.plot_3d_trajectories_from_sim(
    #        sim, N=60_000, max_plot=600, elev=22, azim=-35, show_eta=True,
    #        interactive=True,       # enables rotation + snapshot
    #        save_key="s"            # press s to save; closing also saves once
    #    )  # -> 3D_trajectories.png
    
    # ------------------------------------------
    # -- X occupancy per layer -----------------
    # ------------------------------------------
    if hasattr(plots, "plot_x_occupancy_per_layer"):
        plots.plot_x_occupancy_per_layer(res, basename="x_occupancy_eta_layer")  # -> _1.._6.png


    # ------------------------------------------
    # -- Acceptance vs eta ---------------------
    # ------------------------------------------
    eta_res: Optional[dict] = None
    if hasattr(sim, "simulate_eta_acceptance"):
        try:
            eta_res = sim.simulate_eta_acceptance(N=100_000)
            if hasattr(plots, "plot_eta_acceptance_by_layer"):
                # This saves to plots/eta_acc_by_layer.png by default
                plots.plot_eta_acceptance_by_layer(eta_res, annotate=True)
                # or give a custom name:
                # plots.plot_eta_acceptance_by_layer(eta_res, annotate=True, filename="eta_acc_by_layer.png")
        except Exception as e:
            print("simulate_eta_acceptance failed:", e)

    # --- 3D plot of tracks that hit exactly 5 of 6 layers (one miss) ---
    stats_5 = None
    if hasattr(plots, "plot_3d_trajectories_5of6_from_sim"):
        stats_5 = plots.plot_3d_trajectories_5of6_from_sim(
            sim,
            N=100_000,
            max_plot=600,
            elev=22,
            azim=-35,
            show_eta=True,
            filename="3D_trajectories_5of6.png",
            interactive=True,
            save_key="s" 
        )
    elif hasattr(sim, "compute_5of6_stats"):
        stats_5 = sim.compute_5of6_stats(N=100_000)

    # ------------------------------------------
    # -- Write text summary --------------------
    # ------------------------------------------
    write_text_report(out, geom, res, eta_counts, eta_res, fname="summary.txt", stats_5=stats_5)

    # -- Done ----------------------------------
    plt.close("all")
    print(f"\nAll outputs saved to: {out.resolve()}")

if __name__ == "__main__":
    main()