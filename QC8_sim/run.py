# run.py — generate & save all plots + write a text summary (Py3.9 compatible)
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

def _try(step_name, fn, *args, **kwargs):
    """Run a step; on error, print a short message and continue."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[skip] {step_name}: {e}")
        return None

def write_text_report(
    outdir: Path,
    res: dict,
    eta_counts: Optional[np.ndarray] = None,
    eta_res: Optional[dict] = None, 
    fname: str = "summary.txt",
    stats_5: Optional[dict] = None,
    stats_5_eta1: Optional[dict] = None,
    stats_5_eta8: Optional[dict] = None,
    miss_axis: Optional[dict] = None,
    r_top_e1: Optional[dict] = None,
    r_top_e8: Optional[dict] = None,
    r_bot_e1: Optional[dict] = None,
    r_bot_e8: Optional[dict] = None, 
    r_top_e8_X: Optional[dict] = None,
    r_top_e8_Y: Optional[dict] = None,
    r_bot_e8_X: Optional[dict] = None,
    r_bot_e8_Y: Optional[dict] = None,
    r_top_e1_X: Optional[dict] = None,
    r_top_e1_Y: Optional[dict] = None,
    r_bot_e1_X: Optional[dict] = None,
    r_bot_e1_Y: Optional[dict] = None,
    
):
    """Summary of simulation """
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / fname

    n_gen   = int(res.get("n_generated", 0))
    n_coinc = int(res.get("n_coinc", 0))
    layer_hits = np.asarray(res.get("layer_hits", []), dtype=int)
    acc = np.asarray(res.get("acceptance", []), dtype=float)

    misses = (n_coinc - layer_hits) if layer_hits.size else np.array([], dtype=int)

    with fp.open("w", encoding="utf-8") as f:
        # ---------- small helpers ----------
        def H(title: str):
            line = "=" * (len(title) + 8)
            f.write(f"\n{line}\n=== {title} ===\n{line}\n")

        def sub(title: str):
            f.write(f"\n-- {title} --\n")

        def fmt_row(cols, widths, sep="  "):
            f.write(sep.join(f"{c:>{w}}" for c, w in zip(cols, widths)) + "\n")

        def print_layer_table(layer_hits, misses, acc):
            H("Per-layer hits and acceptance")
            widths = (5, 9, 9, 10)
            fmt_row(("Layer","Hits","Misses","Acceptance"), widths)
            fmt_row(("-"*5,"-"*9,"-"*9,"-"*10), widths)
            for i, (h, m, e) in enumerate(zip(layer_hits, misses, acc), start=1):
                fmt_row((i, h, m, f"{e:.4f}"), widths)
            fmt_row(("-"*5,"-"*9,"-"*9,"-"*10), widths)
            fmt_row(("TOTAL", int(layer_hits.sum()), int(misses.sum()), ""), widths)

        def print_eta_counts(eta_counts):
            L, K = eta_counts.shape
            H("Per-η hit counts per layer")
            header = ["Layer"] + [f"η{k}" for k in range(1, K+1)] + ["|", "Sum"]
            widths = [5] + [4]*K + [2, 5]
            fmt_row(header, widths)
            fmt_row(["-"*5] + ["-"*4]*K + ["--", "-"*5], widths)
            for ell in range(L):
                row = [ell+1] + [int(x) for x in eta_counts[ell]] + ["|", int(eta_counts[ell].sum())]
                fmt_row(row, widths)
            fmt_row(["-"*5] + ["-"*4]*K + ["--", "-"*5], widths)
            per_eta = eta_counts.sum(axis=0)
            fmt_row(["Sum"] + [int(x) for x in per_eta] + ["|", int(per_eta.sum())], widths)

        def print_eta_acceptance(eta_acc, layer_eps):
            L, K = eta_acc.shape
            H("Per-η acceptance per layer")
            header = ["Layer"] + [f"η{k}" for k in range(1, K+1)] + ["|", "Layer ε"]
            widths = [5] + [7]*K + [2, 8]
            fmt_row(header, widths)
            fmt_row(["-"*5] + ["-"*7]*K + ["--", "-"*8], widths)
            for ell in range(L):
                row = [ell+1] + [f"{eta_acc[ell,k]:.3f}" for k in range(K)] + ["|", f"{layer_eps[ell]:.3f}"]
                fmt_row(row, widths)

        def print_miss_axis(miss_axis):
            H("5-of-6 miss directions")
            f.write("Shows directions of misses for tracks with exactly 5 hits.\n")
            if not (isinstance(miss_axis, dict) and "counts" in miss_axis):
                f.write("No miss-axis data.\n")
                return
            #for cat in ("Y-below", "Y-above", "X-left", "X-right"):
                #f.write(f"{cat:>16}: {miss_axis['counts'].get(cat, 0)}\n")

            # Optional per-η breakdown
            if "per_eta" in miss_axis:
                per_eta = miss_axis["per_eta"]
                # infer K
                K = next((len(v) for v in per_eta.values() if hasattr(v, "__len__")), 0)
                if K:
                    f.write("\nMisses by η (counts):\n")
                    widths = [12] + [4]*K + [2, 5]
                    fmt_row([""] + [f"η{i}" for i in range(1, K+1)] + ["|","Sum"], widths)
                    for cat in ("Y-below", "Y-above", "X-left", "X-right"):
                        row = per_eta.get(cat)
                        if row is not None and len(row) == K:
                            fmt_row([cat] + [int(v) for v in row] + ["|", int(sum(row))], widths)

        def print_5v6_block(eta_label, stats_5_eta, r_top, r_bot, r_top_X, r_top_Y, r_bot_X, r_bot_Y):
            H(f"5-of-6 vs 6-of-6 in {eta_label}")
            if stats_5_eta:
                sub(f"Exactly 5-of-6 in {eta_label}")
                f.write(f"Coincidences: {stats_5_eta['n_coinc']}\n")
                f.write(f"5/6 total   : {stats_5_eta['n_exactly_Lminus1']} "
                        f"({100*stats_5_eta['frac_exactly_Lminus1']:.3f}%)\n")
                f.write("Missed per layer (1..6): " + " ".join(f"{x:>4}" for x in stats_5_eta["missed_hist"]) + "\n")

            def _print_axis(tag, r):
                if not r: return
                f.write(f"   {tag}:\n")
                f.write(f"     5-of-6 : {r['n5']}")
                if r.get("n5_breakdown"):
                    f.write(f"          breakdown={r['n5_breakdown']}")
                f.write("\n")
                #f.write(f"     6-of-6 : {r['n6']}\n")
                f.write(f"     Eff    : {100*r['ratio_5_over_5p6']:.3f}%\n")

            if r_top:
                sub(f"TOP layer ({eta_label})")
                f.write(f"5-of-6 : {r_top['n5']}\n")
                f.write(f"6-of-6 : {r_top['n6']}\n")
                #f.write(f"Eff    : {100*r_top['ratio_5_over_5p6']:.3f}%\n")
                f.write("\n")
            _print_axis("X-only miss", r_top_X)
            f.write("\n")
            _print_axis("Y-only miss", r_top_Y)

            if r_bot:
                sub(f"BOTTOM layer ({eta_label})")
                f.write(f"5-of-6 : {r_bot['n5']}\n")
                f.write(f"6-of-6 : {r_bot['n6']}\n")
                #f.write(f"Eff    : {100*r_bot['ratio_5_over_5p6']:.3f}%\n")
                f.write("\n")
            _print_axis("X-only miss", r_bot_X)
            f.write("\n")
            _print_axis("Y-only miss", r_bot_Y)

        # ---------- header ----------
        H("QC8 Simulation Summary")
        f.write(f"Generated muons           : {n_gen}\n")
        f.write(f"Scintillator coincidences : {n_coinc}  ({(100*n_coinc/max(n_gen,1)):.2f}% of generated)\n")

        # ---------- per-layer table ----------
        if layer_hits.size:
            print_layer_table(layer_hits, misses, acc)

        # ---------- per-η counts ----------
        if eta_counts is not None and eta_counts.size:
            print_eta_counts(eta_counts)

        # ---------- per-η acceptance ----------
        if eta_res is not None and "acc" in eta_res:
            acc_eta = np.asarray(eta_res["acc"])            # (L, K)
            layer_eps = acc_eta.sum(axis=1)
            print_eta_acceptance(acc_eta, layer_eps)

        # ---------- global 5-of-6 ----------
        H("Exactly 5 vs 6 hits (global)")
        f.write("Tracks with at least one coincidence are considered.\n")
        f.write("Only tracks with exactly 5 or exactly 6 hits are counted in the efficiency.\n")
        f.write("\n")
        if stats_5:
            f.write(f"Coincidences considered : {stats_5['n_coinc']}\n")
            f.write(f"Tracks with exactly 5/6  : {stats_5['n_exactly_Lminus1']} "
                    f"({100*stats_5['frac_exactly_Lminus1']:.2f}%)\n")
            f.write("Missed per layer (1..6): " + " ".join(f"{x:>4}" for x in stats_5["missed_hist"]) + "\n")

        # ---------- miss-axis summary ----------
        print_miss_axis(miss_axis)

        # ---------- diagram ----------
        diagram = """
                  Y-above     
             -----------------
            |                 |
            |                 |
            \                 / 
     X-left  \               /  X-right
              \             /  
               \           /   
                \_________/    
                  Y-below 
            """
        f.write("\n" + diagram + "\n")

        # ---------- η8 (with TOP/BOTTOM and X/Y splits) ----------
        print_5v6_block(
            "η8",
            stats_5_eta8,
            r_top_e8, r_bot_e8,
            r_top_e8_X, r_top_e8_Y,
            r_bot_e8_X, r_bot_e8_Y
        )

        # ---------- η1 (with TOP/BOTTOM and X/Y splits) ----------
        print_5v6_block(
            "η1",
            stats_5_eta1,
            r_top_e1, r_bot_e1,
            r_top_e1_X, r_top_e1_Y,
            r_bot_e1_X, r_bot_e1_Y
        )

    print(f"Summary written to: {fp.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", choices=["high", "low"], default="high",
                        help="Choose geometry position (default: high)")
    parser.add_argument("--N", type=int, default=500_000,
                        help="Number of muons to simulate (default: 500k)")
    parser.add_argument("--interactive", action="store_true",
                        help="Make 3D plots interactive (press 's' to save)")
    args = parser.parse_args()
    
    out = Path("Outputs")
    out.mkdir(exist_ok=True)

    # -- Geometry ------------------------------
    geom = ME0_Geometry(position=args.position)
    geom.enable_default_me0_eta()

    # -- Simulation & Plots ---------------------
    sim = GEMTrajectorySimulator(geom, seed=42)
    plots = Plots(geom, output_dir=out, simulator=sim)  
    N_muons = args.N
    ELEV, AZIM = 22, -35 
    MAX_PLOT = 600  
    SAVE_KEY = "s"  

    ELEV5_6 = 24
    AZIM5_6 = -65  

    # -- Acceptance simulation ------------------
    res = sim.simulate_acceptance(N=N_muons, return_hit_xy=True)
    print(f"Generated {res.get('n_generated')} muons, {res.get('n_coinc')} with scintillator coincidence.")
    print("Layer hits:", res.get('layer_hits'))
    print("Layer acceptances:", res.get('acceptance'))

    # -- Per-η hit counts -----------------------
    eta_counts = None
    if res.get('hit_xy') is not None:
        counts_dict = _try("count_hits_by_eta", count_hits_by_eta, geom, res['hit_xy'])
        if isinstance(counts_dict, dict):
            eta_counts = np.asarray(counts_dict.get("counts", []))

    # -- Geometry and plots ---------------------
    _try("four_views", plots.four_views, show_eta=True, 
         interactive=args.interactive, filename="stack_geometry.png")
    _try("plot_acceptance_histogram", plots.plot_acceptance_histogram, res)
    _try("plot_hit_maps", plots.plot_hit_maps, res, show_eta=True)
    _try("plot_3d_trajectories_from_sim", plots.plot_3d_trajectories_from_sim,
            sim, N=min(60_000, N_muons), max_plot=MAX_PLOT, elev=ELEV, azim=AZIM, show_eta=True,         
            interactive=args.interactive, save_key=SAVE_KEY)

    # -- X occupancy per layer -----------------
    _try("plot_x_occupancy_per_layer", plots.plot_x_occupancy_per_layer, res, basename="x_occupancy_eta_layer")

    # -- Acceptance vs eta ---------------------
    eta_res = _try("simulate_eta_acceptance", sim.simulate_eta_acceptance, N=N_muons)
    if eta_res is not None:
        _try("plot_eta_acceptance_by_layer", plots.plot_eta_acceptance_by_layer, eta_res, annotate=True)

    # -- Build per-track hit mask --------------
    data = _try("track_layer_hits", sim.track_layer_hits, N=N_muons)
    miss_axis = _try("summarize_5_of_6_miss_axis", sim.summarize_5_of_6_miss_axis, data)

    # --- 5-of-6 plots & summaries (only if data is valid) ---
    stats_5 = stats_5_eta1 = stats_5_eta8 = None
    r_top_e1 = r_top_e8 = r_bot_e1 = r_bot_e8 = None
    r_top_e8_X = r_top_e8_Y = r_bot_e8_X = r_bot_e8_Y = None
    r_top_e1_X = r_top_e1_Y = r_bot_e1_X = r_bot_e1_Y = None

    if data is None:
        stats_5 = stats_5_eta1 = stats_5_eta8 = None
    else:
        # global 5 of 6
        stats_5 = _try("plot_3d_trajectories_5of6_from_data", plots.plot_3d_trajectories_5of6_from_data,
                        data, max_plot=MAX_PLOT, elev=ELEV5_6, azim=AZIM5_6, show_eta=True,
                        filename="3D_trajectories_5of6.png",
                        interactive=args.interactive, save_key=SAVE_KEY)
        
        # eta 1 restricted 5 of 6
        stats_5_eta1 = _try("plot_3d_trajectories_5of6_eta_from_data (η1)",
                            plots.plot_3d_trajectories_5of6_eta_from_data,
                            data, target_eta=1, max_plot=MAX_PLOT, elev=ELEV5_6, azim=AZIM5_6,
                            show_eta=True, filename="3D_trajectories_5of6_eta1.png",
                            interactive=args.interactive, save_key=SAVE_KEY)
        
        # eta 8 restricted 5 of 6
        stats_5_eta8 = _try("plot_3d_trajectories_5of6_eta_from_data (η8)",
                            plots.plot_3d_trajectories_5of6_eta_from_data,
                            data, target_eta=8, max_plot=MAX_PLOT, elev=ELEV5_6, azim=AZIM5_6,
                            show_eta=True, filename="3D_trajectories_5of6_eta8.png",
                            interactive=args.interactive, save_key=SAVE_KEY)
        
        r_top_e1 = _try("summarize_5v6_in_eta_top_strict (η1)", sim.summarize_5v6_in_eta_top_strict, data, 1)
        r_top_e8 = _try("summarize_5v6_in_eta_top_strict (η8)", sim.summarize_5v6_in_eta_top_strict, data, 8)
        r_bot_e1 = _try("summarize_5v6_in_eta_bottom_strict (η1)", sim.summarize_5v6_in_eta_bottom_strict, data, 1)
        r_bot_e8 = _try("summarize_5v6_in_eta_bottom_strict (η8)", sim.summarize_5v6_in_eta_bottom_strict, data, 8)

        # Axis-split variants (guarded + via _try)
        r_top_e8_X = _try("top_strict_axis (η8, X)", sim.summarize_5v6_in_eta_top_strict_axis, data, 8, "X")
        r_top_e8_Y = _try("top_strict_axis (η8, Y)", sim.summarize_5v6_in_eta_top_strict_axis, data, 8, "Y")
        r_bot_e8_X = _try("bot_strict_axis (η8, X)", sim.summarize_5v6_in_eta_bottom_strict_axis, data, 8, "X")
        r_bot_e8_Y = _try("bot_strict_axis (η8, Y)", sim.summarize_5v6_in_eta_bottom_strict_axis, data, 8, "Y")

        r_top_e1_X = _try("top_strict_axis (η1, X)", sim.summarize_5v6_in_eta_top_strict_axis, data, 1, "X")
        r_top_e1_Y = _try("top_strict_axis (η1, Y)", sim.summarize_5v6_in_eta_top_strict_axis, data, 1, "Y")
        r_bot_e1_X = _try("bot_strict_axis (η1, X)", sim.summarize_5v6_in_eta_bottom_strict_axis, data, 1, "X")
        r_bot_e1_Y = _try("bot_strict_axis (η1, Y)", sim.summarize_5v6_in_eta_bottom_strict_axis, data, 1, "Y")

    # -- Write text summary --------------------
    _try("write_text_report", write_text_report,
        out, res, eta_counts, eta_res,
        stats_5=stats_5, stats_5_eta1=stats_5_eta1, stats_5_eta8=stats_5_eta8,
        miss_axis=miss_axis,
        r_top_e1=r_top_e1, r_top_e8=r_top_e8, r_bot_e1=r_bot_e1, r_bot_e8=r_bot_e8,
        r_top_e8_X=r_top_e8_X, r_top_e8_Y=r_top_e8_Y,
        r_bot_e8_X=r_bot_e8_X, r_bot_e8_Y=r_bot_e8_Y,
        r_top_e1_X=r_top_e1_X, r_top_e1_Y=r_top_e1_Y,
        r_bot_e1_X=r_bot_e1_X, r_bot_e1_Y=r_bot_e1_Y
    )

    plt.close("all")
    print(f"\nAll outputs saved to: {out.resolve()}")

if __name__ == "__main__":
    main()