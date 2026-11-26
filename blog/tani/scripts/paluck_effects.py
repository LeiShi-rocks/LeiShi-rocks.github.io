"""Densities of the PNA adapter's predicted own and peer effects (school experiment).

Needs the restricted ICPSR data processed locally; only aggregate densities are plotted.
Run from the repository root:

    python blog/scripts/paluck_effects.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from code.paluck_data import build_base, exposure_design  # noqa: E402
from code.run_paluck import DERIVED  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "paluck_effects.png"
ADAPTER = "#d9480f"


def main():
    base = build_base()
    observed = ~np.isnan(base["outcomes"]["wrist"])
    eligible = base["eligible"]
    prob = exposure_design()["prob"]
    preds = np.load(DERIVED / "wrist_split_oof.npz")
    panels = [
        ("Own effect of an invitation (eligible students)",
         preds["adapter_pna__ind"][observed & eligible]),
        ("Peer effect of friends' invitations (never-eligible students)",
         preds["adapter_pna__peer"][observed & ~eligible & (prob > 0) & (prob < 1)]),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, (title, v) in zip(axes, panels):
        lo, hi = np.quantile(v, [.001, .999])
        pad = (hi - lo) * .1
        grid = np.linspace(lo - pad, hi + pad, 400)
        dens = gaussian_kde(v)(grid)
        ax.fill_between(grid, dens, color=ADAPTER, alpha=.2, lw=0)
        ax.plot(grid, dens, color=ADAPTER, lw=1.6)
        ax.axvline(v.mean(), color="0.25", ls="--", lw=.9)
        ax.text(.98, .92, f"mean {v.mean():.3f}\nSD {v.std():.3f}\nn = {len(v):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("predicted change in P(wore wristband)")
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
