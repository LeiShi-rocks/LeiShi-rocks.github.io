"""Generate the blog figures.

Hard-coded numbers are copied from the result tables of the TANI paper (means and
sample SDs over realizations).  The label-budget and node-level figures read the
corresponding result files of the project.  Run from the repository root:

    python blog/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blog" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared visual language: proposed adapter = orange, fixed features = blue,
# raw TabPFN = grey, network-causal baselines = green, oracle = black.
ADAPTER = "#d9480f"
FIXED = "#1c7ed6"
FIXED2 = "#74c0fc"
RAW = "#868e96"
BASE = "#2b8a3e"
OTHER = "#ae3ec9"
ORACLE = "#212529"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def hero():
    """Exp. 3 vs Exp. 4 individual PEHE: raw, best fixed block, PNA adapter."""
    panels = [
        ("Effects depend on neighbor dispersion\nand two-hop structure (Exp. 3)",
         "fixed: diffusion",
         {"BlogCatalog": [(.2588, .0130), (.1198, .0254), (.0825, .0065)],
          "Flickr": [(.2869, .0130), (.1577, .0192), (.0859, .0080)]}),
        ("Effects depend only on the\none-hop neighbor mean (Exp. 4)",
         "fixed: aggregate",
         {"BlogCatalog": [(.2221, .0093), (.0383, .0042), (.1146, .0135)],
          "Flickr": [(.2504, .0075), (.0578, .0012), (.1140, .0154)]}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, (title, fixed_label, data) in zip(axes, panels):
        labels = ["raw TabPFN (no graph)", fixed_label, "learned PNA adapter"]
        colors = [RAW, FIXED, ADAPTER]
        y0 = 0
        yt, yl = [], []
        for g, vals in data.items():
            for k, (m, s) in enumerate(vals):
                y = y0 + k * 0.8
                ax.barh(y, m, xerr=s, height=.65, color=colors[k],
                        error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "0.3"},
                        label=labels[k] if g == "BlogCatalog" else None)
                ax.text(m + s + .005, y, f"{m:.3f}", va="center", fontsize=8.5)
            yt.append(y0 + .8)
            yl.append(g)
            y0 += 3.2
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel("individual-effect PEHE (lower is better)")
    ax.set_xlim(0, .33)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [RAW, FIXED, ADAPTER]]
    fig.legend(handles, ["raw TabPFN (no graph)", "best fixed graph features", "learned PNA adapter"],
               loc="lower center", ncol=3, bbox_to_anchor=(.5, -.08), fontsize=9.5)
    fig.tight_layout(w_pad=3)
    save(fig, "hero.png")


def toy_dispersion():
    """Illustration: identical neighbor means, different neighbor spread."""
    rng = np.random.default_rng(3)
    a = .5 + rng.normal(0, .03, 8)
    b = np.r_[rng.normal(.08, .03, 4), rng.normal(.92, .03, 4)]
    a, b = a - a.mean() + .5, b - b.mean() + .5
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    for y, vals, name in [(1, a, "node u"), (0, b, "node v")]:
        ax.scatter(vals, np.full_like(vals, y) + rng.uniform(-.08, .08, len(vals)),
                   s=55, color=FIXED2, edgecolor=FIXED, zorder=3)
        ax.plot([.5, .5], [y - .25, y + .25], color=FIXED, lw=2.5)
        ax.text(-.03, y, name, ha="right", va="center", fontsize=10, weight="bold")
        ax.text(1.04, y, f"mean {vals.mean():.2f}   sd {vals.std(ddof=1):.2f}",
                va="center", fontsize=9.5)
    ax.set_xlim(-.2, 1.45)
    ax.set_ylim(-.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("neighbors' covariate value  (blue bar = neighbor mean)")
    save(fig, "toy_dispersion.png")


def exp1():
    """Exp. 1 PEHE relative to raw TabPFN-v3, by network-confounding strength."""
    kappa = [.5, 1, 2]
    d = {  # BlogCatalog then Flickr
        "raw TabPFN": ([5.82, 7.60, 17.92], [7.65, 13.82, 25.87], RAW, "o", "-"),
        "TabPFN + diffusion": ([3.70, 4.59, 8.62], [4.45, 6.64, 12.16], FIXED, "s", "-"),
        "TabPFN + aggsumlog": ([2.88, 3.31, 5.24], [4.30, 5.74, 9.75], FIXED2, "s", "--"),
        "GCN adapter": ([3.00, 3.52, 6.11], [3.88, 5.53, 9.48], "#f08c00", "^", "--"),
        "PNA adapter": ([3.09, 3.47, 6.11], [3.80, 5.19, 8.91], ADAPTER, "D", "-"),
        "NetDeconfounder": ([4.75, 4.25, 8.92], [4.33, 5.57, 9.53], BASE, "v", "-"),
        "GraphPFN": ([5.67, 5.49, 14.17], [4.97, 7.67, 13.81], OTHER, "P", ":"),
    }
    raw_bc, raw_fl = d["raw TabPFN"][0], d["raw TabPFN"][1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, g, idx, raw in [(axes[0], "BlogCatalog", 0, raw_bc), (axes[1], "Flickr", 1, raw_fl)]:
        for name, (bc, fl, c, mk, ls) in d.items():
            v = np.array([bc, fl][idx]) / np.array(raw)
            ax.plot(range(3), v, color=c, marker=mk, ls=ls, lw=1.8, ms=6, label=name)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"κ = {k}" for k in kappa])
        ax.set_title(g, loc="left")
        ax.axhline(1, color="0.8", lw=.8, zorder=0)
        ax.set_ylim(.25, 1.1)
    axes[0].set_ylabel("PEHE / PEHE of raw TabPFN")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "exp1_relative.png")


def exp1_input_matched():
    """Same graph features, different predictors (Exp. 1, κ = 2)."""
    preds = ["TabPFN-v3", "Ridge", "HistGB", "XGBoost"]
    feats = ["raw", "aggregate", "aggsumlog", "diffusion"]
    vals = {
        "BlogCatalog, κ = 2": [[17.92, 8.61, 5.24, 8.62], [19.40, 19.17, 19.33, 18.99],
                               [21.41, 23.53, 23.62, 23.56], [20.64, 21.54, 23.30, 21.55]],
        "Flickr, κ = 2": [[25.87, 13.20, 9.75, 12.16], [26.54, 26.52, 25.73, 26.21],
                          [28.47, 30.47, 31.82, 29.79], [27.86, 28.33, 32.38, 28.58]],
    }
    cols = [RAW, FIXED2, "#1864ab", FIXED]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
    w = .2
    for ax, (title, rows) in zip(axes, vals.items()):
        for j, f in enumerate(feats):
            ax.bar(np.arange(4) + (j - 1.5) * w, [r[j] for r in rows], w, color=cols[j], label=f)
        ax.set_xticks(range(4))
        ax.set_xticklabels(preds)
        ax.set_title(title, loc="left")
        ax.set_ylabel("PEHE")
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=3)
    fig.legend(h, lab, title="graph features", loc="lower center", ncol=4,
               bbox_to_anchor=(.5, -.12), fontsize=9, title_fontsize=9)
    save(fig, "exp1_input_matched.png")


EXP2 = {  # BC ind, BC peer, FL ind, FL peer
    "TabPFN + aggregate": (.0077, .0327, .0091, .0127, FIXED2, "s"),
    "TabPFN + diffusion": (.0085, .0306, .0092, .0139, FIXED, "s"),
    "GCN adapter": (.0082, .0318, .0085, .0136, "#ffa94d", "^"),
    "GAT adapter": (.0085, .0304, .0085, .0123, "#ffa94d", "<"),
    "GIN adapter": (.0080, .0309, .0092, .0166, "#ffa94d", ">"),
    "PNA adapter": (.0077, .0301, .0088, .0172, ADAPTER, "D"),
    "GraphPFN": (.0237, .0815, .0261, .0712, OTHER, "P"),
    "NetDeconfounder": (.0214, .0873, .0248, .0209, BASE, "v"),
    "NetEstimator": (.0224, .1777, .0422, .0675, BASE, "X"),
    "RRNet": (.0215, .0768, .0272, .0277, BASE, "h"),
    "TARNet": (.0251, .0729, .0230, .0273, "#69db7c", "o"),
    "CFR": (.0104, .0438, .0153, .0118, "#69db7c", "*"),
}
EXP3 = {
    "TabPFN + aggregate": (.1347, .1224, .2137, .1381, FIXED2, "s"),
    "TabPFN + diffusion": (.1198, .1167, .1577, .1244, FIXED, "s"),
    "GCN adapter": (.1528, .1311, .2212, .1459, "#ffa94d", "^"),
    "GAT adapter": (.1425, .1231, .1924, .1297, "#ffa94d", "<"),
    "GIN adapter": (.1586, .1261, .2231, .1427, "#ffa94d", ">"),
    "PNA adapter": (.0825, .1175, .0859, .0864, ADAPTER, "D"),
    "GraphPFN": (.1741, .2031, .2343, .1861, OTHER, "P"),
    "NetDeconfounder": (.2070, .1553, .2538, .1540, BASE, "v"),
    "NetEstimator": (.2171, .2637, .2765, .1903, BASE, "X"),
    "RRNet": (.2129, .2143, .2775, .1876, BASE, "h"),
    "TARNet": (.2659, .1649, .2868, .1582, "#69db7c", "o"),
    "CFR": (.2669, .2534, .2878, .2423, "#69db7c", "*"),
    "oracle (privileged)": (.0264, .0812, .0311, .0434, ORACLE, "x"),
}


def scatter_exp23():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for r, (exp, data) in enumerate([("Exp. 2 (constant effects)", EXP2),
                                     ("Exp. 3 (dispersion-sensitive effects)", EXP3)]):
        for c, g in enumerate(["BlogCatalog", "Flickr"]):
            ax = axes[r, c]
            for name, v in data.items():
                x, y = v[2 * c], v[2 * c + 1]
                ax.scatter(x, y, color=v[4], marker=v[5], s=70 if "PNA" in name else 45,
                           edgecolor="k" if "PNA" in name else None, lw=.6, label=name, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
                axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))
                axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"{exp} — {g}", loc="left", fontsize=10)
            ax.set_xlabel("individual-effect PEHE")
            ax.set_ylabel("peer-effect PEHE")
            ax.grid(True, which="both", color="0.92", lw=.6)
    h, lab = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, .8, 1))
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(.8, .5), fontsize=9)
    save(fig, "scatter_exp2_exp3.png")


def ladder():
    labels = ["raw", "one-hop\nmean", "mean +\ndegree", "diffusion\n(1–3 hops)", "PNA\nadapter", "oracle"]
    bc = [.2588, .1347, .1384, .1198, .0825, .0264]
    fl = [.2869, .2137, .2164, .1577, .0859, .0311]
    cols = [RAW, FIXED2, FIXED2, FIXED, ADAPTER, ORACLE]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for vals, ls, g in [(bc, "-", "BlogCatalog"), (fl, "--", "Flickr")]:
        ax.plot(range(6), vals, color="0.6", ls=ls, lw=1.3, zorder=1, label=g)
        ax.scatter(range(6), vals, c=cols, s=55, zorder=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_ylabel("individual-effect PEHE")
    ax.set_ylim(0, .31)
    ax.legend(fontsize=9)
    save(fig, "ladder_exp3.png")


def node_level():
    z = np.load(ROOT / "results/interference/effect_diagnostics/Flickr_seed0_hard.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in [(axes[0], "ind", "Individual effect"), (axes[1], "peer", "Peer effect")]:
        t = z[f"truth_{key}"]
        for pk, c, lab in [("diff", FIXED, "TabPFN + diffusion"), ("pna", ADAPTER, "PNA adapter")]:
            p = z[f"pred_{pk}_{key}"]
            ax.scatter(t, p, s=6, alpha=.35, color=c, lw=0,
                       label=f"{lab} (PEHE {np.sqrt(np.mean((p - t) ** 2)):.3f})")
        lo, hi = np.quantile(t, [.002, .998])
        pad = (hi - lo) * .15
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("true effect")
        ax.set_ylabel("estimated effect")
        ax.set_title(title, loc="left")
        leg = ax.legend(fontsize=8.5, loc="upper left", markerscale=3)
        for h in leg.legend_handles:
            h.set_alpha(1)
    fig.tight_layout(w_pad=3)
    save(fig, "node_level_flickr.png")


def ablation():
    rows = [
        ("Full adapter → refit TabPFN", .075, .087, ADAPTER),
        ("No MMD penalty", .081, .085, "#ffa94d"),
        ("No diffusion warm start", .086, .095, "#ffa94d"),
        ("6-dim representation (vs 24)", .081, .099, "#ffa94d"),
        ("128-row training context (vs 700)", .062, .079, "#ffa94d"),
        ("Predict with the training decoder", .090, .100, "#ced4da"),
        ("Predict with HistGB instead of TabPFN", .136, .135, "#adb5bd"),
        ("Train encoder through an MLP decoder", .206, .251, "#495057"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, j, g in [(axes[0], 1, "BlogCatalog"), (axes[1], 2, "Flickr")]:
        y = np.arange(len(rows))
        ax.barh(y, [r[j] for r in rows], color=[r[3] for r in rows], height=.7)
        for k, r in enumerate(rows):
            ax.text(r[j] + .003, k, f"{r[j]:.3f}", va="center", fontsize=8.5)
        ax.axvline(rows[0][j], color=ADAPTER, lw=.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(g, loc="left")
        ax.set_xlabel("individual-effect PEHE")
        ax.set_xlim(0, .29)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    save(fig, "ablation.png")


def label_budget():
    d = json.load(open(ROOT / "results/interference/label_budget/hard_v3.json"))["datasets"]
    methods = [("frozen_diffusion", "TabPFN + diffusion", FIXED, "s"),
               ("pna_tabpfn", "PNA adapter", ADAPTER, "D"),
               ("histgb_diffusion", "HistGB + diffusion", BASE, "o"),
               ("ridge_diffusion", "Ridge + diffusion", RAW, "v")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for c, g in enumerate(["BC", "Flickr"]):
        fracs = sorted(d[g], key=float)
        x = np.arange(len(fracs))
        for r, est in enumerate(["individual_pehe", "peer_pehe"]):
            ax = axes[r, c]
            for key, lab, col, mk in methods:
                v = np.array([[s[key][est] for s in d[g][f]["per_seed"]] for f in fracs])
                ax.errorbar(x, v.mean(1), yerr=v.std(1, ddof=1), color=col, marker=mk,
                            ms=5, lw=1.6, capsize=2, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{round(100 * float(f))}%" for f in fracs])
            name = "BlogCatalog" if g == "BC" else "Flickr"
            ax.set_title(f"{name} — {'individual' if r == 0 else 'peer'} effect", loc="left", fontsize=10)
            ax.set_ylabel("PEHE")
            if r == 1:
                ax.set_xlabel("share of nodes with an observed outcome")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.05), fontsize=9.5)
    save(fig, "label_budget.png")


if __name__ == "__main__":
    hero()
    toy_dispersion()
    exp1()
    exp1_input_matched()
    scatter_exp23()
    ladder()
    node_level()
    ablation()
    label_budget()
