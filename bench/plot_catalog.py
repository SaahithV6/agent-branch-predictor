"""Figure for the catalog scale test: bench/results/catalog/scale.json -> docs/catalog_scale.png

Usage: uv run --with matplotlib python3 -I bench/plot_catalog.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = [r for r in json.load(open(os.path.join(R, "bench/results/catalog/scale.json"))) if not r.get("trained_with_catalog")]
trained = next(r for r in json.load(open(os.path.join(R, "bench/results/catalog/scale.json"))) if r.get("trained_with_catalog"))
INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#6b6a64", "#e4e3de", "#fcfcfb", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

x = [max(r["catalog_tools"], 300) for r in rows]  # 0 drawn at the left edge of the log axis
labels = ["none" if r["catalog_tools"] == 0 else f"{r['catalog_tools'] // 1000:,}k" if r["catalog_tools"] < 10 ** 6 else "1M" for r in rows]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xscale("log")
    ax.set_xticks(x, labels)
    ax.minorticks_off()
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0)
    ax.set_xlabel("tools in the catalog (real definitions; 1M adds tenant copies)")

a.plot(x, [r["coverage"] * 100 for r in rows], color=BLUE, lw=2, marker="o", ms=8, zorder=3, label="lookups predicted")
a.plot(x, [r["precision"] * 100 for r in rows], color=ORANGE, lw=2, marker="o", ms=8, zorder=3, label="guesses right")
a.plot([10 ** 6], [trained["coverage"] * 100], marker="D", ms=8, color=BLUE, mfc=SURF, mew=2, zorder=4)
a.annotate("trained with the 1M catalog", (10 ** 6, trained["coverage"] * 100), (0.45 * 10 ** 6, 66), fontsize=8.5, color=MUTED,
           ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set_ylim(50, 100)
a.set_ylabel("%")
a.set_title("Retail accuracy, held-out customers", loc="left", fontsize=11, fontweight="bold")
a.legend(loc="lower left", frameon=False, fontsize=9)
a.text(0.99, 0.03, "catalog tools guessed: 0 at every size", transform=a.transAxes, ha="right", fontsize=8.5, color=MUTED)

b.plot(x, [r["ms_per_prediction"] for r in rows], color=BLUE, lw=2, marker="o", ms=8, zorder=3)
for xi, r in zip(x, rows):
    b.text(xi, r["ms_per_prediction"] + 1.2, f"{r['ms_per_prediction']:.1f}", ha="center", fontsize=8.5, color=INK)
b.set_ylim(0, 30)
b.set_ylabel("ms per prediction")
b.set_title("Cost per prediction", loc="left", fontsize=11, fontweight="bold")
b.text(0.99, 0.03, "at most 128 catalog candidates scored per prediction", transform=b.transAxes, ha="right", fontsize=8.5, color=MUTED)

fig.suptitle("A million tools in the catalog: accuracy unchanged, cost flat", x=0.012, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(R, "docs/catalog_scale.png"), dpi=160, facecolor=SURF)
