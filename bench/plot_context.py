"""Figure for the sentence-context features: bench/results/gate/context.json -> docs/context.png

Usage: uv run --with matplotlib python3 -I bench/plot_context.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
j = json.load(open(os.path.join(R, "bench/results/gate/context.json")))
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6a64", "#e4e3de", "#fcfcfb"
BEFORE, AFTER = "#b9b7ae", "#2a78d6"  # baseline recedes; the new model carries the color
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]}, facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

# left: share of reads predicted, per domain, at the same precision target
doms = ["retail", "airline", "telecom"]
w = 0.34
for i, d in enumerate(doms):
    for k, (arm, col) in enumerate((("before", BEFORE), ("context", AFTER))):
        r = j[d][arm]
        x = i + (k - 0.5) * (w + 0.08)
        a.bar(x, r["coverage"] * 100, w, color=col, zorder=2)
        a.text(x, r["coverage"] * 100 + 1.5, f"{r['coverage'] * 100:.1f}%\n{r['precision'] * 100:.0f}% right", ha="center",
               va="bottom", fontsize=8, color=INK if arm == "context" else MUTED, linespacing=1.15)
a.set_xticks(range(3), [d.capitalize() for d in doms])
a.set_ylim(0, 100)
a.yaxis.grid(True, color=GRID, zorder=0)
a.tick_params(length=0)
a.set_ylabel("lookups predicted before the agent asks (%)")
a.set_title("Held-out customers, all three domains", loc="left", fontsize=11, fontweight="bold", pad=10)

# right: retail, per tool, reads covered
t0, t1 = j["retail"]["before"]["tools"], j["retail"]["context"]["tools"]
tools = sorted(t0, key=lambda t: t0[t]["reads"])
tools = [t for t in tools if t0[t]["reads"] >= 200]
y = range(len(tools))
b.barh([v + 0.2 for v in y], [t0[t]["covered"] / t0[t]["reads"] * 100 for t in tools], 0.38, color=BEFORE, zorder=2, label="before")
b.barh([v - 0.2 for v in y], [t1[t]["covered"] / t1[t]["reads"] * 100 for t in tools], 0.38, color=AFTER, zorder=2,
       label="with sentence context")
for v, t in zip(y, tools):
    b.text(101, v, f"{t0[t]['reads']:,} reads", va="center", fontsize=8.5, color=MUTED)
b.set_yticks(list(y), [t.replace("_", " ") for t in tools])
b.set_xlim(0, 118)
b.set_xticks([0, 25, 50, 75, 100])
b.xaxis.grid(True, color=GRID, zorder=0)
b.tick_params(length=0)
b.set_xlabel("reads predicted (%)")
b.set_title("Retail, by tool", loc="left", fontsize=11, fontweight="bold", pad=10)
fig.legend(*b.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.012, 0.935), ncol=2, frameon=False, fontsize=9.5, handlelength=1.2)
fig.suptitle('"my name is Ivan Hernandez, zip 92133": knowing where a word sits in the sentence', x=0.012, ha="left",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(R, "docs/context.png"), dpi=160, facecolor=SURF)
