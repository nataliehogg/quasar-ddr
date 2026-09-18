"""Exploratory figures for the eta=0.925 requirements forecast.

Not publication style — quick-look plots for the collaboration.
"""


import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from tools.error_models import (
    SARM_ERROR_OPTIONS,
    UVX_EFFECTIVE_OPTIONS,
    resolve_sarm_error,
    resolve_uvx_error,
)
from tools.forecast import required_sample_grid

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = {
    "current": "#2a78d6",
    "future": "#008300",
    "targeted": "#e87ba4",
}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ETA = 0.925
UVX_GAMMA = 0.60
ABS_LN_ETA = abs(np.log(ETA))

SCENARIOS = {
    name: (resolve_uvx_error(name, gamma=UVX_GAMMA), resolve_sarm_error(name))
    for name in UVX_EFFECTIVE_OPTIONS
    if name in SARM_ERROR_OPTIONS
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "font.size": 11,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
style_axes(ax)

n_grid = np.geomspace(10, 4000, 400)
for name, (uvx, sarm) in SCENARIOS.items():
    sigma_ln_eta = np.hypot(uvx.sigma_ln_dl, sarm.sigma_ln_da)
    z_of_n = ABS_LN_ETA * np.sqrt(n_grid) / sigma_ln_eta
    color = SERIES[name]
    ax.plot(n_grid, z_of_n, color=color, linewidth=2, zorder=3, label=name)
    for sig in (3.0, 5.0):
        n_req = int(np.ceil((sig * sigma_ln_eta / ABS_LN_ETA) ** 2))
        ax.plot(n_req, sig, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
        if sig == 3.0:
            ax.annotate(
                f"N={n_req}", (n_req, sig), xytext=(4, -22),
                textcoords="offset points", ha="left", fontsize=9, color=INK2,
            )

for sig, lab in ((3.0, "3σ"), (5.0, "5σ")):
    ax.axhline(sig, color=INK2, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax.annotate(lab, (12, sig), xytext=(0, 4), textcoords="offset points",
                fontsize=10, color=INK2)

# direct labels at the right edge of each curve
for name, (uvx, sarm) in SCENARIOS.items():
    sigma_ln_eta = np.hypot(uvx.sigma_ln_dl, sarm.sigma_ln_da)
    z_end = ABS_LN_ETA * np.sqrt(n_grid[-1]) / sigma_ln_eta
    ax.annotate(name, (n_grid[-1], z_end), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=10,
                color=INK)

ax.set_xscale("log")
ax.set_xlim(10, 4000)
ax.set_ylim(0, 14)
ax.set_xlabel("number of same-object quasar pairs N")
ax.set_ylabel("detection significance of η = 0.925  [σ]")
ax.set_title(
    "Ideal iid detection significance vs sample size, by error scenario\n"
    "(known relative calibration; dots mark required N at 3σ and 5σ)",
    fontsize=11, loc="left",
)
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig("results/detection_significance_vs_N.png",
            bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
sig_dl = np.linspace(0.05, 0.80, 220)
sig_da = np.linspace(0.02, 0.45, 220)
grid = required_sample_grid(ETA, sig_dl, sig_da, significance=3.0)

cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
style_axes(ax)
ax.grid(False)

mesh = ax.pcolormesh(
    sig_da, sig_dl, grid, cmap=cmap, norm=LogNorm(vmin=10, vmax=3000),
    shading="auto", rasterized=True,
)
levels = [20, 60, 200, 700, 2000]
cs = ax.contour(sig_da, sig_dl, grid, levels=levels, colors=SURFACE, linewidths=1.2)
ax.clabel(cs, fmt=lambda v: f"N={v:.0f}", fontsize=9, colors=SURFACE)

for name, (uvx, sarm) in SCENARIOS.items():
    x, y = sarm.sigma_ln_da, uvx.sigma_ln_dl
    ax.plot(x, y, "o", ms=9, color=SERIES[name], mec=SURFACE, mew=2, zorder=5)
    ax.annotate(name, (x, y), xytext=(10, 4), textcoords="offset points",
                fontsize=10, color=SURFACE, fontweight="bold")

cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label("required N for a 3σ detection of η = 0.925")
cb.outline.set_visible(False)

ax.set_xlabel("per-object angular-distance error  σ(ln d$_A$)  [SARM/VLBI]")
ax.set_ylabel("per-object luminosity-distance error  σ(ln d$_L$)  [UV–X]")
ax.set_title(
    "Sample-size requirement over the per-object precision plane\n"
    "(iid errors, known relative calibration; dots = named scenarios)",
    fontsize=11, loc="left",
)
fig.tight_layout()
fig.savefig("results/requirements_grid_3sigma.png",
            bbox_inches="tight")
plt.close(fig)

print("wrote results/detection_significance_vs_N.png and results/requirements_grid_3sigma.png")
