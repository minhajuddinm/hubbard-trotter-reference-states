"""Fig. 1: routes to the second-order Trotter error coefficient c (schematic)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, EDGE, FILL, ACCENT = "#1f1f1e", "#6b6a63", "#b9b8ae", "#f3f2ee", "#2a78d6"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 7, "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, body, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#e8f1fc" if accent else FILL, ec=ACCENT if accent else EDGE, lw=0.9))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=6.8, color=INK, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.035, body, ha="center", va="center", fontsize=6.2, color=INK,
            linespacing=1.35)


def arrow(ax, p, q, text=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.01, (p[1] + q[1]) / 2, text, fontsize=6, color=MUTED, va="center")


fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1)
ax.axis("off")

box(ax, 0.30, 0.83, 0.40, 0.15, "Hamiltonian", r"$H = T + V$,  $U_2(\delta t)$")
box(ax, 0.00, 0.40, 0.31, 0.33, "Bounds", "state-independent\n" + r"$c_{\mathrm{bound}}$, $\|\mathcal{E}\|$")
box(ax, 0.345, 0.40, 0.31, 0.33, "Exact", "eigenphase of $U_2$\nfollowing $\\psi_0$\n" + r"$c_{\mathrm{exact}}$")
box(ax, 0.69, 0.40, 0.31, 0.33, "Perturbative", r"$\langle\psi|\mathcal{E}|\psi\rangle$" + "\nneeds a reference\nstate $\\psi$", accent=True)
box(ax, 0.40, 0.00, 0.60, 0.30, "This paper",
    r"$\psi \in \{\psi_0$, free-fermion, Néel," + "\nGutzwiller" + r"$\}$; family $c_\lambda$" + "\n"
    + r"estimate $c_{1/2}$, check $|r|/48$", accent=True)
box(ax, 0.00, 0.00, 0.34, 0.30, "Step count", r"steps $\propto \sqrt{|c|}$" + "\nT gates = steps\n" + r"$\times$ T/step")

for x in (0.155, 0.50, 0.845):
    arrow(ax, (0.50, 0.83), (x, 0.73))
arrow(ax, (0.845, 0.40), (0.80, 0.30))
arrow(ax, (0.40, 0.15), (0.34, 0.15))
arrow(ax, (0.155, 0.40), (0.155, 0.30))
arrow(ax, (0.50, 0.40), (0.30, 0.30))

fig.savefig(ROOT / "figures" / "fig1_routes.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(ROOT / "figures" / "fig1_routes.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("fig1 written")
