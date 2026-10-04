"""Figures 2-4 and the table rows of the paper, built only from results/sweep_*.csv.

Usage:
    python make_figures.py            # uses whichever of 2x2, 2x3, 2x4 exist

Outputs:
    figures/fig2_validity.pdf   single column, largest lattice available
    figures/fig3_sensitivity.pdf  full width, one panel per lattice
    figures/fig4_family.pdf     full width, number line + residual check
    results/tables.tex                Table I and Table II rows
    results/summary.txt               every number quoted in the text, with its source
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
LATTICES = ["2x2", "2x3", "2x4"]
QUBITS = {"2x2": 8, "2x3": 12, "2x4": 16}
DIM = {"2x2": 36, "2x3": 400, "2x4": 4900}
PROXIES = ["free_fermion", "neel", "gutzwiller"]
LABEL = {"free_fermion": "Free-fermion", "neel": "Néel", "gutzwiller": "Gutzwiller"}
# Categorical slots 1-3 of the dataviz reference palette (validated: CVD and normal-vision
# checks pass; contrast warning is relieved by marker shapes, legend and direct labels).
COLOR = {"free_fermion": "#2a78d6", "neel": "#eb6834", "gutzwiller": "#1baf7a"}
MARKER = {"free_fermion": "o", "neel": "s", "gutzwiller": "^"}
U_RAMP = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue ramp for U/t
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load():
    data = {}
    for lat in LATTICES:
        f = RES / f"sweep_{lat}.csv"
        if f.exists():
            with open(f) as fh:
                data[lat] = list(csv.DictReader(fh))
    if not data:
        raise SystemExit("no results/sweep_*.csv files found; run run_sweep.py first")
    return data


def num(r, k):
    return float(r[k])


def rows(data, lat, state):
    return sorted((r for r in data[lat] if r["state"] == state), key=lambda r: num(r, "u"))


def fig2(data, lat):
    dts = [0.005, 0.01, 0.02, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.axhspan(-5, 5, color=GRID, alpha=0.6, lw=0)
    for color, r in zip(U_RAMP, rows(data, lat, "exact_gs")):
        c_pt = num(r, "c_pennylane")
        dev = [100 * (num(r, f"dE_over_dt2_{dt}") / c_pt - 1) for dt in dts]
        ax.plot(dts, dev, color=color, marker="o", ms=4, label=f"$U/t={num(r, 'u'):g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"step size $\delta t$")
    ax.set_ylabel(r"$\Delta E/\delta t^2$ vs. estimate (%)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(FIG / "fig2_validity.pdf")
    fig.savefig(FIG / "fig2_validity.png", dpi=200)
    plt.close(fig)


def fig3(data):
    lats = list(data)
    fig, axes = plt.subplots(1, len(lats), figsize=(7.0, 2.1), sharey=True, squeeze=False)
    for ax, lat in zip(axes[0], lats):
        ax.axhline(0, color=MUTED, lw=0.8)
        for s in PROXIES:
            rr = rows(data, lat, s)
            u = [num(r, "u") for r in rr]
            err = [100 * (num(r, "c_standard") - num(r, "c_exact")) / num(r, "c_exact") for r in rr]
            ax.plot(u, err, color=COLOR[s], lw=1.4, zorder=2)
            for x, y, r in zip(u, err, rr):
                wrong = np.sign(num(r, "c_standard")) != np.sign(num(r, "c_exact"))
                ax.plot(x, y, MARKER[s], ms=5, color=COLOR[s], mfc="white" if wrong else COLOR[s],
                        mec=COLOR[s], mew=1.2, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("$U/t$")
        ax.set_title(f"{lat} ({QUBITS[lat]} qubits)", fontsize=8, color=INK)
    axes[0][0].set_ylabel("signed relative error of $c$ (%)")
    handles = [plt.Line2D([], [], color=COLOR[s], marker=MARKER[s], ms=5, label=LABEL[s]) for s in PROXIES]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", ms=5, mfc="white", ls="", label="wrong sign"))
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig3_sensitivity.pdf")
    fig.savefig(FIG / "fig3_sensitivity.png", dpi=200)
    plt.close(fig)


def fig4(data, lat):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y, ticks, labels = 0, [], []
    forms = [("c_formB", "D", r"$c_0$"), ("c_formA", "v", r"$c_1$"), ("c_half", "o", r"$c_{1/2}$"),
             ("c_standard", "x", r"standard $c_{-1}$")]
    for s in PROXIES:
        for r in rows(data, lat, s):
            ce = num(r, "c_exact")
            vals = [num(r, k) / ce for k, _, _ in forms]
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=3, zorder=1)
            hw = num(r, "half_width") / ce
            ax.plot([vals[2] - hw, vals[2] + hw], [y, y], color=COLOR[s], lw=2, zorder=2)
            for (k, m, _), v in zip(forms, vals):
                ax.plot(v, y, m, ms=4.5, color=COLOR[s], mfc="white" if m in "Dv" else COLOR[s], zorder=3)
            ax.plot(1, y, "*", ms=8, color=INK, zorder=4)
            ticks.append(y)
            labels.append(f"{LABEL[s]}, $U/t$={num(r, 'u'):g}")
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$c_\lambda / c_{\mathrm{exact}}$ (" + lat + ")")
    hl = [plt.Line2D([], [], color=MUTED, marker=m, ls="", ms=5, mfc="white" if m in "Dv" else MUTED, label=lab)
          for _, m, lab in forms]
    hl.append(plt.Line2D([], [], color=INK, marker="*", ls="", ms=8, label=r"$c_{\mathrm{exact}}$"))
    ax.legend(handles=hl, frameon=False, fontsize=6.5, loc="lower right")

    for s in PROXIES:
        xs, ys = [], []
        for lat2 in data:
            for r in rows(data, lat2, s):
                ce = abs(num(r, "c_exact"))
                xs.append(num(r, "half_width") / ce)
                ys.append(abs(num(r, "c_half") - num(r, "c_exact")) / ce)
        bx.plot(xs, ys, MARKER[s], ms=5, color=COLOR[s], label=LABEL[s], ls="")
    lim = [1e-3, 4]
    bx.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    bx.text(1.3e-3, 2.4e-3, "error $= |r|/48$", color=MUTED, fontsize=6.5)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_xlabel(r"$|r|/48$ relative to $c_{\mathrm{exact}}$")
    bx.set_ylabel(r"$|c_{1/2}-c_{\mathrm{exact}}|$ relative")
    bx.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig4_family.pdf")
    fig.savefig(FIG / "fig4_family.png", dpi=200)
    plt.close(fig)


def tables(data):
    out, summ = [], []
    out.append("% Table I rows: Lattice & Qubits & Sector dim. & Exact (s) & Estimate (s) & T/step")
    for lat in data:
        rr = [r for r in data[lat] if r["state"] == "exact_gs"]
        ex = np.mean([num(r, "seconds_exact") for r in rr])
        pl = np.mean([num(r, "seconds_pennylane") for r in rr])
        out.append(f"${lat.replace('x', r' \times ')}$ & {QUBITS[lat]} & {DIM[lat]} & {ex:.1f} & {pl:.2f} & \\todo{{}} \\\\")
    out.append("% Table II rows: State & Sign (std.) & Max error std. & Max error c_half & c_exact in interval")
    for s in PROXIES:
        rr = [r for lat in data for r in data[lat] if r["state"] == s]
        sign_ok = sum(np.sign(num(r, "c_standard")) == np.sign(num(r, "c_exact")) for r in rr)
        e_std = max(abs(num(r, "c_standard") / num(r, "c_exact") - 1) for r in rr)
        e_half = max(abs(num(r, "c_half") / num(r, "c_exact") - 1) for r in rr)
        inside = sum(r["in_interval"] == "True" for r in rr)
        out.append(f"{LABEL[s]} & {sign_ok}/{len(rr)} & {100 * e_std:.0f}\\% & {100 * e_half:.0f}\\% & {inside}/{len(rr)} \\\\")
    (RES / "tables.tex").write_text("\n".join(out) + "\n", encoding="utf-8")

    summ.append(f"lattices: {', '.join(data)}")
    pe = [abs(num(r, "c_pennylane") / num(r, "c_exact") - 1) for lat in data for r in data[lat] if r["state"] == "exact_gs"]
    summ.append(f"exact-gs estimate vs exact, max relative error: {max(pe):.2e}")
    for key, name in (("c_bound", "commutator bound"), ("norm_E", "||E||")):
        ratios = [np.sqrt(num(r, key) / abs(num(r, "c_exact"))) for lat in data for r in data[lat] if r["state"] == "exact_gs"]
        summ.append(f"{name} step ratio range: {min(ratios):.2f} to {max(ratios):.2f}")
    for s in PROXIES:
        rr = [r for lat in data for r in data[lat] if r["state"] == s]
        summ.append(f"{s}: max |err| standard {max(abs(num(r, 'c_standard') / num(r, 'c_exact') - 1) for r in rr):.3f}, "
                    f"c_half {max(abs(num(r, 'c_half') / num(r, 'c_exact') - 1) for r in rr):.3f}, "
                    f"formA {max(abs(num(r, 'c_formA') / num(r, 'c_exact') - 1) for r in rr):.3f}, "
                    f"formB {max(abs(num(r, 'c_formB') / num(r, 'c_exact') - 1) for r in rr):.3f}, "
                    f"in interval {sum(r['in_interval'] == 'True' for r in rr)}/{len(rr)}")
    allr = [r for lat in data for r in data[lat] if r["state"] in PROXIES]
    x = np.log([num(r, "half_width") / abs(num(r, "c_exact")) for r in allr])
    yv = np.log([max(1e-12, abs(num(r, "c_half") / num(r, "c_exact") - 1)) for r in allr])
    summ.append(f"log-log correlation of |r|/48 with |c_half error| over {len(allr)} cases: {np.corrcoef(x, yv)[0, 1]:.2f}")
    (RES / "summary.txt").write_text("\n".join(summ) + "\n", encoding="utf-8")
    print("\n".join(summ))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    big = list(data)[-1]
    fig2(data, big)
    fig3(data)
    fig4(data, big)
    tables(data)
    print(f"figures written to {FIG} using {', '.join(data)}")


if __name__ == "__main__":
    main()
