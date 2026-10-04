"""Figures 2-5 and the table rows of the paper, built only from results/sweep_*.csv.

Figures are drawn at their printed size (IEEE column 3.5 in, text width 7.16 in) in Times with
STIX mathematics, matching the IEEEtran body text.

Usage:
    python make_figures.py

Outputs:
    figures/fig2_validity.pdf     single column
    figures/fig3_sensitivity.pdf  full width, one panel per lattice
    figures/fig4_family.pdf       full width, family of forms on the 2x4 ladder
    figures/fig5_check.pdf        single column, residual check against actual error
    results/tables.tex            Table I and Table II rows
    results/summary.txt           every number quoted in the text
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
PROXIES = ["free_fermion", "neel", "gutzwiller", "uhf"]
LABEL = {"free_fermion": "Free-fermion", "neel": "Néel", "gutzwiller": "Gutzwiller", "uhf": "UHF"}
COLOR = {"free_fermion": "C0", "neel": "C1", "gutzwiller": "C2", "uhf": "C3"}
MARKER = {"free_fermion": "o", "neel": "s", "gutzwiller": "^", "uhf": "D"}
U_COLOR = ["C0", "C1", "C2", "C3"]
COL, TEXT = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "lines.linewidth": 1.2, "lines.markersize": 5,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "legend.borderpad": 0.3,
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


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def fig2(data, lat):
    dts = [0.005, 0.01, 0.02, 0.05, 0.1]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    for color, r in zip(U_COLOR, rows(data, lat, "exact_gs")):
        c_pt = num(r, "c_pennylane")
        dev = [100 * (num(r, f"dE_over_dt2_{dt}") / c_pt - 1) for dt in dts]
        ax.plot(dts, dev, color=color, marker="o", label=f"$U/t={num(r, 'u'):g}$")
    ax.axhline(5, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel(r"Trotter step $\delta t$")
    ax.set_ylabel(r"$(\Delta E/\delta t^2)/c_{\mathrm{PT}} - 1$ (%)")
    ax.legend(loc="upper left", ncol=2, handlelength=1.8, columnspacing=1.0)
    save(fig, "fig2_validity")


def fig3(data):
    lats = list(data)
    fig, axes = plt.subplots(1, len(lats), figsize=(TEXT, 2.0), sharey=True, squeeze=False)
    for ax, lat in zip(axes[0], lats):
        ax.axhline(1, color="k", lw=0.8, ls="--")
        for s in PROXIES:
            rr = rows(data, lat, s)
            u = [num(r, "u") for r in rr]
            ratio = [abs(num(r, "c_standard") / num(r, "c_exact")) for r in rr]
            ax.plot(u, ratio, color=COLOR[s], lw=1.0)
            for x, y, r in zip(u, ratio, rr):
                wrong = np.sign(num(r, "c_standard")) != np.sign(num(r, "c_exact"))
                ax.plot(x, y, MARKER[s], color=COLOR[s], mfc="white" if wrong else COLOR[s], mew=1.1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
        ax.minorticks_off()
        ax.set_ylim(0.02, 15)
        ax.set_xlabel("$U/t$")
        ax.set_title(f"{lat.replace('x', r'$\times$')} ({QUBITS[lat]} qubits)")
    axes[0][0].set_ylabel(r"$|c_{-1}|\,/\,c_{\mathrm{exact}}$")
    handles = [plt.Line2D([], [], color=COLOR[s], marker=MARKER[s], label=LABEL[s]) for s in PROXIES]
    handles.append(plt.Line2D([], [], color="k", marker="o", mfc="white", ls="", label="wrong sign"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig3_sensitivity")


def fig4(data, lat):
    lo, hi = -1.7, 3.2
    forms = [("c_formB", "D", r"$c_0$"), ("c_formA", "v", r"$c_1$"), ("c_half", "o", r"$c_{1/2}$"),
             ("c_standard", "x", r"$c_{-1}$ (standard)")]
    fig, axes = plt.subplots(1, len(PROXIES), figsize=(TEXT, 2.0), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes[0], PROXIES):
        ax.axvline(1, color="k", lw=0.8, ls="--")
        for y, r in enumerate(rows(data, lat, s)):
            ce = num(r, "c_exact")
            vals = [num(r, k) / ce for k, _, _ in forms]
            hw = num(r, "half_width") / ce
            a, b = max(lo, vals[2] - hw), min(hi, vals[2] + hw)
            ax.plot([a, b], [y, y], color=COLOR[s], lw=1.0)
            ax.plot([a, a], [y - 0.15, y + 0.15], color=COLOR[s], lw=1.0)
            ax.plot([b, b], [y - 0.15, y + 0.15], color=COLOR[s], lw=1.0)
            n_off = 0
            for (_, m, _), v in zip(forms, vals):
                vc = min(max(v, lo + 0.06), hi - 0.06)
                ax.plot(vc, y, m, color=COLOR[s], mfc="white" if m in "Dv" else COLOR[s], mew=1.1,
                        ms=5.5 if m != "x" else 6)
                if v > hi or v < lo:
                    dy = 5 if n_off == 0 else -11
                    n_off += 1
                    ax.annotate(f"{m if m != 'x' else 'std'}: {v:.1f}".replace("D:", r"$c_0$:").replace("std:", r"$c_{-1}$:"),
                                (vc, y), xytext=(-3 if v > hi else 3, dy), textcoords="offset points",
                                ha="right" if v > hi else "left", fontsize=7)
        ax.set_yticks(range(4), ["1", "2", "4", "8"])
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(lo, hi)
        ax.set_title(LABEL[s])
        ax.set_xlabel(r"$c_\lambda / c_{\mathrm{exact}}$")
    axes[0][0].set_ylabel("$U/t$")
    hl = [plt.Line2D([], [], color="k", marker=m, ls="", mfc="white" if m in "Dv" else "k", label=lab)
          for _, m, lab in forms]
    hl.append(plt.Line2D([], [], color="k", lw=1.0, label=r"$c_{1/2} \pm |r|/48$"))
    fig.legend(handles=hl, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.09), fontsize=8)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig4_family")


def fig5(data):
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    lim = [1e-3, 5]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    for s in PROXIES:
        xs, ys = [], []
        for lat in data:
            for r in rows(data, lat, s):
                ce = abs(num(r, "c_exact"))
                xs.append(num(r, "half_width") / ce)
                ys.append(max(1.2e-3, abs(num(r, "c_half") - num(r, "c_exact")) / ce))
        ax.plot(xs, ys, MARKER[s], color=COLOR[s], mfc="none", mew=1.1, ls="", label=LABEL[s])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"$|r|/(48\,c_{\mathrm{exact}})$")
    ax.set_ylabel(r"$|c_{1/2} - c_{\mathrm{exact}}|\,/\,c_{\mathrm{exact}}$")
    ax.legend(loc="upper left", handletextpad=0.2)
    save(fig, "fig5_check")


def pct(x):
    return f"{100 * x:.0f}\\%"


def tables(data):
    out, summ = [], []
    out.append("% Table I rows: Lattice & Qubits & Sector dim. & Exact (s) & Estimate (s)")
    for lat in data:
        rr = [r for r in data[lat] if r["state"] == "exact_gs"]
        ex = np.mean([num(r, "seconds_exact") for r in rr])
        pl = np.mean([num(r, "seconds_pennylane") for r in rr])
        out.append(f"${lat.replace('x', r' \times ')}$ & {QUBITS[lat]} & {DIM[lat]} & {ex:.1f} & {pl:.2f} \\\\")
    out.append("% Table II rows: State & Sign & Std. & c_half & |r|/48 range & Inside")
    for s in PROXIES:
        rr = [r for lat in data for r in data[lat] if r["state"] == s]
        sign_ok = sum(np.sign(num(r, "c_standard")) == np.sign(num(r, "c_exact")) for r in rr)
        e_std = max(abs(num(r, "c_standard") / num(r, "c_exact") - 1) for r in rr)
        e_half = max(abs(num(r, "c_half") / num(r, "c_exact") - 1) for r in rr)
        hw = [num(r, "half_width") / abs(num(r, "c_exact")) for r in rr]
        inside = sum(r["in_interval"] == "True" for r in rr)
        out.append(f"{LABEL[s]} & {sign_ok}/{len(rr)} & {pct(e_std)} & {pct(e_half)} & "
                   f"{min(hw):.2f}--{max(hw):.2f} & {inside}/{len(rr)} \\\\")
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
    for group, states in (("without UHF", PROXIES[:3]), ("all four", PROXIES)):
        allr = [r for lat in data for r in data[lat] if r["state"] in states]
        x = np.log([num(r, "half_width") / abs(num(r, "c_exact")) for r in allr])
        yv = np.log([max(1e-12, abs(num(r, "c_half") / num(r, "c_exact") - 1)) for r in allr])
        summ.append(f"log-log correlation of |r|/48 with |c_half error|, {group} ({len(allr)} cases): "
                    f"{np.corrcoef(x, yv)[0, 1]:.2f}")
    (RES / "summary.txt").write_text("\n".join(summ) + "\n", encoding="utf-8")
    print("\n".join(summ))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    big = list(data)[-1]
    fig2(data, big)
    fig3(data)
    fig4(data, big)
    fig5(data)
    tables(data)
    print(f"figures written to {FIG} using {', '.join(data)}")


if __name__ == "__main__":
    main()
