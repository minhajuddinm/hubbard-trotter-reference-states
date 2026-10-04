"""Independent check of results/sweep_2x2.csv and results/sweep_2x3.csv.

Recomputes, with dense matrices and no shortcuts:
  * c_exact from a full eigendecomposition of U2(dt) (no inverse iteration), and
  * A = <[V,[V,T]]> and B = <[T,[T,V]]> for every reference state,
then reports the largest relative difference from the sweep files. 2x4 is skipped because
the dense path needs several GB of memory; the sweep code is the same for all lattices.

Usage:
    python validate.py
"""

import csv
from pathlib import Path

import numpy as np

from common import (
    comm,
    double_occupancy,
    exact_energy_shift,
    gutzwiller_state,
    hubbard_parts,
    neel_state,
    random_state,
    restrict,
    sector_indices,
)

RES = Path(__file__).resolve().parent.parent / "results"


def check(lattice):
    lx, ly = lattice
    tag = f"{lx}x{ly}"
    rows = list(csv.DictReader(open(RES / f"sweep_{tag}.csv")))
    n_sites = lx * ly
    nq = 2 * n_sites
    idx = sector_indices(n_sites, n_sites // 2, n_sites - n_sites // 2)
    d_occ = double_occupancy(idx, n_sites)
    worst = 0.0
    for u in sorted({float(r["u"]) for r in rows}):
        T, V = hubbard_parts([lx, ly], 1.0, u)
        t, v = restrict(T, nq, idx), restrict(V, nq, idx)
        h = t + v
        w, vecs = np.linalg.eigh(h)
        e0, psi0 = w[0], vecs[:, 0]
        c_exact = exact_energy_shift(t, v, e0, psi0, 0.005) / 0.005**2
        psi_free = np.linalg.eigh(t + 1e-3 * v)[1][:, 0]
        psi_gw = gutzwiller_state(psi_free, d_occ, h, np.linspace(0.02, 1.0, 50))[0]
        states = {"exact_gs": psi0, "free_fermion": psi_free, "gutzwiller": psi_gw,
                  "neel": neel_state((lx, ly), idx, nq), "random_s7": random_state(len(idx), 7)}
        a_op, b_op = comm(v, comm(v, t)), comm(t, comm(t, v))
        for r in (r for r in rows if float(r["u"]) == u):
            p = states[r["state"]]
            ref = {"c_exact": c_exact,
                   "A": np.real(p.conj() @ a_op @ p),
                   "B": np.real(p.conj() @ b_op @ p)}
            for k, val in ref.items():
                worst = max(worst, abs(float(r[k]) - val) / max(1.0, abs(val)))
    print(f"{tag}: largest difference from sweep_{tag}.csv = {worst:.1e}")
    return worst


if __name__ == "__main__":
    ok = all(check(lat) < 1e-6 for lat in [(2, 2), (2, 3)])
    print("PASS" if ok else "FAIL")
