"""Sweep for the paper: 2x2, 2x3 and 2x4 Fermi-Hubbard lattices with open boundaries.

Memory use is kept low so that 2x4 (sector dimension 4900) runs on a laptop (peak about 2 GB):

  * T, the nested commutators [T,[T,V]] and [V,[V,T]], E and H are kept sparse; V is a
    diagonal vector. Expectation values, spectral norms (eigsh) and ground states (eigsh)
    never form a dense matrix.
  * The exact Trotter shift is the only dense step: T is diagonalised once (real, 4900^2),
    and for each dt one complex U2(dt) is built, LU-factorised in place, and freed before
    the next dt. Inverse iteration near exp(-i E0 dt) gives the eigenphase that follows the
    ground state.
  * Results are written to the CSV after every U/t value, so a stopped run keeps its rows.

Output columns include norm_E (the leading-order worst case ||E||), c_half
(-(A+B)/48), residual r = B - A, half_width |r|/48, in_interval (c_exact within c_half +- |r|/48).

Usage (run with python -u so the log is not buffered):
    python -u run_sweep.py --lattice 2 2
    python -u run_sweep.py --lattice 2 3
    python -u run_sweep.py --lattice 2 4
"""

import argparse
import csv
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse import csr_array, diags
from scipy.sparse.linalg import eigsh

from common import (
    double_occupancy,
    gutzwiller_state,
    check_block_diagonal,
    hubbard_parts,
    neel_state,
    pennylane_pt,
    random_state,
    sector_indices,
    t_count_per_step,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def rss_mb():
    """Resident memory of this process in MB (psutil if available)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def restrict_sparse(op, n_qubits, idx):
    m = op.sparse_matrix(wire_order=range(n_qubits)).tocsr()
    return csr_array(m[idx][:, idx])


def lowest(h_sparse, k=2):
    w, v = eigsh(h_sparse, k=k, which="SA")
    order = np.argsort(w)
    return w[order], v[:, order]


def spectral_norm(herm_sparse):
    return float(np.abs(eigsh(herm_sparse, k=1, which="LM", return_eigenvectors=False))[0])


def expval(op, psi):
    return float(np.real(np.vdot(psi, op @ psi)))


class LowMemTrotter:
    """Exact Trotter eigenphase with one dense complex matrix alive at a time."""

    def __init__(self, t_sparse):
        lam, w = np.linalg.eigh(t_sparse.toarray())
        self.lam, self.w = lam, w

    def energy_shift(self, v_diag, e0, psi0, dt, tol=1e-15, max_iter=20):
        half_v = np.exp(-0.5j * dt * v_diag)
        u2 = (self.w * np.exp(-1j * dt * self.lam)) @ self.w.T  # complex dense
        u2 *= half_v[:, None]
        u2 *= half_v[None, :]
        sigma = np.exp(-1j * e0 * dt)
        x = psi0.astype(complex)
        u2_x = lambda y: u2 @ y  # noqa: E731
        m = u2.copy()
        m[np.diag_indices_from(m)] -= sigma
        lu = sla.lu_factor(m, overwrite_a=True, check_finite=False)
        del m
        lam_old = None
        for _ in range(max_iter):
            x = sla.lu_solve(lu, x, check_finite=False)
            x /= np.linalg.norm(x)
            lam_val = np.vdot(x, u2_x(x))
            if lam_old is not None and abs(lam_val - lam_old) < tol:
                break
            lam_old = lam_val
        overlap = abs(np.vdot(psi0, x)) ** 2
        del lu, u2
        gc.collect()
        return -np.angle(lam_val * np.exp(1j * e0 * dt)) / dt, overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", type=int, nargs=2, default=[2, 4])
    ap.add_argument("--u", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--dt", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1])
    args = ap.parse_args()

    lattice = tuple(args.lattice)
    n_sites = lattice[0] * lattice[1]
    n_qubits = 2 * n_sites
    idx = sector_indices(n_sites, n_sites // 2, n_sites - n_sites // 2)
    d_occ = double_occupancy(idx, n_sites)
    tag = f"{lattice[0]}x{lattice[1]}"
    out = RESULTS / f"sweep_{tag}.csv"
    print(f"Lattice {tag} (open): {n_qubits} qubits, sector dim {len(idx)}")
    print(f"Host: {platform.processor()} | Python {platform.python_version()}")

    t_sparse, trot, t_diag_T, writer, fh = None, None, None, None, None
    for u in args.u:
        t_u0 = time.perf_counter()
        T, V = hubbard_parts(list(lattice), 1.0, u, periodic=False)
        leak = max(check_block_diagonal(T, n_qubits, idx), check_block_diagonal(V, n_qubits, idx))
        if leak > 1e-10:
            raise RuntimeError(f"Sector is not invariant (leak {leak:.2e})")
        if t_sparse is None:  # T does not depend on U
            t_sparse = restrict_sparse(T, n_qubits, idx)
            t0 = time.perf_counter()
            trot = LowMemTrotter(t_sparse)
            t_diag_T = time.perf_counter() - t0
            print(f"  diagonalised T in {t_diag_T:.1f} s, RSS {rss_mb():.0f} MB")
        v_sp = restrict_sparse(V, n_qubits, idx)
        v_diag = np.real(v_sp.diagonal())
        if abs(v_sp - diags(v_diag)).max() > 1e-12:
            raise RuntimeError("V is not diagonal in the occupation basis")
        v_op = csr_array(diags(v_diag))
        h_sp = csr_array(t_sparse + v_op)

        t0 = time.perf_counter()
        w, vecs = lowest(h_sp)
        e0, psi0, gap = w[0], vecs[:, 0], w[1] - w[0]
        t_gs = time.perf_counter() - t0

        psi_free = lowest(csr_array(t_sparse + 1e-3 * v_op), k=1)[1][:, 0]
        psi_gw, g_opt, _ = gutzwiller_state(psi_free, d_occ, h_sp, np.linspace(0.02, 1.0, 50))
        states = {"exact_gs": psi0, "free_fermion": psi_free, "gutzwiller": psi_gw,
                  "neel": neel_state(lattice, idx, n_qubits), "random_s7": random_state(len(idx), 7)}

        tv = t_sparse @ v_op - v_op @ t_sparse            # [T,V]
        b_op = csr_array(t_sparse @ tv - tv @ t_sparse)   # [T,[T,V]]
        a_op = csr_array(-(v_op @ tv - tv @ v_op))        # [V,[V,T]] = -[V,[T,V]]
        e_op = csr_array(-b_op / 12 + a_op / 24)
        c_bound = spectral_norm(b_op) / 12 + spectral_norm(a_op) / 24
        norm_e = spectral_norm(e_op)

        t0 = time.perf_counter()
        shifts, track = [], []
        for dt in args.dt:
            s, ov = trot.energy_shift(v_diag, e0, psi0, dt)
            shifts.append(s)
            track.append(ov)
        t_exact = time.perf_counter() - t0 + t_gs
        c_exact = shifts[0] / args.dt[0] ** 2

        t0 = time.perf_counter()
        raw = pennylane_pt(t_sparse, v_op, list(states.values()), args.dt[0])
        t_pl = time.perf_counter() - t0

        rows = []
        for k, p in states.items():
            a, b = expval(a_op, p), expval(b_op, p)
            std = -b / 12 + a / 24
            rows.append(dict(
                lattice=tag, u=u, state=k, e0=e0, gap=gap, g_opt=g_opt if k == "gutzwiller" else "",
                overlap_gs=abs(np.vdot(psi0, p)) ** 2, energy=expval(h_sp, p),
                c_exact=c_exact, c_bound=c_bound, norm_E=norm_e, A=a, B=b,
                c_standard=std, c_formA=-a / 24, c_formB=-b / 24, c_half=-(a + b) / 48,
                residual=b - a, half_width=abs(b - a) / 48,
                in_interval=abs(c_exact - (-(a + b) / 48)) <= abs(b - a) / 48 + 1e-6 * abs(c_exact),
                **{f"dE_over_dt2_{dt}": s / dt**2 for dt, s in zip(args.dt, shifts)},
                min_track_overlap=min(track), seconds_exact=t_exact,
                seconds_pennylane=t_pl, seconds_diag_T=t_diag_T))
        scale = rows[0]["c_standard"] * args.dt[0] ** 2 / raw[0]
        for r, rw in zip(rows, raw):
            r["c_pennylane"] = float(np.real(rw * scale)) / args.dt[0] ** 2
        pl_ok = all(abs(r["c_pennylane"] - r["c_standard"]) <= 1e-6 * max(1.0, abs(r["c_standard"]))
                    for r in rows)

        if writer is None:
            fh = open(out, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()

        print(f"\nU/t = {u}: E0 = {e0:.6f}, gap = {gap:.4f}, g* = {g_opt:.2f}")
        print(f"  c_exact = {c_exact:+.5e}; bound step ratio {np.sqrt(c_bound / abs(c_exact)):.2f}, "
              f"||E|| step ratio {np.sqrt(norm_e / abs(c_exact)):.2f}")
        print("  dt: " + "  ".join(f"{dt}:{s / dt**2:+.4e}" for dt, s in zip(args.dt, shifts)))
        print(f"  {'state':<13} {'overlap':>7} {'standard':>9} {'-A/24':>9} {'-B/24':>9} {'c_half':>9} {'|r|/48':>8} in")
        for r in rows:
            print(f"  {r['state']:<13} {r['overlap_gs']:>7.4f} {r['c_standard']:>+9.4f} {r['c_formA']:>+9.4f} "
                  f"{r['c_formB']:>+9.4f} {r['c_half']:>+9.4f} {r['half_width']:>8.4f} {int(r['in_interval'])}")
        print(f"  exact {t_exact:.1f} s, PennyLane {t_pl:.2f} s (consistent: {pl_ok}), "
              f"U step {time.perf_counter() - t_u0:.1f} s, RSS {rss_mb():.0f} MB")


    fh.close()
    try:
        print(f"\nqml.estimator T gates per step: {t_count_per_step(T, V, n_qubits)}")
    except Exception as exc:
        print(f"\nqml.estimator call failed: {exc!r}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
