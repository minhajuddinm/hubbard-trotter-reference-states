"""Shared helpers: Hamiltonian, symmetry sector, reference states, error operator, PennyLane estimate.

The Fermi-Hubbard model is built with qml.spin.fermi_hubbard (Jordan-Wigner, interleaved wire order)
and split into the hopping part T and the on-site part V. Everything is restricted to the half-filled
Sz = 0 sector. The second-order step is U2(dt) = exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2).
"""

import warnings
from itertools import combinations

import numpy as np
import scipy.linalg as sla
from scipy.sparse import csr_array

warnings.filterwarnings("ignore")

import pennylane as qml  # noqa: E402
import pennylane.estimator as qre  # noqa: E402
from pennylane.labs.trotter_error import (  # noqa: E402
    ProductFormula,
    perturbation_error,
    sparse_fragments,
)
from pennylane.labs.trotter_error.fragments.sparse_fragments import SparseState  # noqa: E402



# ---------------------------------------------------------------------------
# Hamiltonian and symmetry sector
# ---------------------------------------------------------------------------
def hubbard_parts(lattice, t, u, periodic=False):
    """Return the hopping part T and interaction part V as PennyLane operators."""
    kw = dict(mapping="jordan_wigner", boundary_condition=periodic)
    T = qml.spin.fermi_hubbard("square", lattice, hopping=t, coulomb=0.0, **kw)
    V = qml.spin.fermi_hubbard("square", lattice, hopping=0.0, coulomb=u, **kw)
    return T, V


def sector_indices(n_sites, n_up, n_down):
    """Basis indices with n_up spin-up and n_down spin-down electrons.

    PennyLane's fermi_hubbard uses interleaved ordering: wire 2i is site i spin-up,
    wire 2i+1 is site i spin-down. Wire 0 is the most significant bit of the index.
    An occupied mode is |1>.
    """
    n_qubits = 2 * n_sites
    idx = []
    for ups in combinations(range(n_sites), n_up):
        for downs in combinations(range(n_sites), n_down):
            k = 0
            for s in ups:
                k |= 1 << (n_qubits - 1 - 2 * s)
            for s in downs:
                k |= 1 << (n_qubits - 1 - (2 * s + 1))
            idx.append(k)
    return np.array(sorted(idx))


def restrict(op, n_qubits, idx):
    """Dense matrix of op restricted to the basis states in idx."""
    m = op.sparse_matrix(wire_order=range(n_qubits)).tocsr()
    return m[idx][:, idx].toarray()


def check_block_diagonal(op, n_qubits, idx):
    """Max matrix element leaking out of the sector. Should be ~0."""
    m = op.sparse_matrix(wire_order=range(n_qubits)).tocsr()
    mask = np.ones(m.shape[0], dtype=bool)
    mask[idx] = False
    return np.abs(m[idx][:, mask]).max() if mask.any() else 0.0


# ---------------------------------------------------------------------------
# Proxy states (all inside the sector)
# ---------------------------------------------------------------------------
def neel_state(lattice, idx, n_qubits):
    """Antiferromagnetic product state: up on even-parity sites, down on odd-parity sites."""
    lx, ly = lattice
    k = 0
    for x in range(lx):
        for y in range(ly):
            s = x * ly + y
            wire = 2 * s if (x + y) % 2 == 0 else 2 * s + 1
            k |= 1 << (n_qubits - 1 - wire)
    psi = np.zeros(len(idx), dtype=complex)
    pos = np.searchsorted(idx, k)
    if pos >= len(idx) or idx[pos] != k:
        raise ValueError("Neel state is not in the chosen sector")
    psi[pos] = 1.0
    return psi


def ground_state(h):
    w, v = np.linalg.eigh(h)
    return w[0], v[:, 0], w[1] - w[0]


def random_state(dim, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Trotter error: exact, bound, perturbative
# ---------------------------------------------------------------------------
def comm(a, b):
    return a @ b - b @ a


def eps_coefficient_operator(t_mat, v_mat):
    """Leading error operator E with H_eff = H + dt^2 E for U2 = e^{-iV/2} e^{-iT} e^{-iV/2}.

    From the symmetric BCH formula,
    log(e^{A/2} e^{B} e^{A/2}) = A + B + [B,[B,A]]/12 - [A,[A,B]]/24 + O(5),
    with A = -i V dt and B = -i T dt this gives
    E = -[T,[T,V]]/12 + [V,[V,T]]/24.
    """
    return -comm(t_mat, comm(t_mat, v_mat)) / 12 + comm(v_mat, comm(v_mat, t_mat)) / 24


def exact_energy_shift(t_mat, v_mat, e0, psi0, dt):
    """Energy of the Trotter eigenstate that follows the ground state, minus e0."""
    half_v = sla.expm(-0.5j * dt * v_mat)
    u2 = half_v @ sla.expm(-1j * dt * t_mat) @ half_v
    w, vecs = np.linalg.eig(u2)
    k = np.argmax(np.abs(vecs.conj().T @ psi0))
    phase = -np.angle(w[k])
    # unwrap so the effective energy sits next to e0
    delta = (phase - e0 * dt + np.pi) % (2 * np.pi) - np.pi
    return delta / dt


def pennylane_pt(t_mat, v_mat, states, dt):
    """<psi|eps|psi> from pennylane.labs.trotter_error for each state.

    Fragment 0 = V, fragment 1 = T. ProductFormula follows U = prod e^{i t a_k H_k};
    the sign and factor convention is calibrated against the explicit formula in main().
    """
    frags = dict(enumerate(sparse_fragments([csr_array(v_mat), csr_array(t_mat)])))
    pf = ProductFormula([0, 1, 0], coeffs=[0.5, 1.0, 0.5])
    sstates = [SparseState(csr_array(p.reshape(1, -1))) for p in states]
    out = perturbation_error(pf, frags, sstates, max_order=3, timestep=dt)
    return [sum(d.values()) for d in out]


# ---------------------------------------------------------------------------
# Gate cost from qml.estimator
# ---------------------------------------------------------------------------
def pauli_groups(op):
    """Count Pauli-string shapes (e.g. 'XZX') in op, skipping the identity."""
    counts = {}
    for pw in op.pauli_rep:
        if len(pw) == 0:
            continue
        key = "".join(pw[w] for w in sorted(pw.keys()))
        counts[key] = counts.get(key, 0) + 1
    return counts


def t_count_per_step(T, V, n_qubits):
    ham = qre.PauliHamiltonian(num_qubits=n_qubits, pauli_terms=[pauli_groups(V), pauli_groups(T)])
    res = qre.estimate(qre.TrotterPauli(ham, num_steps=1, order=2))
    return res.gate_counts.get("T", float("nan")) if hasattr(res, "gate_counts") else res


# ---------------------------------------------------------------------------
# Gutzwiller reference state
# ---------------------------------------------------------------------------
def double_occupancy(idx, n_sites):
    """Number of doubly occupied sites for each basis index (interleaved wire order)."""
    n_qubits = 2 * n_sites
    d = np.zeros(len(idx), dtype=int)
    for s in range(n_sites):
        up = (idx >> (n_qubits - 1 - 2 * s)) & 1
        dn = (idx >> (n_qubits - 1 - (2 * s + 1))) & 1
        d += up & dn
    return d


def gutzwiller_state(psi_free, d_occ, h_mat, grid):
    """Variational Gutzwiller projection g^D |free>, with g minimising <H> on the grid."""
    best = None
    for g in grid:
        p = psi_free * g ** d_occ
        p = p / np.linalg.norm(p)
        e = np.real(p.conj() @ h_mat @ p)
        if best is None or e < best[0]:
            best = (e, g, p)
    return best[2], best[1], best[0]


# ---------------------------------------------------------------------------
# Unrestricted Hartree-Fock reference state
# ---------------------------------------------------------------------------
def one_body_hopping(T, n_sites):
    """Single-particle hopping matrix h1[s, s'] read from the PennyLane operator T.

    In the one-electron sector the Jordan-Wigner strings act on empty wires, so the
    matrix element between spin-up modes 2s and 2s' equals h1[s, s'].
    """
    nq = 2 * n_sites
    m = T.sparse_matrix(wire_order=range(nq)).tocsr()
    k = [1 << (nq - 1 - 2 * s) for s in range(n_sites)]
    return np.real(m[k][:, k].toarray())


def uhf_orbitals(h1, u, n_up, n_down, bias=0.3, tol=1e-11, max_iter=5000, mix=0.5):
    """Collinear unrestricted Hartree-Fock for the Hubbard model.

    Starts from an antiferromagnetic density guess (and from a uniform one) and returns the
    solution with the lower mean-field energy: (occupied up orbitals, occupied down orbitals,
    energy, converged).
    """
    n = h1.shape[0]
    sign = np.array([(-1) ** s for s in range(n)], dtype=float)
    best = None
    for b in (bias, 1e-3):
        n_a = n_up / n + b * sign
        n_b = n_down / n - b * sign
        conv = False
        for _ in range(max_iter):
            ea, ca = np.linalg.eigh(h1 + u * np.diag(n_b))
            eb, cb = np.linalg.eigh(h1 + u * np.diag(n_a))
            new_a = np.sum(ca[:, :n_up] ** 2, axis=1)
            new_b = np.sum(cb[:, :n_down] ** 2, axis=1)
            diff = max(np.abs(new_a - n_a).max(), np.abs(new_b - n_b).max())
            n_a = mix * new_a + (1 - mix) * n_a
            n_b = mix * new_b + (1 - mix) * n_b
            if diff < tol:
                conv = True
                break
        energy = ea[:n_up].sum() + eb[:n_down].sum() - u * np.dot(n_a, n_b)
        if best is None or energy < best[2] - 1e-12:
            best = (ca[:, :n_up], cb[:, :n_down], energy, conv)
    return best


def slater_state(orb_up, orb_down, idx, n_sites):
    """Amplitudes of the determinant prod_a c+_{phi_a} |0> on the sector basis idx.

    Modes are ordered by wire (2s up, 2s+1 down), matching PennyLane's Jordan-Wigner
    convention, so the amplitude of an occupation pattern is det(M[occupied wires, :]).
    """
    nq = 2 * n_sites
    m = np.zeros((nq, orb_up.shape[1] + orb_down.shape[1]))
    m[0::2, :orb_up.shape[1]] = orb_up
    m[1::2, orb_up.shape[1]:] = orb_down
    psi = np.empty(len(idx))
    for i, k in enumerate(idx):
        occ = [w for w in range(nq) if (k >> (nq - 1 - w)) & 1]
        psi[i] = np.linalg.det(m[occ, :])
    return psi.astype(complex)
