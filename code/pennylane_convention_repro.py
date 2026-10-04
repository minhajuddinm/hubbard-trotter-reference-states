"""Minimal reproduction of the return convention of pennylane.labs.trotter_error.perturbation_error.

For the second-order formula U(t) = exp(itA/2) exp(itB) exp(itA/2), the symmetric BCH expansion gives
log U(t) = it(A + B) + it^3 E + O(t^5) with E = -[B,[B,A]]/12 + [A,[A,B]]/24, so the effective
Hamiltonian is H_eff = A + B + t^2 E and H_eff - H = t^2 E.

The docstring says perturbation_error returns <psi| H_eff - H |psi> = t^2 <E>. This script compares the
returned value with t^2 <E> computed explicitly, for random Hermitian A, B and a random state.

Usage:
    python pennylane_convention_repro.py
"""

import warnings

import numpy as np
from scipy.sparse import csr_array

warnings.filterwarnings("ignore")

import pennylane as qml  # noqa: E402
from pennylane.labs.trotter_error import ProductFormula, perturbation_error, sparse_fragments  # noqa: E402
from pennylane.labs.trotter_error.fragments.sparse_fragments import SparseState  # noqa: E402


def comm(x, y):
    return x @ y - y @ x


rng = np.random.default_rng(1)
dim = 6
a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
b = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
A, B = (a + a.conj().T) / 2, (b + b.conj().T) / 2
psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
psi /= np.linalg.norm(psi)

E = -comm(B, comm(B, A)) / 12 + comm(A, comm(A, B)) / 24
frags = dict(enumerate(sparse_fragments([csr_array(A), csr_array(B)])))
pf = ProductFormula([0, 1, 0], coeffs=[0.5, 1.0, 0.5])
state = SparseState(csr_array(psi.reshape(1, -1)))

print(f"PennyLane {qml.__version__}")
for t in (0.1, 0.01):
    returned = sum(perturbation_error(pf, frags, [state], max_order=3, timestep=t)[0].values())
    documented = t**2 * np.vdot(psi, E @ psi)  # <psi| H_eff - H |psi>
    print(f"t = {t}: returned = {complex(returned):.6e}, t^2<E> = {complex(documented):.6e}, "
          f"returned / (i t * t^2<E>) = {complex(returned / (1j * t * documented)):.6f}")
