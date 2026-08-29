import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from sim import Params, cohesion, displacement


def random_orthogonal(d: int, rng: np.random.Generator) -> NDArray[np.float64]:
    G = rng.normal(0,1,(d,d))
    Q, R = np.linalg.qr(G)
    # Canonicalizes factorization, makes dist uniform
    # sign(0) = 0, occurs with measure 0
    Q = Q @ np.diag(np.sign(np.diag(R))) 
    return Q

def positions(n:int=8, d:int=2):
    return arrays(np.float64, (n, d),
                  elements=st.floats(-10, 10, allow_nan=False))

def shift_vector(d:int=2):
    return arrays(np.float64, (d,),
                  elements=st.floats(-10, 10, allow_nan=False))

def cohesion_from(pos, params):
    disp = displacement(pos, bounds=params.bounds)
    return cohesion(disp, np.linalg.norm(disp, axis=-1), params)

def cohesion_params(radius=2.0, weight=1.0):
    return Params(bounds=None, cohesion_radius=radius, cohesion_weight=weight)

# Random Generation Tests

@given(seed=st.integers(0, 2**32 - 1), d=st.integers(1, 5))
def test_random_orthogonal_is_orthogonal(seed, d):
    Q = random_orthogonal(d, np.random.default_rng(seed))
    assert np.allclose(Q.T @ Q, np.eye(d))
    assert np.isclose(abs(np.linalg.det(Q)), 1.0)


# Displacement Invariance tests

@given(pos=positions())
def test_displacement_antisymmetric(pos):
    disp = displacement(pos, bounds=None)
    assert np.allclose(disp, -disp.transpose(1, 0, 2))

@given(pos=positions())
def test_displacement_zero_diagonal(pos):
    disp = displacement(pos, bounds=None)
    assert np.allclose(np.diagonal(disp, axis1=0, axis2=1), 0.0)

@given(pos=positions(), shift=shift_vector())
def test_displacement_shift_invariant(pos, shift):
    disp = displacement(pos, bounds=None)
    disp_shift = displacement(pos + shift, bounds=None)
    assert np.allclose(disp, disp_shift)

@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_displacement_rotation_equivariant(pos, seed):
    Q = random_orthogonal(2, np.random.default_rng(seed))
    disp = displacement(pos, bounds=None)
    disp_rot = displacement(pos @ Q.T, bounds=None)
    assert np.allclose(disp_rot, disp @ Q.T)


# Displacement Known Value tests

def test_displacement_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    known_disp = np.array([[[ 0.0, 0.0], [ 1.0, 0.0], [ 0.0, 1.0]],
                           [[-1.0,  0.0], [ 0.0,  0.0], [-1.0,  1.0]],
                           [[ 0.0, -1.0], [ 1.0, -1.0], [ 0.0,  0.0]]])
    disp = displacement(pos, bounds=None)
    assert np.allclose(known_disp, disp)


# Cohesion Known Value tests

def test_cohesion_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = Params(cohesion_radius=100, cohesion_weight=10, bounds=None)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 5.0], [5.0, -10.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_partial_neighborhood():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = Params(cohesion_radius=1.2, cohesion_weight=10, bounds=None)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 0.0], [0.0, -10.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_all_isolated():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = Params(cohesion_radius=0.5, cohesion_weight=10, bounds=None)
    known_cohesion =  np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)


# Cohesion Invariance tests

@given(pos=positions(), shift=shift_vector())
def test_cohesion_translation_invariant(pos, shift):
    p = cohesion_params()
    assert np.allclose(cohesion_from(pos + shift, p), cohesion_from(pos, p))


@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_cohesion_rotation_equivariant(pos, seed):
    p = cohesion_params()
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(cohesion_from(pos @ Q.T, p), cohesion_from(pos, p) @ Q.T)


@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_cohesion_permutation_equivariant(pos, seed):
    p = cohesion_params()
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(cohesion_from(pos[perm], p), cohesion_from(pos, p)[perm])


@given(pos=positions(), lam=st.floats(0.1, 10))
def test_cohesion_scaling(pos, lam):
    p, p_scaled = cohesion_params(radius=2.0), cohesion_params(radius=2.0 * lam)
    assert np.allclose(cohesion_from(lam * pos, p_scaled),
                       lam * cohesion_from(pos, p))