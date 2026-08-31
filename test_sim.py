import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from sim import Params, State, alignment, cohesion, displacement, forces


def away_from_radii(pos, radii, eps=1e-6):
    disp = displacement(pos, bounds=None)
    dist = np.linalg.norm(disp, axis=-1)
    return all(np.all(np.abs(dist - r) > eps) for r in radii)

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

def velocities(n: int=8, d:int=2):
    return arrays(np.float64, (n, d),
                  elements=st.floats(-10, 10, allow_nan=False))
def shift_vector(d:int=2):
    return arrays(np.float64, (d,),
                  elements=st.floats(-10, 10, allow_nan=False))

def cohesion_from(pos, params):
    disp = displacement(pos, bounds=params.bounds)
    return cohesion(disp, np.linalg.norm(disp, axis=-1), params)

def cohesion_params(radius=2.0, weight=1.0):
    return Params(bounds=None, cohesion_radius=radius, cohesion_weight=weight, alignment_radius=1.0, alignment_weight=0.0)

def alignment_from(pos, vel, params):
    disp = displacement(pos, bounds=params.bounds)
    return alignment(vel, np.linalg.norm(disp, axis=-1), params)

def alignment_params(radius=2.0, weight=1.0):
    return Params(bounds=None, cohesion_radius=1.0, cohesion_weight=0.0, alignment_radius=radius, alignment_weight=weight)

def force_from(pos, vel, params):
    state = State(pos, vel)
    force = forces(state, params)
    return force


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
    params =cohesion_params(radius=100.0, weight=10.0)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 5.0], [5.0, -10.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_partial_neighborhood():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = cohesion_params(radius=1.2, weight=10)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 0.0], [0.0, -10.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_all_isolated():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = cohesion_params(radius=0.4, weight=10.0)
    known_cohesion =  np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    calc_cohesion = cohesion_from(pos, params)
    assert np.allclose(known_cohesion, calc_cohesion)


# Cohesion Invariance tests

@given(pos=positions(), shift=shift_vector())
def test_cohesion_translation_invariant(pos, shift):
    p = cohesion_params()
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(cohesion_from(pos + shift, p), cohesion_from(pos, p))

@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_cohesion_rotation_equivariant(pos, seed):
    p = cohesion_params()
    assume(away_from_radii(pos, p.radii))
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(cohesion_from(pos @ Q.T, p), cohesion_from(pos, p) @ Q.T)

@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_cohesion_permutation_equivariant(pos, seed):
    p = cohesion_params()
    assume(away_from_radii(pos, p.radii))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(cohesion_from(pos[perm], p), cohesion_from(pos, p)[perm])

@given(pos=positions(), lam=st.floats(0.1, 10))
def test_cohesion_scaling(pos, lam):
    p, p_scaled = cohesion_params(radius=2.0), cohesion_params(radius=2.0 * lam)
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(cohesion_from(lam * pos, p_scaled),
                       lam * cohesion_from(pos, p))


# Alignment Known Values test

def test_alignment_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params =alignment_params(radius=100.0, weight=10.0)
    known_alignment =  np.array([[5.0, 5.0], [-10.0, 5.0], [5.0, -10.0]])
    calc_alignment = alignment_from(pos, vel, params)
    assert np.allclose(known_alignment, calc_alignment)

def test_alignment_partial_neighborhood():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = alignment_params(radius=1.2, weight=10)
    known_alignment =  np.array([[5.0, 5.0], [-10.0, 0.0], [0.0, -10.0]])
    calc_alignment = alignment_from(pos, vel, params)
    assert np.allclose(known_alignment, calc_alignment)

def test_alignment_all_isolated():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = alignment_params(radius=0.4, weight=10.0)
    known_alignment =  np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    calc_alignment = alignment_from(pos, vel, params)
    assert np.allclose(known_alignment, calc_alignment)


# Alignment Invariance tests

@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_alignment_translation_invariant(pos, vel, shift):
    p = alignment_params()
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(alignment_from(pos + shift, vel, p), alignment_from(pos, vel, p))

@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_alignment_galilean_invariant(pos, vel, shift):
    p = alignment_params()
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(alignment_from(pos, vel + shift, p), alignment_from(pos, vel, p))

@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_alignment_rotation_equivariant(pos, vel, seed):
    p = alignment_params()
    assume(away_from_radii(pos, p.radii))
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(alignment_from(pos @ Q.T, vel @ Q.T, p), alignment_from(pos, vel, p) @ Q.T)

@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_alignment_permutation_equivariant(pos, vel, seed):
    p = alignment_params()
    assume(away_from_radii(pos, p.radii))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(alignment_from(pos[perm], vel[perm], p), alignment_from(pos, vel, p)[perm])

@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_alignment_scaling_pos(pos, vel, lam):
    p, p_scaled = alignment_params(radius=2.0), alignment_params(radius=2.0 * lam)
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(alignment_from(lam * pos, vel, p_scaled),
                        alignment_from(pos, vel, p))

@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_alignment_scaling_vel(pos, vel, lam):
    p = alignment_params(radius=2.0)
    assume(away_from_radii(pos, p.radii))
    assert np.allclose(alignment_from(pos, lam * vel, p),
                       lam * alignment_from(pos, vel, p))

# All Forces Known Value tests

def test_combined_forces_known_values():
    pos = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    vel = np.array([[-1.0, 3.0], [3.0, 5.0], [-2.0, 1.0]])
    params = Params(bounds=None, cohesion_radius=5, cohesion_weight=10, alignment_radius=20, alignment_weight=40)
    known_alignment =  np.array([[60.0, 0.0], [-180.0, -120.0], [120.0, 120.0]])
    known_cohesion = np.array([[20.0, 20.0], [0.0, 0.0], [-20.0, -20.0]])
    calc_force = force_from(pos, vel, params)
    assert np.allclose(known_alignment + known_cohesion, calc_force)