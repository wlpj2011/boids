from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from sim import Params, State, alignment, cohesion, displacement, forces, separation


def well_separated(pos, radii, eps=1e-6, min_sep=1e-3):
    disp = displacement(pos, bounds=None)
    dist = np.linalg.norm(disp, axis=-1)
    off_radii = all(np.all(np.abs(dist - r) > eps) for r in radii)
    not_nearly_coincident = np.all((dist == 0) | (dist > min_sep))
    return off_radii and not_nearly_coincident

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

def cohesion_from(pos, vel, params):
    disp = displacement(pos, bounds=params.bounds)
    return cohesion(disp, np.linalg.norm(disp, axis=-1), params)

def cohesion_params(radius=2.0, weight=1.0):
    return Params(bounds=None, eps_smooth=0.01,
                  cohesion_radius=radius, cohesion_weight=weight, 
                  alignment_radius=1.0, alignment_weight=0.0, 
                  separation_radius=1.0, separation_weight=0.0,
                  min_speed=1.0, max_speed=100.0)

def alignment_from(pos, vel, params):
    disp = displacement(pos, bounds=params.bounds)
    return alignment(vel, np.linalg.norm(disp, axis=-1), params)

def alignment_params(radius=2.0, weight=1.0):
    return Params(bounds=None, eps_smooth=0.01, 
                  cohesion_radius=1.0, cohesion_weight=0.0, 
                  alignment_radius=radius, alignment_weight=weight, 
                  separation_radius=1.0, separation_weight=0.0,
                  min_speed=1.0, max_speed=100.0)

def separation_from(pos, vel, params):
    disp = displacement(pos, bounds=params.bounds)
    return separation(disp, np.linalg.norm(disp, axis=-1), params)

def separation_params(radius=2.0, weight=1.0, eps = 0.01):
    return Params(bounds=None, eps_smooth=eps, 
                  cohesion_radius=1.0, cohesion_weight=0.0, 
                  alignment_radius=1.0, alignment_weight=0.0, 
                  separation_radius=radius, separation_weight=weight,
                  min_speed=1.0, max_speed=100.0)


def full_params(radius=2.0, weight=1.0, eps = 0.01):
    return Params(bounds=None, eps_smooth=eps, 
                  cohesion_radius=radius, cohesion_weight=weight, 
                  alignment_radius=radius, alignment_weight=weight, 
                  separation_radius=radius, separation_weight=weight,
                  min_speed=1.0, max_speed=100.0)

def forces_from(pos, vel, params):
    state = State(pos, vel)
    force = sum(forces(state, params).values())
    return force

def separation_reference(pos, params):
    n, d = pos.shape
    out = np.zeros((n, d))
    for i in range(n):
        acc, count = np.zeros(d), 0
        for j in range(n):
            if i == j: continue
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r)
            if 0 < dist < params.separation_radius:
                acc -= r / (dist**2 + params.eps_smooth **2)
                count += 1
        out[i] = acc #/ count if count else 0
    return out * params.separation_weight

# Random Generation Tests

@given(seed=st.integers(0, 2**32 - 1), d=st.integers(1, 5))
def test_random_orthogonal_is_orthogonal(seed, d):
    Q = random_orthogonal(d, np.random.default_rng(seed))
    assert np.allclose(Q.T @ Q, np.eye(d))
    assert np.isclose(abs(np.linalg.det(Q)), 1.0)


FORCES = [
    pytest.param(cohesion_from, cohesion_params(), 1, id="cohesion"),
    pytest.param(alignment_from, alignment_params(), 0, id="alignment"),
    pytest.param(separation_from, separation_params(), -1, id="separation"),
]

SUBJECTS = FORCES + [pytest.param(forces_from, full_params(), None, id="all")]

LENGTH_FIELDS = ("cohesion_radius", "alignment_radius",
                 "separation_radius", "eps_smooth")

VELOCITY_FIELDS = ("min_speed", "max_speed",)

def scale_lengths(p: Params, lam: float) -> Params:
    return replace(p, **{f: getattr(p, f) * lam for f in LENGTH_FIELDS})

@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_rotation_equivariant(force_from, p, exponent, pos, vel, seed):
    assume(well_separated(pos, p.radii))
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(force_from(pos @ Q.T, vel @ Q.T, p),
                       force_from(pos, vel, p) @ Q.T)

@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_translation_invariant(force_from, p, exponent, shift, pos, vel):
    assume(well_separated(pos, p.radii))
    assert np.allclose(force_from(pos + shift, vel, p), force_from(pos, vel, p))

@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_galilean_invariant(force_from, p, exponent, shift, pos, vel):
    assume(well_separated(pos, p.radii))
    assert np.allclose(force_from(pos, vel + shift, p), force_from(pos, vel, p))

@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_permutation_equivariant(force_from, p, exponent, pos, vel, seed):
    assume(well_separated(pos, p.radii))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(force_from(pos[perm], vel[perm], p), force_from(pos, vel, p)[perm])

@pytest.mark.parametrize("force_from, p, exponent", FORCES)
@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_position_scaling(force_from, p, exponent, pos, vel, lam):
    assume(well_separated(pos, p.radii))
    assert np.allclose(force_from(lam * pos, vel, scale_lengths(p, lam)),
                       lam ** exponent * force_from(pos, vel, p))

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
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params =cohesion_params(radius=100.0, weight=10.0)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 5.0], [5.0, -10.0]])
    calc_cohesion = cohesion_from(pos, vel, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_partial_neighborhood():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params = cohesion_params(radius=1.2, weight=10)
    known_cohesion =  np.array([[5.0, 5.0], [-10.0, 0.0], [0.0, -10.0]])
    calc_cohesion = cohesion_from(pos, vel, params)
    assert np.allclose(known_cohesion, calc_cohesion)

def test_cohesion_all_isolated():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params = cohesion_params(radius=0.4, weight=10.0)
    known_cohesion =  np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    calc_cohesion = cohesion_from(pos, vel, params)
    assert np.allclose(known_cohesion, calc_cohesion)


# Alignment Known Values tests

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


# Alignment Equivariance tests

@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_alignment_scaling_vel(pos, vel, lam):
    p = alignment_params(radius=2.0)
    assume(well_separated(pos, p.radii))
    assert np.allclose(alignment_from(pos, lam * vel, p),
                       lam * alignment_from(pos, vel, p))


# Separation Match Loop test

@given(pos=positions(), vel=velocities())
def test_separation_numpy_loop(pos, vel):
    p = separation_params()
    assume(well_separated(pos, p.radii))
    assert np.allclose(separation_from(pos, vel, p), separation_reference(pos, p))


# Separation Known Values tests

def test_separation_known_values():
    pos = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    vel = np.array([[-1.0, 3.0], [3.0, 5.0], [-2.0, 1.0]])
    params = separation_params(radius=4.0, weight=2.0, eps=0.00)
    known_separation = np.array([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]])
    calc_separation = separation_from(pos,vel, params)
    assert np.allclose(known_separation, calc_separation)

def test_separation_partial_neighborhood():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params = separation_params(radius=1.2, weight=10, eps=0.00)
    known_separation =  np.array([[-10.0, -10.0], [10.0, 0.0], [0.0, 10.0]])
    calc_separation = separation_from(pos, vel, params)
    assert np.allclose(known_separation, calc_separation)

def test_separation_all_isolated():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = separation_params(radius=0.4, weight=10, eps=0.00)
    known_separation =  np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    calc_separation = separation_from(pos, vel, params)
    assert np.allclose(known_separation, calc_separation)

def test_separation_coincidence():
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params = separation_params(radius=4.0, weight=2, eps=0.00)
    known_separation =  np.array([[0.0, -2.0], [0.0, -2.0], [0.0, 4.0]])
    calc_separation = separation_from(pos, vel, params)
    assert np.allclose(known_separation, calc_separation)

# All Forces Known Value tests

def test_combined_forces_known_values():
    pos = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    vel = np.array([[-1.0, 3.0], [3.0, 5.0], [-2.0, 1.0]])
    params = Params(bounds=None, eps_smooth=0.01, 
                    cohesion_radius=5, cohesion_weight=10, 
                    alignment_radius=20, alignment_weight=40,
                    separation_radius=4.0, separation_weight=2.0,
                    min_speed=1.0, max_speed=100.0)
    known_alignment =  np.array([[60.0, 0.0], [-180.0, -120.0], [120.0, 120.0]])
    known_cohesion = np.array([[20.0, 20.0], [0.0, 0.0], [-20.0, -20.0]])
    known_separation = np.array([[-0.49999375, -0.49999375], [0.0, 0.0], [0.49999375, 0.49999375]])
    calc_force = forces_from(pos, vel, params)
    assert np.allclose(known_alignment + known_cohesion + known_separation, calc_force)