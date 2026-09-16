from dataclasses import replace

import numpy as np
import pytest
from helpers import (
    FORCES,
    LENGTH_FIELDS,
    SUBJECTS,
    TOPOLOGIES,
    alignment_from,
    alignment_params,
    bounded_velocites,
    cohesion_from,
    cohesion_params,
    forces_from,
    full_params,
    positions,
    random_orthogonal,
    scale_lengths,
    separation_from,
    separation_params,
    separation_reference,
    shift_vector,
    step_from,
    torus_params,
    unclamped_params,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sim import (
    SPEED_FLOOR,
    Flat,
    Params,
    State,
    Torus,
    displacement,
    step,
)

# Random Generation Tests

@given(seed=st.integers(0, 2**32 - 1), d=st.integers(1, 5))
def test_random_orthogonal_is_orthogonal(seed, d):
    Q = random_orthogonal(d, np.random.default_rng(seed))
    assert np.allclose(Q.T @ Q, np.eye(d))
    assert np.isclose(abs(np.linalg.det(Q)), 1.0)


# Equivariance and Invariance Tests

@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_rotation_equivariant(force_from, p, exponent, pos, vel, seed):
    assume(well_separated(pos, p))
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(force_from(pos @ Q.T, vel @ Q.T, p),
                       force_from(pos, vel, p) @ Q.T)
    
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_translation_invariant(topology, force_from, p, exponent, shift, pos, vel):
    p = replace(p, topology=topology)
    assume(well_separated(pos, p))
    assert np.allclose(force_from(pos + shift, vel, p), force_from(pos, vel, p))

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_galilean_invariant(topology, force_from, p, exponent, shift, pos, vel):
    p = replace(p, topology=topology)
    assume(well_separated(pos, p))
    assert np.allclose(force_from(pos, vel + shift, p), force_from(pos, vel, p))

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_permutation_equivariant(topology, force_from, p, exponent, pos, vel, seed):
    p = replace(p, topology=topology)
    assume(well_separated(pos, p))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(force_from(pos[perm], vel[perm], p), force_from(pos, vel, p)[perm])

@pytest.mark.parametrize("topology", TOPOLOGIES)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_step_permutation_equivariant(topology, pos, vel, seed):
    p = full_params()
    p = replace(p, topology=topology)
    assume(well_separated(pos, p))
    perm = np.random.default_rng(seed).permutation(len(pos))
    next_step = step_from(pos, vel, 0.01, p)
    next_step_permuted = step_from(pos[perm], vel[perm], 0.01, p)
    assert np.allclose((next_step[0][perm], next_step[1][perm]), next_step_permuted)

@pytest.mark.parametrize("force_from, p, exponent", FORCES)
@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_position_scaling(force_from, p, exponent, pos, vel, lam):
    assume(well_separated(pos, p))
    assert np.allclose(force_from(lam * pos, vel, scale_lengths(p, lam)),
                       lam ** exponent * force_from(pos, vel, p))

# Displacement Invariance tests
@pytest.mark.parametrize("topology", TOPOLOGIES)
@given(pos=positions())
def test_displacement_antisymmetric(topology, pos):
    disp = displacement(pos, topology=topology)
    assert np.allclose(disp, -disp.transpose(1, 0, 2))

@pytest.mark.parametrize("topology", TOPOLOGIES)
@given(pos=positions())
def test_displacement_zero_diagonal(topology, pos):
    disp = displacement(pos, topology=topology)
    assert np.allclose(np.diagonal(disp, axis1=0, axis2=1), 0.0)

@pytest.mark.parametrize("topology", TOPOLOGIES)
@given(pos=positions(), shift=shift_vector())
def test_displacement_shift_invariant(topology, pos, shift):
    assume(well_separated(pos, torus_params(20.0)))
    disp = displacement(pos, topology=topology)
    disp_shift = displacement(pos + shift, topology=topology)
    assert np.allclose(disp, disp_shift)

@given(pos=positions(), seed=st.integers(0, 2**32 - 1))
def test_displacement_rotation_equivariant(pos, seed):
    Q = random_orthogonal(2, np.random.default_rng(seed))
    disp = displacement(pos, topology=Flat())
    disp_rot = displacement(pos @ Q.T, topology=Flat())
    assert np.allclose(disp_rot, disp @ Q.T)


# Displacement Known Value tests

def test_displacement_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    known_disp = np.array([[[ 0.0, 0.0], [ 1.0, 0.0], [ 0.0, 1.0]],
                           [[-1.0,  0.0], [ 0.0,  0.0], [-1.0,  1.0]],
                           [[ 0.0, -1.0], [ 1.0, -1.0], [ 0.0,  0.0]]])
    disp = displacement(pos, topology=Flat())
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
    assume(well_separated(pos, p))
    assert np.allclose(alignment_from(pos, lam * vel, p),
                       lam * alignment_from(pos, vel, p))


# Separation Match Loop test

@given(pos=positions(), vel=velocities())
def test_separation_numpy_loop(pos, vel):
    p = separation_params()
    assume(well_separated(pos, p))
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
    params = Params(topology=Flat(), eps_smooth=0.01, 
                    cohesion_radius=5, cohesion_weight=10, 
                    alignment_radius=20, alignment_weight=40,
                    separation_radius=4.0, separation_weight=2.0,
                    min_speed=1.0, max_speed=100.0)
    known_alignment =  np.array([[60.0, 0.0], [-180.0, -120.0], [120.0, 120.0]])
    known_cohesion = np.array([[20.0, 20.0], [0.0, 0.0], [-20.0, -20.0]])
    known_separation = np.array([[-0.49999375, -0.49999375], [0.0, 0.0], [0.49999375, 0.49999375]])
    calc_force = forces_from(pos, vel, params)
    assert np.allclose(known_alignment + known_cohesion + known_separation, calc_force)


# Step Tests
@given(pos=positions(), vel=velocities())
def test_step_speed_bounds(pos, vel):
    state = State(pos, vel)
    params = full_params()
    new_state = step(state, params, 0.01)
    speed = np.linalg.norm(new_state.vel, axis=1)
    tol = 1e-9
    speed_bounded = (params.min_speed - tol <= speed) & (speed <= params.max_speed + tol)
    stationary = speed <= SPEED_FLOOR
    assert np.all(speed_bounded | stationary)

NO_FORCE_PARAMS = replace(full_params(weight=0.0), min_speed=1.0, max_speed=5.0)
@given(pos=positions(), vel=bounded_velocites(NO_FORCE_PARAMS.min_speed, NO_FORCE_PARAMS.max_speed), dt=st.floats(0.01,1))
def test_step_no_force(pos, vel, dt):
    state = State(pos, vel)
    params = NO_FORCE_PARAMS
    new_state = step(state, params, dt)
    assert np.allclose(state.pos + state.vel * dt, new_state.pos)
    assert np.allclose(state.vel, new_state.vel)

@given(pos=positions(), vel=velocities(), boost=shift_vector())
def test_step_galilean_covariant_unclamped(pos, vel, boost):
    p = unclamped_params()
    assume(well_separated(pos, p))
    dt = 0.01
    pos1, vel1 = step_from(pos, vel, dt, p)
    pos2, vel2 = step_from(pos, vel + boost, dt, p)
    np.testing.assert_allclose(vel2, vel1 + boost)
    np.testing.assert_allclose(pos2, pos1 + dt * boost)

@pytest.mark.xfail(strict=True, reason="speed clamp breaks Galilean covariance")
@given(pos=positions(), vel=velocities(), boost=shift_vector())
def test_step_galilean_covariant_clamped(pos, vel, boost):
    p = full_params()
    assume(well_separated(pos, p))
    dt = 0.01
    _, vel1 = step_from(pos, vel, dt, p)
    _, vel2 = step_from(pos, vel + boost, dt, p)
    np.testing.assert_allclose(vel2, vel1 + boost)


# Flat Topology Tests
@given(pos=positions(), vel=velocities())
def test_flat_wrap(pos, vel):
    wrap_pos, wrap_vel = Flat().wrap(pos, vel)
    assert np.array_equal(wrap_pos, pos) and np.array_equal(wrap_vel, vel)


# Torus Topology Tests

@given(pos = positions(), L=st.floats(0.1, 10))
def test_torus_minimum_image(pos, L):
    params = torus_params(L)
    assume(well_separated(pos, params))
    disp = displacement(pos, params.topology)
    assert np.all(np.abs(disp) <= L / 2)

def test_torus_displacement_known_value():
    pos = np.array([[0.1, 0.5], [0.9, 0.6]])
    disp_known = np.array([[[0.0, 0.0], [-0.2, 0.1]], 
                           [[0.2, -0.1], [0.0, 0.0]]])
    params = torus_params(size=1.0)
    disp_calc = displacement(pos, params.topology)
    assert np.allclose(disp_calc, disp_known)

@given(pos=positions(), L=st.floats(1.0, 10.0))
def test_torus_agrees_with_flat_locally(pos, L):
    local = pos * (L / 80)   # positions in [-10,10] -> [-L/4, L/4] at most... use L/80 for a safe margin
    assert np.allclose(Torus(L).displacement(local), Flat().displacement(local))

@given(pos=positions(), vel=velocities(), L = st.floats(0.1,10))
def test_torus_wrap(pos, vel, L):
    p = torus_params(L)
    assume(well_separated(pos,p))
    wrap_pos, wrap_vel = Torus(L).wrap(pos, vel)
    assert np.all((0 <= wrap_pos) & (wrap_pos < L)) and np.array_equal(wrap_vel, vel)


# Order Parameter Tests

def test_order_parameter_aligned_is_one():
    vel = np.tile([[3.0, -1.0]], (5, 1))
    assert np.isclose(State(np.zeros((5, 2)), vel).order_parameter, 1.0)

def test_order_parameter_antiparallel_is_zero():
    vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.isclose(State(np.zeros((2, 2)), vel).order_parameter, 0.0)

def test_order_parameter_four_cardinal_is_zero():
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert np.isclose(State(np.zeros((4, 2)), vel).order_parameter, 0.0)

@given(vel=bounded_velocites(0.1, 10.0), seed=st.integers(0, 2**32 - 1))
def test_order_parameter_rotation_invariant(vel, seed):
    Q = random_orthogonal(2, np.random.default_rng(seed))
    pos = np.zeros_like(vel)
    assert np.isclose(State(pos, vel @ Q.T).order_parameter,
                      State(pos, vel).order_parameter)

@given(vel=bounded_velocites(0.1, 10.0),
       factors=arrays(np.float64, (8, 1), elements=st.floats(0.1, 10.0)))
def test_order_parameter_speed_invariant(vel, factors):
    pos = np.zeros_like(vel)
    assert np.isclose(State(pos, vel * factors).order_parameter,
                      State(pos, vel).order_parameter)