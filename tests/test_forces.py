import numpy as np
from helpers import (
    alignment_from,
    alignment_params,
    box_params,
    cohesion_from,
    cohesion_params,
    forces_from,
    positions,
    separation_from,
    separation_params,
    separation_reference,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st

from sim import Flat, Params

# Cohesion Known Value tests

def test_cohesion_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    params = cohesion_params(radius=100.0, weight=10.0)
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
    params = alignment_params(radius=100.0, weight=10.0)
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

def test_box_boundary_force_known_values():
    params = box_params(size=2.0, weight=0.0, eps=0.5,
                        wall_radius=0.5, wall_weight=2.0)
    pos = np.array([[0.25, 1.0], [1.75, 1.0],
                    [1.0, 0.25], [1.0, 1.75],
                    [0.25, 1.75], [1.0, 1.0],
                    [0.5, 1.5], [-0.25, 2.25]])
    # At distance 0.25, the softened image force is
    # 2 * 0.5 / (0.5**2 + 0.5**2) = 2, directed into the box.
    # Interior points and points exactly at the cutoff feel no wall force.
    expected = np.array([[2.0, 0.0], [-2.0, 0.0],
                         [0.0, 2.0], [0.0, -2.0],
                         [2.0, -2.0], [0.0, 0.0],
                         [0.0, 0.0], [2.0, -2.0]])
    np.testing.assert_allclose(forces_from(pos, np.zeros_like(pos), params), expected)

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
