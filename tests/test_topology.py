import numpy as np
import pytest
from helpers import (
    TOPOLOGIES,
    positions,
    random_orthogonal,
    shift_vector,
    torus_params,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st

from sim import Flat, Torus, displacement

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
    local = pos * (L / 80)   # positions in [-10,10] -> [-L/8, L/8], so displacements stay well inside L/2
    assert np.allclose(Torus(L).displacement(local), Flat().displacement(local))

@given(pos=positions(), vel=velocities(), L = st.floats(0.1,10))
def test_torus_wrap(pos, vel, L):
    p = torus_params(L)
    assume(well_separated(pos,p))
    wrap_pos, wrap_vel = Torus(L).wrap(pos, vel)
    assert np.all((0 <= wrap_pos) & (wrap_pos < L)) and np.array_equal(wrap_vel, vel)