import dataclasses

import numpy as np
import pytest
from helpers import (
    TOPOLOGIES,
    TRANSLATION_TOPOLOGIES,
    boundary_from,
    box_params,
    forces_from,
    full_params,
    positions,
    random_orthogonal,
    shift_vector,
    torus_params,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st

from sim import Box, Flat, State, Torus, displacement

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

@pytest.mark.parametrize("topology", TRANSLATION_TOPOLOGIES)
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

@pytest.mark.parametrize("topology", [Flat(), Box(1.0, 0.05, 1.0)])
@given(pos=positions(), vel=velocities())
def test_wrap_identity(topology, pos, vel):
    state = State(pos, vel)
    assert topology.wrap(state) is state


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
    wrapped = Torus(L).wrap(State(pos, vel))
    assert np.all((0 <= wrapped.pos) & (wrapped.pos < L))
    assert np.array_equal(wrapped.vel, vel)

# Box Topology Tests

@given(pos=positions(), vel=velocities())
def test_box_wrap(pos, vel):
    state = State(pos, vel)
    wrapped = Box(20.0, 2.0, 1.0).wrap(state)
    assert np.array_equal(wrapped.pos, pos) and np.array_equal(wrapped.vel, vel)

def test_box_boundary_one_wall():
    # Boid 0 is 1.0 inside the lower-x wall and nowhere near any other;
    # boid 1 is interior. Kernel with eps=0 gives 1/(2d) = 0.5 per wall.
    pos = np.array([[1.0, 10.0], [10.0, 10.0]])
    vel = np.array([[1.0, 2.0], [-3.0, 4.0]])
    params = box_params(wall_weight=2.0, eps=0.0)
    known = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(boundary_from(pos, vel, params), known)

def test_box_boundary_upper_wall():
    pos = np.array([[19.0, 10.0], [10.0, 10.0]])
    vel = np.array([[1.0, 2.0], [-3.0, 4.0]])
    params = box_params(wall_weight=2.0, eps=0.0)
    known = np.array([[-1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(boundary_from(pos, vel, params), known)

def test_box_boundary_corner():
    # 1.0 from lower-x, 0.5 from lower-y: 1/(2*1) and 1/(2*0.5).
    pos = np.array([[1.0, 0.5], [10.0, 10.0]])
    vel = np.array([[1.0, 2.0], [-3.0, 4.0]])
    params = box_params(wall_weight=1.0, eps=0.0)
    known = np.array([[0.5, 1.0], [0.0, 0.0]])
    assert np.allclose(boundary_from(pos, vel, params), known)

def test_box_boundary_interior():
    pos = np.array([[10.0, 10.0], [5.0, 15.0], [3.0, 3.0]])
    vel = np.array([[1.0, 2.0], [-3.0, 4.0], [0.0, 1.0]])
    params = box_params(eps=0.0)
    assert np.allclose(boundary_from(pos, vel, params), 0.0)

@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_box_boundary_permutation_equivariant(pos, vel, seed):
    p = box_params()
    assume(well_separated(pos, p))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(boundary_from(pos[perm], vel[perm], p),
                       boundary_from(pos, vel, p)[perm])

@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_box_boundary_galilean_invariant(pos, vel, shift):
    p = box_params()
    assume(well_separated(pos, p))
    assert np.allclose(boundary_from(pos, vel + shift, p),
                       boundary_from(pos, vel, p))

# Generators of the square's symmetry group, as linear maps about the center.
DIHEDRAL_GENERATORS = [
    pytest.param(np.array([[-1.0, 0.0], [0.0, 1.0]]), id="reflect_x"),
    pytest.param(np.array([[0.0, -1.0], [1.0, 0.0]]), id="quarter_turn"),
]

@pytest.mark.parametrize("Q", DIHEDRAL_GENERATORS)
@given(pos=positions(), vel=velocities())
def test_box_dihedral_equivariant(Q, pos, vel):
    box = Box(20.0, 2.0, 1.0)
    p = dataclasses.replace(full_params(), topology=box)
    assume(well_separated(pos, p))
    center = np.full(pos.shape[1], box.size / 2)
    pos_sym = center + (pos - center) @ Q.T
    assert np.allclose(forces_from(pos_sym, vel @ Q.T, p),
                       forces_from(pos, vel, p) @ Q.T)