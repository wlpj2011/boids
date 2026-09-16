import dataclasses

import numpy as np
import pytest
from helpers import (
    TOPOLOGIES,
    bounded_velocities,
    full_params,
    positions,
    shift_vector,
    step_from,
    unclamped_params,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st

from sim import SPEED_FLOOR, State, step


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

NO_FORCE_PARAMS = dataclasses.replace(full_params(weight=0.0), min_speed=1.0, max_speed=5.0)
@given(pos=positions(), vel=bounded_velocities(NO_FORCE_PARAMS.min_speed, NO_FORCE_PARAMS.max_speed), dt=st.floats(0.01,1))
def test_step_no_force(pos, vel, dt):
    state = State(pos, vel)
    params = NO_FORCE_PARAMS
    new_state = step(state, params, dt)
    assert np.allclose(state.pos + state.vel * dt, new_state.pos)
    assert np.allclose(state.vel, new_state.vel)

@pytest.mark.parametrize("topology", TOPOLOGIES)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_step_permutation_equivariant(topology, pos, vel, seed):
    p = full_params()
    p = dataclasses.replace(p, topology=topology)
    assume(well_separated(pos, p))
    perm = np.random.default_rng(seed).permutation(len(pos))
    next_step = step_from(pos, vel, 0.01, p)
    next_step_permuted = step_from(pos[perm], vel[perm], 0.01, p)
    assert np.allclose((next_step[0][perm], next_step[1][perm]), next_step_permuted)

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