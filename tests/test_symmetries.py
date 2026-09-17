import dataclasses

import numpy as np
import pytest
from helpers import (
    BOUNDARY_FORCES,
    FORCES,
    SUBJECTS,
    TOPOLOGIES,
    TRANSLATION_TOPOLOGIES,
    positions,
    random_orthogonal,
    scale_lengths,
    shift_vector,
    velocities,
    well_separated,
)
from hypothesis import assume, given
from hypothesis import strategies as st


@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_rotation_equivariant(force_from, p, exponent, pos, vel, seed):
    assume(well_separated(pos, p))
    Q = random_orthogonal(2, np.random.default_rng(seed))
    assert np.allclose(force_from(pos @ Q.T, vel @ Q.T, p),
                       force_from(pos, vel, p) @ Q.T)

@pytest.mark.parametrize("topology", TRANSLATION_TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_translation_invariant(topology, force_from, p, exponent, shift, pos, vel):
    p = dataclasses.replace(p, topology=topology)
    assume(well_separated(pos, p))
    assert np.allclose(force_from(pos + shift, vel, p), force_from(pos, vel, p))

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), shift=shift_vector())
def test_galilean_invariant(topology, force_from, p, exponent, shift, pos, vel):
    p = dataclasses.replace(p, topology=topology)
    assume(well_separated(pos, p))
    assert np.allclose(force_from(pos, vel + shift, p), force_from(pos, vel, p))

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", SUBJECTS)
@given(pos=positions(), vel=velocities(), seed=st.integers(0, 2**32 - 1))
def test_permutation_equivariant(topology, force_from, p, exponent, pos, vel, seed):
    p = dataclasses.replace(p, topology=topology)
    assume(well_separated(pos, p))
    perm = np.random.default_rng(seed).permutation(len(pos))
    assert np.allclose(force_from(pos[perm], vel[perm], p), force_from(pos, vel, p)[perm])

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("force_from, p, exponent", FORCES)
@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_position_scaling(topology, force_from, p, exponent, pos, vel, lam):
    p = dataclasses.replace(p, topology=topology)
    assume(well_separated(pos, p))
    assert np.allclose(force_from(lam * pos, vel, scale_lengths(p, lam)),
                       lam ** exponent * force_from(pos, vel, p))

@pytest.mark.parametrize("force_from, p, exponent", BOUNDARY_FORCES)
@given(pos=positions(), vel=velocities(), lam=st.floats(0.1, 10))
def test_position_scaling_boundary(force_from, p, exponent, pos, vel, lam):
    assume(well_separated(pos, p))
    assert np.allclose(force_from(lam * pos, vel, scale_lengths(p, lam)),
                       lam ** exponent * force_from(pos, vel, p))