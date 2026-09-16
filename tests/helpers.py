# tests/helpers.py
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from sim import (
    Flat,
    Params,
    State,
    Torus,
    alignment,
    cohesion,
    displacement,
    forces,
    separation,
    step,
)

# Test Generation Helpers

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

@st.composite
def bounded_velocites(draw, min_vel, max_vel, n: int=8, d: int=2):
    directions = draw(arrays(np.float64, (n, d), elements=st.floats(-1, 1)))
    speeds = draw(arrays(np.float64, (n, 1), elements=st.floats(min_vel, max_vel)))
    assume(np.all(np.linalg.norm(directions, axis=1) > 1e-6))
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return unit * speeds

def well_separated(pos, params, eps=1e-6, min_sep=1e-3):
    disp = displacement(pos, topology=params.topology)
    dist = np.linalg.norm(disp, axis=-1)
    radii = params.radii
    off_radii = all(np.all(np.abs(dist - r) > eps) for r in radii)
    not_nearly_coincident = np.all((dist == 0) | (dist > min_sep))
    off_unstable = all(np.all(np.abs(np.abs(disp) - s) > eps) for s in params.topology.unstable_distances())
    return off_radii and not_nearly_coincident and off_unstable

# Param Builders

def cohesion_params(radius=2.0, weight=1.0):
    return Params(topology=Flat(), eps_smooth=0.01,
                  cohesion_radius=radius, cohesion_weight=weight, 
                  alignment_radius=1.0, alignment_weight=0.0, 
                  separation_radius=1.0, separation_weight=0.0,
                  min_speed=1.0, max_speed=100.0)

def alignment_params(radius=2.0, weight=1.0):
    return Params(topology=Flat(), eps_smooth=0.01, 
                  cohesion_radius=1.0, cohesion_weight=0.0, 
                  alignment_radius=radius, alignment_weight=weight, 
                  separation_radius=1.0, separation_weight=0.0,
                  min_speed=1.0, max_speed=100.0)

def separation_params(radius=2.0, weight=1.0, eps = 0.01):
    return Params(topology=Flat(), eps_smooth=eps, 
                  cohesion_radius=1.0, cohesion_weight=0.0, 
                  alignment_radius=1.0, alignment_weight=0.0, 
                  separation_radius=radius, separation_weight=weight,
                  min_speed=1.0, max_speed=100.0)

def full_params(radius=2.0, weight=1.0, eps = 0.01):
    return Params(topology=Flat(), eps_smooth=eps, 
                  cohesion_radius=radius, cohesion_weight=weight, 
                  alignment_radius=radius, alignment_weight=weight, 
                  separation_radius=radius, separation_weight=weight,
                  min_speed=1.0, max_speed=100.0)

def torus_params(size = 1.0, radius = 0.25, weight = 1.0, eps = 0.01):
    return Params(topology=Torus(size), eps_smooth=eps, 
                  cohesion_radius=radius, cohesion_weight=weight, 
                  alignment_radius=radius, alignment_weight=weight, 
                  separation_radius=radius, separation_weight=weight,
                  min_speed=1.0, max_speed=100.0)

def unclamped_params():
    return replace(full_params(), min_speed=0.0, max_speed=np.inf)


# State Wrappers
def cohesion_from(pos, vel, params):
    disp = displacement(pos, topology=params.topology)
    return cohesion(disp, np.linalg.norm(disp, axis=-1), params)

def alignment_from(pos, vel, params):
    disp = displacement(pos, topology=params.topology)
    return alignment(vel, np.linalg.norm(disp, axis=-1), params)

def separation_from(pos, vel, params):
    disp = displacement(pos, topology=params.topology)
    return separation(disp, np.linalg.norm(disp, axis=-1), params)

def forces_from(pos, vel, params):
    state = State(pos, vel)
    force = sum(forces(state, params).values())
    return force

def step_from(pos, vel, dt, params):
    state = State(pos, vel)
    new_state = step(state, params, dt)
    return (new_state.pos, new_state.vel)

# Reference Implementation
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


# Global Lists
FORCES = [
    pytest.param(cohesion_from, cohesion_params(), 1, id="cohesion"),
    pytest.param(alignment_from, alignment_params(), 0, id="alignment"),
    pytest.param(separation_from, separation_params(), -1, id="separation"),
]

SUBJECTS = FORCES + [pytest.param(forces_from, full_params(), None, id="all")]

LENGTH_FIELDS = ("cohesion_radius", "alignment_radius",
                 "separation_radius", "eps_smooth")

VELOCITY_FIELDS = ("min_speed", "max_speed",)

TOPOLOGIES = [pytest.param(Flat(), id="flat"), pytest.param(Torus(20.0), id="torus")]

def scale_lengths(p: Params, lam: float) -> Params:
    return replace(p, **{f: getattr(p, f) * lam for f in LENGTH_FIELDS})