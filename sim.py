from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Params:
    """Simulation parameters. Steering weights come later; for one
    particle you need nothing, but keep the class so signatures are
    stable."""


@dataclass(frozen=True)
class State:
    """pos, vel: (N, d) float64 arrays."""
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]


def initial_state(n: int, d: int, rng: np.random.Generator) -> State:
    """Random positions in the unit box, random unit velocities."""
    pos = rng.random((n,d))
    vel = rng.normal(size=(n,d))
    vel = vel/np.linalg.norm(vel, axis=1, keepdims=True)
    return State(pos, vel)


def step(state: State, params: Params, dt: float) -> State:
    """Advance one timestep. For now: pos + dt * vel, constant vel."""
    new_pos = state.pos + dt * state.vel
    return State(new_pos, state.vel)