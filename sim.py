from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Params:
    """Simulation parameters. Steering weights come later; for one
    particle you need nothing, but keep the class so signatures are
    stable."""
    bounds : None
    cohesion_radius : float
    cohesion_weight : float
    alignment_radius : float
    alignment_weight : float


@dataclass(frozen=True)
class State:
    """pos, vel: (n, d) float64 arrays."""
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
    new_force = forces(state, params)
    new_pos = state.pos + dt * state.vel
    new_vel = state.vel + dt * new_force
    return State(new_pos, new_vel)

def forces(state: State, params: Params) -> NDArray[np.float64]:
    disp = displacement(state.pos, bounds=params.bounds) # (n,n,d)
    dist = np.linalg.norm(disp, axis=-1) # (n,n)
    cohesion_force =  cohesion(disp, dist, params)
    alignment_force = alignment(state.pos, dist, params)
    return cohesion_force + alignment_force

def displacement(pos, bounds: None=None) -> NDArray[np.float64]:
    """
    Calculate the displacement between each pair of boids
    displacement[i,j] = pos[j] - pos[i]
    """
    return pos[None, :, :] - pos[:, None, :] # (n,n,d)

def cohesion(disp: NDArray[np.float64], dist: NDArray[np.float64], params: Params) -> NDArray[np.float64]:
    close_boids = (dist < params.cohesion_radius) & ~np.eye(len(dist), dtype=bool) # (n,n)
    count = np.sum(close_boids, axis=1, keepdims=1) # (n,1)
    mean_disp = np.sum(disp * close_boids[..., None], axis=1) / np.maximum(count, 1)  # (n, d)
    cohesion_force = mean_disp * params.cohesion_weight
    return cohesion_force

def alignment(vel, dist, params) -> NDArray[np.float64]:
    close_boids = (dist < params.alignment_radius) & ~np.eye(len(dist), dtype=bool)
    count = np.sum(close_boids, axis=1, keepdims=1) # (n,1)
    vel_diff = vel[None, :, :] - vel[:, None, :]
    mean_vel = np.sum(vel_diff * close_boids[..., None], axis=1) / np.maximum(count, 1)  # (n, d)
    alignment_force = mean_vel * params.alignment_weight
    return alignment_force

if __name__ == "__main__":
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vel = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    state = State(pos, vel)
    params = Params(cohesion_radius=2, cohesion_weight=10, alignment_radius=2, alignment_weight=10, bounds=None)
    disp = displacement(state.pos, bounds=params.bounds) # (n,n,d)
    dist = np.linalg.norm(disp, axis=-1) # (n,n)
    print(f"position:\n{pos}")
    print(f"velocity:\n{vel}")
    print(params)
    print(f"cohesion:\n{cohesion(disp, dist, params)}")
    print(f"alignment:\n{alignment(state.vel, dist, params)}")