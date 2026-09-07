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
    separation_radius : float
    separation_weight : float
    eps_smooth : float
    min_speed : float
    max_speed : float
    @property
    def radii(self) -> tuple[float, ...]:
        return (self.cohesion_radius, self.alignment_radius, self.separation_radius)


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
    """
    Advance one timestep.
    pos' = pos + dt * vel
    vel' = vel + forces * dt
    """
    new_force = sum(forces(state, params).values())
    new_pos = state.pos + dt * state.vel
    new_vel = state.vel + dt * new_force
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)          # (n, 1)
    scale = np.clip(speed, params.min_speed, params.max_speed) / np.maximum(speed, 1e-12)
    new_vel = new_vel * scale
    return State(new_pos, new_vel)

def forces(state: State, params: Params) -> dict[str, NDArray[np.float64]]:
    disp = displacement(state.pos, bounds=params.bounds) # (n,n,d)
    dist = np.linalg.norm(disp, axis=-1) # (n,n)
    cohesion_force =  cohesion(disp, dist, params)
    alignment_force = alignment(state.vel, dist, params) 
    separation_force = separation(disp, dist, params)
    current_forces = {}
    current_forces["cohesion"] = cohesion_force
    current_forces["alignment"] = alignment_force
    current_forces["separation"] = separation_force
    return current_forces

def displacement(pos, bounds: None=None) -> NDArray[np.float64]:
    """
    Calculate the displacement between each pair of boids
    displacement[i,j] = pos[j] - pos[i]
    """
    return pos[None, :, :] - pos[:, None, :] # (n,n,d)

def cohesion(disp: NDArray[np.float64], dist: NDArray[np.float64], params: Params) -> NDArray[np.float64]:
    close_boids = (dist < params.cohesion_radius) & ~np.eye(len(dist), dtype=bool) # (n,n)
    count = np.sum(close_boids, axis=1, keepdims=True) # (n,1)
    mean_disp = np.sum(disp * close_boids[..., None], axis=1) / np.maximum(count, 1)  # (n, d)
    cohesion_force = mean_disp * params.cohesion_weight
    return cohesion_force

def alignment(vel, dist, params) -> NDArray[np.float64]:
    close_boids = (dist < params.alignment_radius) & ~np.eye(len(dist), dtype=bool)
    count = np.sum(close_boids, axis=1, keepdims=True) # (n,1)
    vel_diff = vel[None, :, :] - vel[:, None, :]
    mean_vel = np.sum(vel_diff * close_boids[..., None], axis=1) / np.maximum(count, 1)  # (n, d)
    alignment_force = mean_vel * params.alignment_weight
    return alignment_force

def separation(disp, dist, params) -> NDArray[np.float64]:
    """
    sums -disp[i,j]/dist[i,j]^2 over nearby boids.
    """
    close_boids = (0 < dist) & (dist < params.separation_radius)          # (n, n)
    contrib = np.divide(-disp, dist[..., None] ** 2 + params.eps_smooth **2,
                        out=np.zeros_like(disp),
                        where=close_boids[..., None])                      # (n, n, d)
    sum_contrib = np.sum(contrib, axis=1)
    return sum_contrib * params.separation_weight

if __name__ == "__main__":
    pos = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    vel = np.array([[-1.0, 3.0], [3.0, 5.0], [-2.0, 1.0]])
    
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    vel = np.array([[1.0, 2.0], [1.0, 4.0], [-5.0, -2.0]])
    state = State(pos, vel)
    params = Params(cohesion_radius=5, cohesion_weight=10, 
                    alignment_radius=20, alignment_weight=40, 
                    separation_radius=4.0, separation_weight=2.0, 
                    eps_smooth=0.00, 
                    min_speed=1.0,
                    max_speed=100.0,
                    bounds=None)
    disp = displacement(state.pos, bounds=params.bounds) # (n,n,d)
    dist = np.linalg.norm(disp, axis=-1) # (n,n)
    print(f"position:\n{pos}")
    print(f"velocity:\n{vel}")
    print(params)
    print(f"cohesion:\n{cohesion(disp, dist, params)}")
    print(f"alignment:\n{alignment(state.vel, dist, params)}")
    print(f"separation:\n{separation(disp, dist, params)}")