import numpy as np
import pygame
from numpy.typing import NDArray

from sim import Params, State, initial_state, step

ARROW_SIZE = 0.02

def arrow_vertices(state: State, r: float) -> NDArray[np.float64]:
    """Triangle vertices per particle, sim coords, (N, 3, 2)."""
    unit_heading = state.vel / np.linalg.norm(state.vel, axis=1, keepdims=True)
    normal = unit_heading[..., :: -1] * np.array([-1.0, 1.0])
    nose = state.pos + unit_heading * r
    left = state.pos - unit_heading * r / 2 + normal * r/4
    right = state.pos - unit_heading * r / 2 - normal * r/4
    return np.stack([nose, left, right], axis=1)


def to_screen(pos: NDArray[np.float64], size: tuple[int, int]) -> NDArray[np.float64]:
    """Map sim coordinates (unit box, y up) to pixels (y down).
    The y-flip and scale live here and only here."""
    out = pos * np.array([size[0], size[1]])
    out[..., 1] = (1 - pos[..., 1]) * size[1]
    return out


def draw(screen, state):
    screen.fill((0, 0, 0))
    arrows = arrow_vertices(state, ARROW_SIZE)
    arrows = to_screen(arrows, screen.get_size())
    for arrow in arrows:
        pygame.draw.polygon(screen, (255,0,0), arrow.tolist())


def main() -> None:
    """Owns the window, the clock, and the loop."""
    w = 1000 # Width of pygame screen
    h = 1000 # Height of pygame screen

    n = 100 # Number of boids
    d = 2 # Number of dimensions
    dt = 1/60
    seed = 1 # rng seed

    params = Params(bounds=None, eps_smooth=0.05,
                    cohesion_radius=0.3, cohesion_weight=5, 
                    alignment_radius=0.15, alignment_weight=1, 
                    separation_radius=0.05, separation_weight=0.05)
    rng = np.random.default_rng(seed)
    state = initial_state(n, d, rng)

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        state = step(state, params, dt)

        draw(screen, state)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()