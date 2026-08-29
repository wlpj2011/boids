import numpy as np
import pygame
from numpy.typing import NDArray

from sim import Params, State, initial_state, step


def to_screen(pos: NDArray[np.float64], size: tuple[int, int]) -> NDArray[np.float64]:
    """Map sim coordinates (unit box, y up) to pixels (y down).
    The y-flip and scale live here and only here."""
    out = pos * np.array([size[0], size[1]])
    out[..., 1] = (1 - pos[..., 1]) * size[1]
    return out


def draw(screen, state):
    screen.fill((0, 0, 0))
    px = to_screen(state.pos, screen.get_size())
    for p in px:
        pygame.draw.circle(screen, (255, 0, 0), p.tolist(), 2)


def main() -> None:
    """Owns the window, the clock, and the loop."""
    w = 600 # Width of pygame screen
    h = 600 # Height of pygame screen

    n = 1 # Number of boids
    d = 2 # Number of dimensions
    dt = 1/60
    seed = 1 # rng seed

    params = Params()
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