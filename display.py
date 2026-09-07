from dataclasses import fields, replace

import numpy as np
import pygame
from numpy.typing import NDArray

from sim import Flat, Params, State, Torus, initial_state, step

ARROW_SIZE = 0.02

KEY_BINDINGS = {
    pygame.K_q: ("cohesion_weight", 1.25),  pygame.K_a: ("cohesion_weight", 0.8),
    pygame.K_w: ("alignment_weight", 1.25), pygame.K_s: ("alignment_weight", 0.8),
    pygame.K_e: ("separation_weight", 1.25), pygame.K_d: ("separation_weight", 0.8),
    # radii on the next row, speeds on the row after
}

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


def draw(screen, font, state, params):
    screen.fill((0, 0, 0))
    arrows = arrow_vertices(state, ARROW_SIZE)
    arrows = to_screen(arrows, screen.get_size())
    for arrow in arrows:
        pygame.draw.polygon(screen, (255,0,0), arrow.tolist())
    y = 10
    for f in fields(params):
        surface = font.render(f"{f.name}: {getattr(params, f.name)}", True, (200, 200, 200))
        screen.blit(surface, (10, y))
        y += font.get_linesize()
    surface = font.render(f"phi: {state.order_parameter:.3f}", True, (200, 200, 200))
    screen.blit(surface, (10, y))


def main() -> None:
    """Owns the window, the clock, and the loop."""
    w = 1000 # Width of pygame screen
    h = 1000 # Height of pygame screen

    n = 40 # Number of boids
    d = 2 # Number of dimensions
    dt = 1/60
    seed = 0 # rng seed

    params = Params(topology=Torus(1.0), eps_smooth=0.05,
                cohesion_radius=0.1,   cohesion_weight=10,
                alignment_radius=0.12, alignment_weight=0.35,
                separation_radius=0.05, separation_weight=0.1,
                min_speed=0.2, max_speed=0.5)
    rng = np.random.default_rng(seed)
    state = initial_state(n, d, rng)

    pygame.init()
    font = pygame.font.Font(None, 24)
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN and event.key in KEY_BINDINGS:
                field, factor = KEY_BINDINGS[event.key]
                params = replace(params, **{field: getattr(params, field) * factor})

        state = step(state, params, dt)

        draw(screen, font, state, params)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()