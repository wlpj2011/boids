import numpy as np
from helpers import bounded_velocities, random_orthogonal
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sim import State

# Random Generation Tests

@given(seed=st.integers(0, 2**32 - 1), d=st.integers(1, 5))
def test_random_orthogonal_is_orthogonal(seed, d):
    Q = random_orthogonal(d, np.random.default_rng(seed))
    assert np.allclose(Q.T @ Q, np.eye(d))
    assert np.isclose(abs(np.linalg.det(Q)), 1.0)


# Order Parameter Tests

def test_order_parameter_aligned_is_one():
    vel = np.tile([[3.0, -1.0]], (5, 1))
    assert np.isclose(State(np.zeros((5, 2)), vel).order_parameter, 1.0)

def test_order_parameter_antiparallel_is_zero():
    vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.isclose(State(np.zeros((2, 2)), vel).order_parameter, 0.0)

def test_order_parameter_four_cardinal_is_zero():
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert np.isclose(State(np.zeros((4, 2)), vel).order_parameter, 0.0)

@given(vel=bounded_velocities(0.1, 10.0), seed=st.integers(0, 2**32 - 1))
def test_order_parameter_rotation_invariant(vel, seed):
    Q = random_orthogonal(2, np.random.default_rng(seed))
    pos = np.zeros_like(vel)
    assert np.isclose(State(pos, vel @ Q.T).order_parameter,
                      State(pos, vel).order_parameter)

@given(vel=bounded_velocities(0.1, 10.0),
       factors=arrays(np.float64, (8, 1), elements=st.floats(0.1, 10.0)))
def test_order_parameter_speed_invariant(vel, factors):
    pos = np.zeros_like(vel)
    assert np.isclose(State(pos, vel * factors).order_parameter,
                      State(pos, vel).order_parameter)