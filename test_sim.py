import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sim import displacement


def positions(n=8, d=2):
    return arrays(np.float64, (n, d),
                  elements=st.floats(-10, 10, allow_nan=False))

def shift_vector(d=2):
    return arrays(np.float64, (d,),
                  elements=st.floats(-10, 10, allow_nan=False))

@given(pos=positions())
def test_displacement_antisymmetric(pos):
    disp = displacement(pos, bounds=None)
    assert np.allclose(disp, -disp.transpose(1, 0, 2))


@given(pos=positions())
def test_displacement_zero_diagonal(pos):
    disp = displacement(pos, bounds=None)
    assert np.allclose(np.diagonal(disp, axis1=0, axis2=1), 0.0)

@given(pos=positions(), shift=shift_vector())
def test_displacement_shift_invariant(pos, shift):
    disp = displacement(pos, bounds=None)
    disp_shift = displacement(pos + shift, bounds=None)
    assert np.allclose(disp, disp_shift)

def test_displacement_known_values():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    known_disp = np.array([[[ 0.0, 0.0], [ 1.0, 0.0], [ 0.0, 1.0]],
                           [[-1.0,  0.0], [ 0.0,  0.0], [-1.0,  1.0]],
                           [[ 0.0, -1.0], [ 1.0, -1.0], [ 0.0,  0.0]]])
    disp = displacement(pos, bounds=None)
    assert np.allclose(known_disp, disp)