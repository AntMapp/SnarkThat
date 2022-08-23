import argparse #written module__interface
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.integrate
from numpy.polynomial.polynomial import Polynomial
import pytest 
from pytest import fixture
from snarkthat_main import SnarkThat

snarkthat=SnarkThat()

def test_snarkthat_gates():
    assert snarkthat.gate1(3) == 3**2
    assert snarkthat.gate2(3) == 3**3
    assert snarkthat.gate3(3) == 3*2
    assert snarkthat.gate4(3,3) == 27*6
    assert snarkthat.gate5(3,3) == snarkthat.qeval(3,3)

def test_arithmetic_circuit():
    assert snarkthat.arithmetic_circuit(3,3)[0] == 1
    assert snarkthat.arithmetic_circuit(3,3)[1] == 3
    assert snarkthat.arithmetic_circuit(3,3)[2] == 3
    assert snarkthat.arithmetic_circuit(3,3)[3] == snarkthat.qeval(3,3)
    assert snarkthat.arithmetic_circuit(3,3)[4] == 3**2
    assert snarkthat.arithmetic_circuit(3,3)[5] == 3**3
    assert snarkthat.arithmetic_circuit(3,3)[6] == 3*2
    assert snarkthat.arithmetic_circuit(3,3)[7] == (3**3) * (2*3)

def test_poly_coeffs():
    assert type(snarkthat.A_poly_coeffs()) == np.ndarray
    assert snarkthat.A_poly_coeffs().shape == (8,5)

    assert type(snarkthat.B_poly_coeffs()) == np.ndarray
    assert snarkthat.B_poly_coeffs().shape == (8,5)
    
    assert type(snarkthat.C_poly_coeffs()) == np.ndarray
    assert snarkthat.C_poly_coeffs().shape == (8,5)



