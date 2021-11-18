import pytest
import numpy as np
import NLPsolverSingleConstraint


def testnlpsolver():
    exponent = 2
    ampli = np.array([1, 1])
    offs = 0
    constantOfIneqConstraint = 4
    assert (nlpsolver(exponent, ampli, offs, constantOfIneqConstraint) == -(1/2))


def testintegrationgauss5point():
    def fct(x):
        return 2 * x
    low = 2
    up = 5
    assert (21.5 > gaussquad5(fct, low, up) < 20.5)

