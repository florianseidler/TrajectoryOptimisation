import pytest
import numpy as np
import NLPsolverSingleConstraint

# content of NLPsolverSingleConsraint.py
exponent = 2
ampli = np.array([1, 1])
offs = 0
constantOfIneqConstraint = 4

def testnlpsolver(exponent, ampli, offs, constantOfIneqConstraint):
    assert nlpsolver(exponent, ampli, offs, constantOfIneqConstraint) == -(1/2)
