import numpy as np
import jax.numpy as jnp
from jax import grad
import pytest
import ipm_nlp_solver

def function(weights, slacks, lagrange_multipliers):
    return 2 * weights**4 + slacks + lagrange_multipliers

def testKKT():
    weights = 2
    slacks = 0
    lagrange_multipliers = 0
    kkt_weights, kkt_slacks, kkt_eq_lm, kkt_ineq_lm = ipm_nlp_solver.KKT(function, weights, slacks, lagrange_multipliers, 0, 0)
    assert (kkt_weights == 64)


weights = np.array(2.)
slacks = np.array(0.)
lagrange_multipliers = np.array(0.)
kkt_weights, kkt_slacks, kkt_eq_lm, kkt_ineq_lm = ipm_nlp_solver.KKT(function, weights, slacks, lagrange_multipliers, 0, 0)
print(kkt_weights)