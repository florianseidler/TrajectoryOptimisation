import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, jacfwd
import pytest
import ipm_nlp_solver


def function_1(weights, slacks, lagrange_multipliers):
    return 2 * weights**4 + slacks + lagrange_multipliers


def function_2(weights, slacks, lagrange_multipliers):
    return 2 * weights[0]**4 + weights[1]**2 + slacks[0] + 3 * slacks[1]**4 + 4 * lagrange_multipliers[0] + lagrange_multipliers[1]**2


def function_3(weights, slacks, lagrange_multipliers):
    return 2 * weights[0]**4 + weights[1]**2 + 3 * weights[2]**2 + slacks[0] + 3 * slacks[1]**4 + 8 * slacks[2]**1.5 + 4 * lagrange_multipliers[0] + lagrange_multipliers[1]**2 + lagrange_multipliers[2]**4.5


def function_to_minimize(weights):
    return weights[0]**2 + 3 * weights[1]


def eq_constr_1(weights):
    return jnp.array(weights[0] ** 2 + weights[1] ** 2 - 1)


def ineq_constr_1(weights):
    return weights[0] + 2*weights[1] - 10


def testKKT_1():
    weights = 2
    slacks = 0
    lagrange_multipliers = 0
    kkt_weights, kkt_slacks, kkt_eq_lm, kkt_ineq_lm = ipm_nlp_solver.KKT(function_1, weights, slacks, lagrange_multipliers, 0, 0)
    assert (kkt_weights == 64)


def testKKT_2():
    weights = np.array([2., 1.])
    slacks = np.array([0., 1.])
    lagrange_multipliers = np.array([3., 0.])
    kkt_weights, kkt_slacks, kkt_eq_lm, kkt_ineq_lm = ipm_nlp_solver.KKT(function_2, weights, slacks, lagrange_multipliers, 2, 2)
    assert (kkt_weights == [[64.], [ 2.]])


def testKKT_3():
    weights = np.array([2., 1., 0.003])
    slacks = np.array([0., 1., 6.124])
    lagrange_multipliers = np.array([3., 0., 5.1])
    kkt_weights, kkt_slacks, kkt_eq_lm, kkt_ineq_lm = ipm_nlp_solver.KKT(function_3, weights, slacks, lagrange_multipliers, 3, 3)
    assert (kkt_slacks == [[0.], [12.], [181.85867255]])


def testgradvalues():
    weights = np.array([2., 1.])
    slacks = np.array([3., 1.])
    lagrange_multipliers = np.array([3., 4.])
    vec = ipm_nlp_solver.gradient_values(function_2, weights, slacks, lagrange_multipliers)
    assert (vec == np.array([64.,  2.,  1., 12.,  4.,  8.]).reshape(6, 1))


def test_lagrange_init():
    weights = np.array([2., 1.5])
    number_weights = 2
    number_equality_constraints = 1
    number_inequality_constraints = 1
    vec = ipm_nlp_solver.init_lagrange_multipliers(function_to_minimize, weights, number_weights, number_equality_constraints, number_inequality_constraints, eq_constr_1, ineq_constr_1)
    assert (vec == [[1.0000000e+00], [-4.4408921e-16]])


def test_jacobian_of_constraints():
    weights = np.array([2., 1.5])
    number_weights = 2
    number_equality_constraints = 1
    number_inequality_constraints = 1
    vec = ipm_nlp_solver.jacobian_of_constraints(weights, number_weights, eq_constr_1, ineq_constr_1, number_equality_constraints, number_inequality_constraints)
    assert (vec == [[ 4.,  1.], [ 3.,  2.], [ 0., -1.]])

