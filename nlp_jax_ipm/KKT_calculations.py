import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit

from gradient_and_hessian import gradient_analytic


def KKT(objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints,
        num_equality_constraints, barrier_val):
    """
    Calculate the first-order Karush-Kuhn-Tucker conditions. Irrelevant conditions are set to zero.

    Parameters
    ----------
    objective_function
    weights
    slacks
    lagrange_multipliers
    num_inequality_constraints
    num_equality_constraints
    barrier_val

    Returns
    -------
    kkt_weights: gradient of Lagrangian with respect to x (weights)
    kkt_slacks: gradient of Lagrangian with respect to s (slack variables)
    kkt_equality_lagrange_multipliers: gradient of Lagrangian with respect to equality constraints Lagrange multipliers
    kkt_inequality_lagrange_multipliers: gradient of Lagrangian with respect to ineq. constraints Lagrange multipliers
    """

    kkt_results = gradient_analytic(objective_function, weights, slacks, lagrange_multipliers, barrier_val)
    num_weights = jnp.size(weights)

    if num_inequality_constraints and num_equality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = kkt_results[num_weights:(num_weights + num_inequality_constraints)] * slacks
        kkt_equality_lagrange_multipliers = \
            kkt_results[(num_weights + num_inequality_constraints):(num_weights + num_inequality_constraints +
                                                                    num_equality_constraints)]
        kkt_inequality_lagrange_multipliers = \
            kkt_results[(num_weights + num_inequality_constraints + num_equality_constraints):]
    elif num_equality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = jnp.float32(0.0)
        kkt_equality_lagrange_multipliers = \
            kkt_results[(num_weights + num_inequality_constraints):(num_weights + num_inequality_constraints +
                                                                    num_equality_constraints)]
        kkt_inequality_lagrange_multipliers = jnp.float32(0.0)
    elif num_inequality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = kkt_results[num_weights:(num_weights + num_inequality_constraints)] * slacks
        kkt_equality_lagrange_multipliers = jnp.float32(0.0)
        kkt_inequality_lagrange_multipliers = \
            kkt_results[(num_weights + num_inequality_constraints + num_equality_constraints):]
    else:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = jnp.float32(0.0)
        kkt_equality_lagrange_multipliers = jnp.float32(0.0)
        kkt_inequality_lagrange_multipliers = jnp.float32(0.0)

    return kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers
