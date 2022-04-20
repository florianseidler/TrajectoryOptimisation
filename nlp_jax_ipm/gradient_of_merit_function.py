import jax.numpy as jnp
from jax import grad, jit

from .merit_function import sum_equality_values, sum_inequality_values


def gradient_merit_function(cost_function, weights, slacks, search_direction,
                            num_weights, barrier_val, minimal_step,
                            equality_constraints=None,
                            inequality_constraints=None,
                            num_equality_constraints=0,
                            num_inequality_constraints=0,
                            merit_function_parameter=10.0):

    """
    Calculates the gradient of the merit function.

    Parameters
    ----------
    cost_function
    weights
    slacks
    search_direction
    num_weights
    barrier_val
    minimal_step
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints
    merit_function_parameter

    Returns
    -------
    gradient_merit
    """

    if num_equality_constraints and num_inequality_constraints:
        equality_sum = sum_equality_values(
            equality_constraints, weights, num_equality_constraints)
        inequality_sum = sum_inequality_values(
            inequality_constraints, weights, slacks, num_inequality_constraints)

        gradient_merit = (jnp.dot(jit(grad(cost_function))(weights),
                          search_direction[:num_weights])
                          - merit_function_parameter
                          * (equality_sum + inequality_sum)
                          - jit(jnp.dot)(barrier_val / (slacks + minimal_step),
                          search_direction[num_weights:]))
        return gradient_merit

    elif num_equality_constraints:

        equality_sum = sum_equality_values(
            equality_constraints, weights, num_equality_constraints)

        gradient_merit = (jnp.dot(jit(grad(cost_function))(weights),
                          search_direction[:num_weights])
                          - merit_function_parameter * equality_sum)
        return gradient_merit

    elif num_inequality_constraints:

        inequality_sum = sum_inequality_values(
            inequality_constraints, weights, slacks, num_inequality_constraints)
        gradient_merit = (jnp.dot(jit(grad(cost_function))(weights),
                          search_direction[:num_weights])
                          - merit_function_parameter * inequality_sum
                          - jit(jnp.dot)(barrier_val / (slacks + minimal_step),
                          search_direction[num_weights:]))
        return gradient_merit

    else:
        gradient_merit = jit(jnp.dot)(
            grad(cost_function)(weights), search_direction[:num_weights])
        return gradient_merit
