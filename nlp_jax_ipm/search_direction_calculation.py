import jax.numpy as jnp
from jax import grad, jit
from gradient_and_hessian import (hessian_approximation, hessian_real,
                                  regularize_hessian)


def calc_search_dir(objective_function, objective_function_with_barrier,
                    weights, slacks, lagrange_multipliers,
                    num_weights, num_equality_constraints,
                    num_inequality_constraints, diagonal_shift_val,
                    init_diagonal_shift_val, armijo_val, power_val,
                    barrier_val, approximate_hessian):

    """
    Compute primal-dual direction (Nocedal & Wright 19.12)
    Parameters
    ----------
    objective_function
    objective_function_with_barrier
    weights
    slacks
    lagrange_multipliers
    num_weights
    num_equality_constraints
    num_inequality_constraints
    diagonal_shift_val
    init_diagonal_shift_val
    armijo_val
    power_val
    barrier_val
    Returns
    -------
    search_direction
    """

    gradient_ = jit(grad(objective_function_with_barrier, [0, 1, 2]))(
        weights, slacks, lagrange_multipliers, barrier_val)
    gradient_weights = jnp.asarray(gradient_[0])
    gradient_slacks = jnp.asarray(gradient_[1])
    gradient_lagrange_multipliers = jnp.asarray(gradient_[2])
    gradient = jnp.concatenate(
        [gradient_weights, gradient_slacks, -gradient_lagrange_multipliers],
        axis=0)

    if approximate_hessian:
        hessian_ = hessian_approximation(
            objective_function, weights, slacks, lagrange_multipliers)
    else:
        hessian_ = hessian_real(objective_function_with_barrier, weights,
                                slacks, lagrange_multipliers, barrier_val)

    hessian_regularized, diagonal_shift_val = (
        regularize_hessian(hessian_, num_weights,
                           num_equality_constraints, num_inequality_constraints,
                           diagonal_shift_val, init_diagonal_shift_val,
                           armijo_val, power_val, barrier_val))

    # calculate search_direction
    gradient_transposed = - gradient.reshape((gradient.size, 1))
    search_direction = jit(jnp.linalg.solve)(
        hessian_regularized, gradient_transposed)
    search_direction = jnp.array(search_direction.reshape((gradient.size,)))

    if num_inequality_constraints or num_equality_constraints:
        # change sign definition for the multipliers' search direction
        search_direction = (
            search_direction.at[num_weights + num_inequality_constraints:]
            .set(-search_direction[num_weights + num_inequality_constraints:]))

    return search_direction
