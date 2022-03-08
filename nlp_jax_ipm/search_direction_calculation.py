import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit
from gradient_and_hessian import gradient_analytic, hessian_approximation, regularize_hessian


def calc_search_dir(objective_function, weights, slacks, lagrange_multipliers, num_weights, num_equality_constraints,
                    num_inequality_constraints, diagonal_shift_val, init_diagonal_shift_val, armijo_val, power_val,
                    barrier_val):

    """
    Compute primal-dual direction (Nocedal & Wright 19.12)

    Parameters
    ----------
    objective_function
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

    gradient = gradient_analytic(objective_function, weights, slacks, lagrange_multipliers, barrier_val)

    hessian_approx = hessian_approximation(objective_function, weights, slacks, lagrange_multipliers)

    hessian_regularized, diagonal_shift_val = \
        regularize_hessian(hessian_approx, num_weights, num_equality_constraints, num_inequality_constraints,
                           diagonal_shift_val, init_diagonal_shift_val, armijo_val, power_val, barrier_val)
    # calculate search_direction
    search_direction = jnp.array(jit(jnp.linalg.solve)(hessian_regularized, -gradient.reshape((gradient.size, 1)))
                                 .reshape((gradient.size,)))  # reshape gradient and result

    if num_inequality_constraints or num_equality_constraints:
        # change sign definition for the multipliers' search direction
        search_direction = search_direction.at[num_weights + num_inequality_constraints:].set(
            -search_direction[num_weights + num_inequality_constraints:])

    return search_direction
