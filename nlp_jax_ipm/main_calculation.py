import jax.numpy as jnp
from jax import grad, jit
from KKT_calculations import KKT
from search_direction_calculation import calc_search_dir
from concatenate_constraints import concatenate_constraints
from search import backtracking_line_search
from step import step


def main_calculation(outer_iteration_variable, inner_iteration_variable,
                     objective_function, objective_function_with_barrier,
                     cost_function, equality_constraints,
                     inequality_constraints, weights, slacks,
                     lagrange_multipliers, num_weights,
                     num_equality_constraints, num_inequality_constraints,
                     diagonal_shift_val, init_diagonal_shift_val, armijo_val,
                     verbosity, power_val, barrier_val,
                     merit_function_parameter,
                     merit_function_initialization_parameter,
                     update_factor_merit_function_parameter,
                     backtracking_line_search_parameter,
                     weight_precision_tolerance, minimal_step,
                     approximate_hessian):

    """
    Computes search direction, updates merit function parameter and uses
    backtracking line search to find a better minimum and calculates the
    updated kkt-conditions.

    Parameters
    ----------
    outer_iteration_variable
    inner_iteration_variable
    objective_function
    objective_function_with_barrier
    cost_function
    equality_constraints
    inequality_constraints
    weights
    slacks
    lagrange_multipliers
    num_weights
    num_equality_constraints
    num_inequality_constraints
    diagonal_shift_val
    init_diagonal_shift_val
    armijo_val
    verbosity
    power_val
    barrier_val
    merit_function_parameter
    merit_function_initialization_parameter
    update_factor_merit_function_parameter
    backtracking_line_search_parameter
    weight_precision_tolerance
    minimal_step
    approximate_hessian

    Returns
    -------
    weights
    slacks
    lagrange_multipliers
    barrier_val
    kkt_weights
    kkt_slacks
    kkt_equality_lagrange_multipliers
    kkt_inequality_lagrange_multipliers
    """

    print(outer_iteration_variable + 1, '. Outer iteration     ',
          inner_iteration_variable + 1, '. Inner iteration')

    # compute primal-dual direction
    search_direction = (
        calc_search_dir(objective_function, objective_function_with_barrier,
                        weights, slacks, lagrange_multipliers, num_weights,
                        num_equality_constraints, num_inequality_constraints,
                        diagonal_shift_val, init_diagonal_shift_val, armijo_val,
                        power_val, barrier_val, approximate_hessian))

    # update the merit function parameter, if necessary
    if num_inequality_constraints or num_equality_constraints:
        if num_inequality_constraints:
            barrier_gradient = (
                jnp.concatenate([jit(grad(cost_function))(weights),
                                 - barrier_val / (slacks + minimal_step)]))

        else:
            barrier_gradient = jit(grad(cost_function))(weights)

        direction_gradient = (
            jit(jnp.dot)(
                barrier_gradient,
                search_direction[:num_weights + num_inequality_constraints]))

        sum_weights_slacks = (
            jit(jnp.sum)(jnp.abs(
                concatenate_constraints(weights, slacks, equality_constraints,
                                        inequality_constraints,
                                        num_equality_constraints,
                                        num_inequality_constraints))))

        merit_threshold = (direction_gradient /
                           (1 - update_factor_merit_function_parameter)
                           / sum_weights_slacks)

        if merit_function_initialization_parameter < merit_threshold:
            merit_function_initialization_parameter = jit(jnp.float32)(
                merit_threshold)
            merit_function_parameter = merit_function_initialization_parameter

    if num_inequality_constraints:
        # use fraction-to-the-boundary rule to make sure slacks
        # and multipliers do not decrease too quickly
        alpha_smax = (
            step(slacks,
                 search_direction[num_weights:(num_weights
                                               + num_inequality_constraints)],
                 weight_precision_tolerance,
                 backtracking_line_search_parameter))

        alpha_lmax = (
            step(lagrange_multipliers[num_equality_constraints:],
                 search_direction[(num_weights + num_inequality_constraints
                                   + num_equality_constraints):],
                 weight_precision_tolerance,
                 backtracking_line_search_parameter))

        # use a backtracking line search to update weights,
        # slacks, and multipliers
        weights, slacks, lagrange_multipliers = (
            backtracking_line_search(
                cost_function, weights, slacks, lagrange_multipliers,
                search_direction, jnp.float32(alpha_smax),
                jnp.float32(alpha_lmax), num_weights, barrier_val,
                merit_function_parameter, minimal_step,
                weight_precision_tolerance, equality_constraints,
                inequality_constraints, num_equality_constraints,
                num_inequality_constraints, verbosity,
                backtracking_line_search_parameter, armijo_val))

    else:
        # use a backtracking line search to update weights,
        # slacks, and multipliers
        weights, slacks, lagrange_multipliers = (
            backtracking_line_search(
                cost_function, weights, slacks, lagrange_multipliers,
                search_direction, jnp.float32(1.0), jnp.float32(1.0),
                num_weights, barrier_val, merit_function_parameter,
                minimal_step, weight_precision_tolerance, equality_constraints,
                inequality_constraints, num_equality_constraints,
                num_inequality_constraints, verbosity,
                backtracking_line_search_parameter, armijo_val))

    # calculate the updated KKT conditions
    (kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers,
     kkt_inequality_lagrange_multipliers) = (
        KKT(objective_function_with_barrier, weights, slacks,
            lagrange_multipliers, num_inequality_constraints,
            num_equality_constraints, barrier_val))

    return (weights, slacks, lagrange_multipliers, barrier_val, kkt_weights,
            kkt_slacks, kkt_equality_lagrange_multipliers,
            kkt_inequality_lagrange_multipliers)
