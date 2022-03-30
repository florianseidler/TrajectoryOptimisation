import jax.numpy as jnp
from merit_function import merit_function
from step import step
from jacobian_calculation import jacobian_of_constraints


def decrease_infeasibility(weights, num_weights, equality_constraints,
                           inequality_constraints, num_equality_constraints,
                           num_inequality_constraints, correction_new,
                           cost_function, weights_0, slacks_0, alpha_smax,
                           search_direction_weights, search_direction_slacks,
                           merit_function_parameter, barrier_val, correction,
                           merit_function_result, armijo_val,
                           gradient_merit_function_result,
                           weight_precision_tolerance, verbosity,
                           backtracking_line_search_parameter):

    """
    Try to decrease infeasibility.
    Parameters
    ----------
    weights
    num_weights
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints
    correction_new
    cost_function
    weights_0
    slacks_0
    alpha_smax
    search_direction_weights
    search_direction_slacks
    merit_function_parameter
    barrier_val
    correction
    merit_function_result
    armijo_val
    gradient_merit_function_result
    weight_precision_tolerance
    verbosity
    backtracking_line_search_parameter
    Returns
    -------
    feasibility_restoration_direction, alpha_corr, correction
    """
    alpha_corr = 0.0

    jacobian_matrix_of_constraints = (
        jacobian_of_constraints(weights, num_weights, equality_constraints,
                                inequality_constraints,
                                num_equality_constraints,
                                num_inequality_constraints).T)
    try:
        correction_new = correction_new.reshape((
            num_weights + num_inequality_constraints, 1))
        feasibility_restoration_direction = - jnp.linalg.solve(
            jacobian_matrix_of_constraints, correction_new)
        feasibility_restoration_direction = (
            feasibility_restoration_direction.reshape((
                num_weights + num_inequality_constraints,)))

    except:
        # if Jacobian is not invertible, find the minimum norm solution instead
        feasibility_restoration_direction = (
            - jnp.linalg.lstsq(jacobian_matrix_of_constraints, correction_new,
                               rcond=None)[0])
    if num_inequality_constraints:
        merit_weights = (weights_0 + alpha_smax * search_direction_weights
                         + feasibility_restoration_direction[:num_weights])
        merit_slacks = (slacks_0 + alpha_smax * search_direction_slacks
                        + feasibility_restoration_direction[num_weights:])
        initial_merit_val = (
            merit_function(cost_function, merit_weights, merit_slacks,
                           merit_function_parameter, barrier_val,
                           equality_constraints, inequality_constraints,
                           num_equality_constraints,
                           num_inequality_constraints))
    else:
        merit_weights = (weights_0 + alpha_smax * search_direction_weights
                         + feasibility_restoration_direction)
        initial_merit_val = (
            merit_function(cost_function, merit_weights, slacks_0,
                           merit_function_parameter, barrier_val,
                           equality_constraints, inequality_constraints,
                           num_equality_constraints,
                           num_inequality_constraints))

    merit_comparison = (merit_function_result + alpha_smax * armijo_val
                        * gradient_merit_function_result)
    if initial_merit_val <= merit_comparison:
        if num_inequality_constraints:
            slacks_corr = (alpha_smax * search_direction_slacks
                           + feasibility_restoration_direction[num_weights:])
            alpha_corr = step(slacks_0, slacks_corr,
                              weight_precision_tolerance,
                              backtracking_line_search_parameter)
            weights_alpha = (weights_0 + alpha_corr *
                             (alpha_smax * search_direction_weights
                              + feasibility_restoration_direction[:num_weights])
                             )
            slacks_alpha = (slacks_0 + alpha_corr * slacks_corr)
            merit_val = (
                merit_function(cost_function, weights_alpha,
                               slacks_alpha, merit_function_parameter,
                               barrier_val, equality_constraints,
                               inequality_constraints, num_equality_constraints,
                               num_inequality_constraints))

            merit_comparison = (merit_function_result + alpha_smax * armijo_val
                                * gradient_merit_function_result)
            if merit_val <= merit_comparison:
                if verbosity > 2:
                    print('Second-order feasibility correction accepted')
                # correction accepted
                correction = True
        else:
            # correction accepted
            if verbosity > 2:
                print('Second-order feasibility correction accepted')
            alpha_corr = jnp.float32(1.0)
            correction = True

    return feasibility_restoration_direction, alpha_corr, correction


def update_alphas(weights_0, slacks_0, alpha_smax,
                  alpha_lmax, search_direction_weights, search_direction_slacks,
                  equality_constraints, inequality_constraints,
                  num_equality_constraints, num_inequality_constraints,
                  merit_function_parameter, backtracking_line_search_parameter,
                  cost_function, barrier_val, merit_function_result,
                  armijo_val, gradient_merit_function_result,
                  optimization_return_signal, minimal_step, verbosity):
    # infeasibility has not increased, no correction necessary
    alpha_smax *= backtracking_line_search_parameter
    alpha_lmax *= backtracking_line_search_parameter

    if num_inequality_constraints:
        slacks_ = slacks_0 + alpha_smax * search_direction_slacks
    else:
        slacks_ = slacks_0
    while ((merit_function(cost_function,
                           (weights_0 + alpha_smax * search_direction_weights),
                           slacks_, merit_function_parameter, barrier_val,
                           equality_constraints, inequality_constraints,
                           num_equality_constraints, num_inequality_constraints)
            ) > (merit_function_result + alpha_smax
                 * armijo_val * gradient_merit_function_result)
           and optimization_return_signal != -2):
        # backtracking line search
        if num_inequality_constraints:
            search_step = (jnp.sqrt(jnp.linalg.norm
                                    (alpha_smax * search_direction_weights) ** 2
                           + jnp.linalg.norm(alpha_lmax
                                             * search_direction_slacks) ** 2))
        else:
            search_step = jnp.linalg.norm(alpha_smax * search_direction_weights)
        if search_step < minimal_step:
            # search direction is unreliable to machine precision,
            # stop solver
            if verbosity > 2:
                print('Search direction is unreliable to '
                      'machine precision.')
            optimization_return_signal = -2

        alpha_smax *= backtracking_line_search_parameter
        alpha_lmax *= backtracking_line_search_parameter

    return alpha_smax, alpha_lmax, optimization_return_signal
