import jax.numpy as jnp
from .merit_function import merit_function
from .gradient_of_merit_function import gradient_merit_function
from .search_methods import decrease_infeasibility, update_alphas
from .concatenate_constraints import concatenate_constraints


def backtracking_line_search(cost_function, weights_0, slacks_0,
                             lagrange_multipliers_0, search_direction,
                             alpha_smax, alpha_lmax, num_weights, barrier_val,
                             merit_function_parameter, minimal_step,
                             weight_precision_tolerance,
                             equality_constraints=None,
                             inequality_constraints=None,
                             num_equality_constraints=0,
                             num_inequality_constraints=0, verbosity=1,
                             backtracking_line_search_parameter=0.995,
                             armijo_val=1.0E-4):
    """
    Backtracking line search to find a solution that leads to a smaller value
    of the Lagrangian within the confines of the maximum step length for
    slack variables and Lagrange multipliers found using class function 'step'.

    Parameters
    ----------
    cost_function
    weights_0
    slacks_0
    lagrange_multipliers_0
    search_direction
    alpha_smax
    alpha_lmax
    num_weights
    barrier_val
    merit_function_parameter
    minimal_step
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints
    verbosity
    backtracking_line_search_parameter
    armijo_val
    weight_precision_tolerance

    Returns
    -------
    weights
    slacks
    lagrange_multipliers
    """

    # extract search directions along weights, slacks, and multipliers
    search_direction_weights = search_direction[:num_weights]
    if num_inequality_constraints:
        search_direction_slacks = (
            search_direction[num_weights:(num_weights
                                          + num_inequality_constraints)])
    else:
        search_direction_slacks = jnp.array([])
    if num_equality_constraints or num_inequality_constraints:
        search_direction_lagrange_multipliers = (
            search_direction[(num_weights + num_inequality_constraints):])
    else:
        search_direction_lagrange_multipliers = jnp.float32(0.0)
        alpha_lmax = jnp.float32(0.0)

    (optimization_return_signal, weights, slacks, merit_function_result,
     gradient_merit_function_result, correction) = (
        setup_search(cost_function, weights_0, slacks_0,
                     merit_function_parameter, barrier_val,
                     equality_constraints, inequality_constraints, num_weights,
                     num_equality_constraints, num_inequality_constraints,
                     search_direction, minimal_step))

    if num_inequality_constraints:
        # step search when there are inequality constraints
        new_merit_val = (
            merit_function(cost_function, weights_0 +
                           alpha_smax * search_direction_weights,
                           slacks_0 + alpha_smax * search_direction_slacks,
                           merit_function_parameter, barrier_val,
                           equality_constraints, inequality_constraints,
                           num_equality_constraints,
                           num_inequality_constraints))
        old_merit_val = (merit_function_result + alpha_smax
                         * armijo_val * gradient_merit_function_result)

        if new_merit_val > old_merit_val:
            # second-order correction
            correction_old = concatenate_constraints(
                weights_0, slacks_0, equality_constraints,
                inequality_constraints, num_equality_constraints,
                num_inequality_constraints)
            correction_new = concatenate_constraints(
                weights_0 + alpha_smax * search_direction_weights,
                slacks_0 + alpha_smax * search_direction_slacks,
                equality_constraints, inequality_constraints,
                num_equality_constraints, num_inequality_constraints)
            if jnp.sum(jnp.abs(correction_new)) > jnp.sum(jnp.abs(
                    correction_old)):
                # infeasibility has increased, attempt to correct
                feasibility_restoration_direction, alpha_corr, correction = (
                    decrease_infeasibility(weights, num_weights,
                                           equality_constraints,
                                           inequality_constraints,
                                           num_equality_constraints,
                                           num_inequality_constraints,
                                           correction_new, cost_function,
                                           weights_0, slacks_0, alpha_smax,
                                           search_direction_weights,
                                           search_direction_slacks,
                                           merit_function_parameter,
                                           barrier_val,
                                           correction, merit_function_result,
                                           armijo_val,
                                           gradient_merit_function_result,
                                           weight_precision_tolerance,
                                           verbosity,
                                           backtracking_line_search_parameter))
            if not correction:
                (alpha_smax, alpha_lmax,
                 optimization_return_signal) = update_alphas(
                    weights_0, slacks_0, alpha_smax, alpha_lmax,
                    search_direction_weights, search_direction_slacks,
                    equality_constraints, inequality_constraints,
                    num_equality_constraints, num_inequality_constraints,
                    merit_function_parameter,
                    backtracking_line_search_parameter, cost_function,
                    barrier_val, merit_function_result, armijo_val,
                    gradient_merit_function_result, optimization_return_signal,
                    minimal_step, verbosity)

                if optimization_return_signal == -2:
                    return weights_0, slacks_0, lagrange_multipliers_0
        # update slack variables
        if correction:
            slacks = (slacks_0 + alpha_corr * (
                      alpha_smax * search_direction_slacks
                      + feasibility_restoration_direction[num_weights:]))
        else:
            slacks = slacks_0 + alpha_smax * search_direction_slacks

    else:
        # step search for only equality constraints or unconstrained problems
        # CHANGE: merit_comp_val = - merit_comp_val
        merit_comp_val = -merit_function(
            cost_function,
            weights_0 + alpha_smax * search_direction_weights,
            slacks_0, merit_function_parameter, barrier_val,
            equality_constraints, inequality_constraints,
            num_equality_constraints, num_inequality_constraints)

        # CHANGE: precision_val = - precision_val
        precision_val = -(merit_function_result + alpha_smax
                          * armijo_val * gradient_merit_function_result)

        if merit_comp_val > precision_val:
            if num_equality_constraints:
                # second-order correction
                correction_old = concatenate_constraints(
                    weights_0, slacks_0, equality_constraints,
                    inequality_constraints, num_equality_constraints,
                    num_inequality_constraints)
                correction_new = concatenate_constraints(
                    weights_0 + alpha_smax * search_direction_weights, slacks_0,
                    equality_constraints, inequality_constraints,
                    num_equality_constraints, num_inequality_constraints)
                if jnp.sum(jnp.abs(correction_new)) > jnp.sum(jnp.abs(
                        correction_old)):
                    (feasibility_restoration_direction, alpha_corr,
                     correction) = (
                        decrease_infeasibility(
                            weights, num_weights, equality_constraints,
                            inequality_constraints,
                            num_equality_constraints,
                            num_inequality_constraints,
                            correction_new, cost_function,
                            weights_0, slacks_0, alpha_smax,
                            search_direction_weights,
                            search_direction_slacks,
                            merit_function_parameter,
                            barrier_val, correction,
                            merit_function_result,
                            armijo_val,
                            gradient_merit_function_result,
                            weight_precision_tolerance,
                            verbosity,
                            backtracking_line_search_parameter))

            if not correction:
                (alpha_smax, alpha_lmax,
                 optimization_return_signal) = update_alphas(
                    weights_0, slacks_0, alpha_smax, alpha_lmax,
                    search_direction_weights, search_direction_slacks,
                    equality_constraints, inequality_constraints,
                    num_equality_constraints, num_inequality_constraints,
                    merit_function_parameter,
                    backtracking_line_search_parameter, cost_function,
                    barrier_val, merit_function_result, armijo_val,
                    gradient_merit_function_result, optimization_return_signal,
                    minimal_step, verbosity)

                if optimization_return_signal == -2:
                    return weights_0, slacks_0, lagrange_multipliers_0

    # update weights
    if correction:
        weights = (weights_0 + alpha_corr * (
                   alpha_smax * search_direction_weights +
                   feasibility_restoration_direction[:num_weights]))
    else:
        weights = weights_0 + alpha_smax * search_direction_weights

    # update multipliers (if applicable)
    if num_equality_constraints or num_inequality_constraints:
        lagrange_multipliers = (lagrange_multipliers_0 + alpha_lmax
                                * search_direction_lagrange_multipliers)
    else:
        lagrange_multipliers = lagrange_multipliers_0
    # return updated weights, slacks, and multipliers
    return weights, slacks, lagrange_multipliers


def setup_search(cost_function, weights_0, slacks_0, merit_function_parameter,
                 barrier_val, equality_constraints, inequality_constraints,
                 num_weights, num_equality_constraints,
                 num_inequality_constraints, search_direction, minimal_step):
    """
    Extract weights, slacks, and multipliers, calculate initial merit result
    and gradient merit result and initialize optimization signal and correction.

    Parameters
    ----------
    cost_function
    weights_0
    slacks_0
    merit_function_parameter
    barrier_val
    equality_constraints
    inequality_constraints
    num_weights
    num_equality_constraints
    num_inequality_constraints
    search_direction
    minimal_step

    Returns
    -------
    optimization_return_signal
    weights
    slacks
    merit_function_result
    gradient_merit_function_result
    correction
    """

    weights = weights_0  # no copy
    slacks = slacks_0
    merit_function_result = (
        merit_function(cost_function, weights_0, slacks_0,
                       merit_function_parameter, barrier_val,
                       equality_constraints, inequality_constraints,
                       num_equality_constraints, num_inequality_constraints))

    gradient_merit_function_result = (
        gradient_merit_function(cost_function, weights_0, slacks_0,
                                search_direction[:num_weights
                                                 + num_inequality_constraints],
                                num_weights, barrier_val, minimal_step,
                                equality_constraints, inequality_constraints,
                                num_equality_constraints,
                                num_inequality_constraints,
                                merit_function_parameter))
    optimization_return_signal = 0
    correction = False

    return (optimization_return_signal, weights, slacks, merit_function_result,
            gradient_merit_function_result, correction)
