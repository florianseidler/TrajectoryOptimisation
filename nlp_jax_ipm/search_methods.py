import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit
from merit_function import merit_function, sum_equality_values, sum_inequality_values
from gradient_of_merit_function import gradient_merit_function
from step import step


def decrease_infeasibility(weights, num_weights, equality_constraints, inequality_constraints,
                           num_equality_constraints, num_inequality_constraints, correction_new,
                           cost_function, weights_0, slacks_0, alpha_smax, search_direction_weights,
                           search_direction_slacks, merit_function_parameter, barrier_val, correction,
                           merit_function_result, armijo_val, gradient_merit_function_result,
                           weight_precision_tolerance, verbosity, backtracking_line_search_parameter):

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

    jacobian_matrix_of_constraints = \
        jacobian_of_constraints(weights, num_weights, equality_constraints, inequality_constraints,
                                num_equality_constraints, num_inequality_constraints).T
    try:
        feasibility_restoration_direction = \
            -jnp.linalg.solve(jacobian_matrix_of_constraints, correction_new
                              .reshape((num_weights + num_inequality_constraints, 1))) \
                .reshape((num_weights + num_inequality_constraints,))

    except:
        # if the Jacobian is not invertible, find the minimum norm solution instead
        feasibility_restoration_direction = \
            - jnp.linalg.lstsq(jacobian_matrix_of_constraints, correction_new, rcond=None)[0]
    if num_inequality_constraints:
        initial_merit_val = \
            merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights +
                           feasibility_restoration_direction[:num_weights], slacks_0 + alpha_smax *
                           search_direction_slacks + feasibility_restoration_direction[num_weights:],
                           merit_function_parameter, barrier_val, equality_constraints, inequality_constraints,
                           num_equality_constraints, num_inequality_constraints)
    else:
        initial_merit_val = \
            merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights +
                           feasibility_restoration_direction, slacks_0, merit_function_parameter,
                           barrier_val, equality_constraints, inequality_constraints,
                           num_equality_constraints, num_inequality_constraints)

    if initial_merit_val <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
        if num_inequality_constraints:
            alpha_corr = step(slacks_0, alpha_smax * search_direction_slacks
                              + feasibility_restoration_direction[num_weights:],
                              weight_precision_tolerance, backtracking_line_search_parameter)
            merit_val = \
                merit_function(cost_function, weights_0 + alpha_corr *
                               (alpha_smax * search_direction_weights + feasibility_restoration_direction[:num_weights]),
                               slacks_0 + alpha_corr *
                               (alpha_smax * search_direction_slacks + feasibility_restoration_direction[num_weights:]),
                               merit_function_parameter, barrier_val, equality_constraints,
                               inequality_constraints, num_equality_constraints, num_inequality_constraints)

            if merit_val <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
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





