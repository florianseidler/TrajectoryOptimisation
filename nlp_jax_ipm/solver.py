import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit
from initialization import initialization
from main_calculation import main_calculation

""" Main function of the nonlinear problem solver. Interior Point Method algorithm has been used in combination with
JAX for automatic differentiation. Based on pyipm.py by Joel Kaardal. Usage can be seen in test.py."""


def solve(cost_function, equality_constraints, inequality_constraints, input_weights=None, input_slacks=None,
          input_lagrange_multipliers=None):
    """
    Outer loop for adjusting the barrier parameter.
    Inner loop for finding a feasible minimum using backtracking line search.

    Parameters
    ----------
    cost_function: function that should be minimized.
    equality_constraints: Constraints bounded to an equality (i.e. x + y = 2).
    inequality_constraints: Constraints bounded to an inequality (i.e. x + y >= 2)
    input_weights: input weights for faster convergence (optional).
    input_slacks: input slacks for faster convergence (optional).
    input_lagrange_multipliers: input lagrange multipliers for faster convergence (optional).

    Returns
    -------
    weights
    slacks
    lagrange multipliers
    kkt_weights
    kkt_slacks
    kkt_equality_lagrange_multipliers
    kkt_inequality_lagrange_multipliers
    """

    ''' INITIALIZATION '''
    verbosity, backtracking_line_search_parameter, armijo_val, weight_precision_tolerance, \
        convergence_tolerance_cost_function, power_val, init_diagonal_shift_val, minimal_step, diagonal_shift_val, \
        kkt_tolerance, update_factor_merit_function_parameter, merit_function_parameter, barrier_val, \
        merit_function_initialization_parameter, num_inner_iterations, num_outer_iterations, objective_function, \
        weights, slacks, lagrange_multipliers, num_weights, num_inequality_constraints, num_equality_constraints, \
        kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers, \
        old_cost_function, convergence_tolerance_kkt_converged, convergence_tolerance_cost_function_converged, \
        optimization_return_signal = \
        initialization(cost_function, equality_constraints, inequality_constraints, input_weights, input_slacks,
                       input_lagrange_multipliers)

    ''' MAIN CALCULATIONS '''

    # calculation: Nocedal & Wright Algorithm 19.2
    for outer_iteration_variable in range(num_outer_iterations):  # adjusting barrier parameter
        # if current point converged to kkt_tolerance -> solution found
        kkt_norm = jit(jnp.linalg.norm)
        if all([kkt_norm(kkt_weights) <= kkt_tolerance, kkt_norm(kkt_slacks) <= kkt_tolerance,
                kkt_norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                kkt_norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            optimization_return_signal = 1
            convergence_tolerance_kkt_converged = True
            break

        for inner_iteration_variable in range(num_inner_iterations):
            # check convergence to barrier tolerance precision using the KKT conditions; if True break from inner loop
            barrier_tolerance = jit(jnp.max)(jnp.array([kkt_tolerance, barrier_val]))
            if all([kkt_norm(kkt_weights) <= barrier_tolerance, kkt_norm(kkt_slacks) <= barrier_tolerance,
                    kkt_norm(kkt_equality_lagrange_multipliers) <= barrier_tolerance,
                    kkt_norm(kkt_inequality_lagrange_multipliers) <= barrier_tolerance]):
                if not num_equality_constraints and not num_inequality_constraints:
                    optimization_return_signal = 1
                    convergence_tolerance_kkt_converged = True
                break

            """
            print(outer_iteration_variable+1, '. Outer iteration     ', inner_iteration_variable+1, '. Inner iteration')

            # compute primal-dual direction
            search_direction = \
                calc_search_dir(objective_function, weights, slacks, lagrange_multipliers, num_weights,
                                num_equality_constraints, num_inequality_constraints, diagonal_shift_val,
                                init_diagonal_shift_val, armijo_val, power_val, barrier_val)

            if num_inequality_constraints or num_equality_constraints:
                # update the merit function parameter, if necessary
                if num_inequality_constraints:
                    barrier_gradient = \
                        jnp.concatenate([grad(cost_function)(weights), -barrier_val / (slacks + minimal_step)])
                else:
                    barrier_gradient = jit(grad(cost_function))(weights)
                direction_gradient = \
                    jit(jnp.dot)(barrier_gradient, search_direction[:num_weights + num_inequality_constraints])
                sum_weights_slacks = \
                    jnp.sum(jnp.abs(concatenate_constraints(weights, slacks, equality_constraints,
                                                            inequality_constraints, num_equality_constraints,
                                                            num_inequality_constraints)))

                merit_threshold = direction_gradient / (1 - update_factor_merit_function_parameter) / sum_weights_slacks

                if merit_function_initialization_parameter < merit_threshold:
                    merit_function_initialization_parameter = jit(jnp.float32)(merit_threshold)
                    merit_function_parameter = merit_function_initialization_parameter

            if num_inequality_constraints:
                # use fraction-to-the-boundary rule to make sure slacks and multipliers do not decrease too quickly
                alpha_smax = \
                    step(slacks, search_direction[num_weights:(num_weights + num_inequality_constraints)],
                         weight_precision_tolerance, backtracking_line_search_parameter)
                alpha_lmax = \
                    step(lagrange_multipliers[num_equality_constraints:],
                         search_direction[(num_weights + num_inequality_constraints + num_equality_constraints):],
                         weight_precision_tolerance, backtracking_line_search_parameter)
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers = \
                    backtracking_line_search(cost_function, weights, slacks, lagrange_multipliers, search_direction,
                                             jnp.float32(alpha_smax), jnp.float32(alpha_lmax), num_weights, barrier_val,
                                             merit_function_parameter, equality_constraints, inequality_constraints,
                                             num_equality_constraints, num_inequality_constraints, verbosity,
                                             backtracking_line_search_parameter, armijo_val)
            else:
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers = \
                    backtracking_line_search(cost_function, weights, slacks, lagrange_multipliers, search_direction,
                                             jnp.float32(1.0), jnp.float32(1.0), num_weights, barrier_val,
                                             merit_function_parameter, equality_constraints, inequality_constraints,
                                             num_equality_constraints, num_inequality_constraints, verbosity,
                                             backtracking_line_search_parameter, armijo_val)

            # calculate the updated KKT conditions
            kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = \
                KKT(objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints,
                    num_equality_constraints, barrier_val)
            """

            weights, slacks, lagrange_multipliers, barrier_val, kkt_weights, kkt_slacks, \
                kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers\
                = main_calculation(outer_iteration_variable, inner_iteration_variable, objective_function,
                                   cost_function, equality_constraints, inequality_constraints, weights, slacks,
                                   lagrange_multipliers, num_weights, num_equality_constraints,
                                   num_inequality_constraints, diagonal_shift_val, init_diagonal_shift_val, armijo_val,
                                   verbosity, power_val, barrier_val, merit_function_parameter,
                                   merit_function_initialization_parameter, update_factor_merit_function_parameter,
                                   backtracking_line_search_parameter, weight_precision_tolerance, minimal_step)

            if all([convergence_tolerance_cost_function is not None, not num_inequality_constraints,
                    optimization_return_signal != -2]):
                # for unconstrained and equality constraints only,
                # calculate new cost and check convergence_tolerance_cost_function
                new_cost_function = cost_function(weights)
                if jit(jnp.abs)(old_cost_function - new_cost_function) <= jit(jnp.abs)(
                        convergence_tolerance_cost_function):
                    # converged to convergence_tolerance_cost_function precision
                    optimization_return_signal = 2
                    convergence_tolerance_cost_function_converged = True
                    break
                else:
                    # did not converge, update past cost
                    old_cost_function = new_cost_function

            if optimization_return_signal == -2:  # a bad search direction was chosen, terminating
                break

            if inner_iteration_variable >= num_inner_iterations - 1:
                if verbosity > 0 and num_inequality_constraints:
                    print('MAXIMUM INNER ITERATIONS EXCEEDED')

        if all([convergence_tolerance_cost_function is not None, num_inequality_constraints,
                optimization_return_signal != -2]):
            # when problem has inequality constraints, calculate new cost and check convergence_tolerance_cost_function
            new_cost_function = cost_function(weights)
            if jit(jnp.abs)(old_cost_function - new_cost_function) <= jit(jnp.abs)(
                    convergence_tolerance_cost_function):
                # converged to convergence_tolerance_cost_function precision
                optimization_return_signal = 2
                convergence_tolerance_cost_function_converged = True
            else:
                # did not converge, update past cost
                old_cost_function = new_cost_function

        if convergence_tolerance_cost_function is not None and convergence_tolerance_cost_function_converged:
            # if convergence_tolerance_cost_function convergence reached, break because solution has been found
            break

        if optimization_return_signal == -2:
            # a bad search direction was chosen, terminating
            break

        if num_outer_iterations >= num_inner_iterations - 1:
            optimization_return_signal = -1
            if verbosity > 0:
                if num_inequality_constraints:
                    print('MAXIMUM OUTER ITERATIONS EXCEEDED')
                else:
                    print('MAXIMUM ITERATIONS EXCEEDED')
            break

        if num_inequality_constraints:
            # update the barrier parameter, calculation: Nocedal & Wright 19.20
            update_value = num_inequality_constraints \
                           * jit(jnp.min)(slacks * lagrange_multipliers[num_equality_constraints:]) \
                           / (jit(jnp.dot)(slacks, lagrange_multipliers[num_equality_constraints:]) + minimal_step)
            # calculation: Nocedal & Wright 19.20
            barrier_val = \
                (0.1 * jit(jnp.min)(jnp.array([0.05 * (1.0 - update_value) / (update_value + minimal_step), 2.0])) ** 3
                 * jit(jnp.dot)(slacks, lagrange_multipliers[num_equality_constraints:]) / num_inequality_constraints)
            if jit(jnp.float32)(barrier_val) < jit(jnp.float32)(0.0):
                barrier_val = 0.0

    ''' END OF CALCULATIONS '''

    function_values = cost_function(weights)
    kkt_norm = jit(jnp.linalg.norm)
    if verbosity >= 0:
        msg = []
        if optimization_return_signal == -2:
            msg.append(
                'Terminated due to bad direction in backtracking line search')
        elif all([kkt_norm(kkt_weights) <= kkt_tolerance, kkt_norm(kkt_slacks) <= kkt_tolerance,
                  kkt_norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                  kkt_norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            msg.append('Converged to KKT tolerance')
        elif convergence_tolerance_cost_function is not None and convergence_tolerance_cost_function_converged:
            msg.append('Converged to convergence_tolerance_cost_function tolerance')
        else:
            msg.append('Maximum iterations reached')
            num_outer_iterations = num_outer_iterations
            num_inner_iterations = 0

    # return solution weights, slacks, multipliers, cost, and KKT conditions
    return weights, slacks, lagrange_multipliers, function_values, kkt_weights, kkt_slacks, \
        kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers
