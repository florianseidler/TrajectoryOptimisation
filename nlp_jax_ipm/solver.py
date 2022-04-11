import jax.numpy as jnp
from jax import jit
from initialization import initialization
from main_calculation import main_calculation

"""
Main function of the nonlinear problem solver. Interior Point Method
algorithm has been used in combination with JAX for automatic differentiation.
Based on pyipm.py by Joel Kaardal. Usage can be seen in test.py.
"""


def solve(cost_function, equality_constraints, inequality_constraints,
          input_weights=None, input_slacks=None,
          input_lagrange_multipliers=None, num_inner_iterations=20,
          num_outer_iterations=10, kkt_tolerance=1.0E-4, approximate_hessian=0,
          verbosity=1):
    """
    Outer loop for adjusting the barrier parameter.
    Inner loop for finding a feasible minimum using backtracking line search.

    Parameters
    ----------
    cost_function: function that should be minimized.
    equality_constraints: Constraints bounded to an equality (i.e. x + y = 2).
    inequality_constraints: Constraints bounded to an inequality (i.e. x >= 2)
    input_weights: input weights for faster convergence (optional).
    input_slacks: input slacks for faster convergence (optional).
    input_lagrange_multipliers: input lagrange multipliers for faster
                                convergence (optional).
    num_inner_iterations: Number of iterations through inner loop.
    num_outer_iterations: Number of iterations through outer loop.
    kkt_tolerance: To check if the kkt-condtitions are satisfied.
    approximate_hessian: To use an approximated hessian instead of the real one.
    verbosity: For info while executing the program.


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
    (backtracking_line_search_parameter, armijo_val,
     weight_precision_tolerance, convergence_tolerance_cost_function, power_val,
     init_diagonal_shift_val, minimal_step, diagonal_shift_val,
     update_factor_merit_function_parameter,
     merit_function_parameter, barrier_val,
     merit_function_initialization_parameter, objective_function, objective_function_with_barrier,
     weights, slacks, lagrange_multipliers, num_weights,
     num_inequality_constraints, num_equality_constraints, kkt_weights,
     kkt_slacks, kkt_equality_lagrange_multipliers,
     kkt_inequality_lagrange_multipliers, old_cost_function,
     convergence_tolerance_kkt_converged,
     convergence_tolerance_cost_function_converged,
     optimization_return_signal) = (
        initialization(cost_function, equality_constraints,
                       inequality_constraints, input_weights, input_slacks,
                       input_lagrange_multipliers, kkt_tolerance))

    ''' MAIN CALCULATIONS '''

    # calculation: Nocedal & Wright Algorithm 19.2
    for outer_iteration_variable in range(num_outer_iterations):
        # adjusting barrier parameter
        # if current point converged to kkt_tolerance -> solution found
        kkt_norm = jit(jnp.linalg.norm)
        if all([kkt_norm(kkt_weights) <= kkt_tolerance,
                kkt_norm(kkt_slacks) <= kkt_tolerance,
                kkt_norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                kkt_norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]
               ):
            optimization_return_signal = 1
            convergence_tolerance_kkt_converged = True
            break

        for inner_iteration_variable in range(num_inner_iterations):
            # check convergence to barrier tolerance using KKT conditions
            # if True break from inner loop
            barrier_tolerance = jit(jnp.max)(jnp.array(
                [kkt_tolerance, barrier_val]))
            # check_convergence() TODO (nice to have)
            if all([kkt_norm(kkt_weights) <= barrier_tolerance,
                    kkt_norm(kkt_slacks) <= barrier_tolerance,
                    kkt_norm(kkt_equality_lagrange_multipliers)
                    <= barrier_tolerance,
                    kkt_norm(kkt_inequality_lagrange_multipliers)
                    <= barrier_tolerance]):
                if (not num_equality_constraints and
                        not num_inequality_constraints):
                    optimization_return_signal = 1
                    convergence_tolerance_kkt_converged = True
                break

            (weights, slacks, lagrange_multipliers, barrier_val, kkt_weights,
             kkt_slacks, kkt_equality_lagrange_multipliers,
             kkt_inequality_lagrange_multipliers
             ) = (main_calculation(outer_iteration_variable,
                                   inner_iteration_variable, objective_function,
                                   objective_function_with_barrier,
                                   cost_function, equality_constraints,
                                   inequality_constraints, weights, slacks,
                                   lagrange_multipliers, num_weights,
                                   num_equality_constraints,
                                   num_inequality_constraints,
                                   diagonal_shift_val, init_diagonal_shift_val,
                                   armijo_val, verbosity, power_val,
                                   barrier_val, merit_function_parameter,
                                   merit_function_initialization_parameter,
                                   update_factor_merit_function_parameter,
                                   backtracking_line_search_parameter,
                                   weight_precision_tolerance, minimal_step,
                                   approximate_hessian))

            if all([convergence_tolerance_cost_function is not None,
                    not num_inequality_constraints,
                    optimization_return_signal != -2]):
                # for unconstrained and equality constraints only,
                # calculate new cost and check convergence tolerance
                new_cost_function = cost_function(weights)
                if jit(jnp.abs)(old_cost_function - new_cost_function) <= jit(
                        jnp.abs)(convergence_tolerance_cost_function):
                    # converged to convergence_tolerance_cost_function precision
                    optimization_return_signal = 2
                    convergence_tolerance_cost_function_converged = True
                    break
                else:
                    # did not converge, update past cost
                    old_cost_function = new_cost_function

            if optimization_return_signal == -2:  # a bad search direction
                break

            if inner_iteration_variable >= num_inner_iterations - 1:
                if verbosity > 0 and num_inequality_constraints:
                    print('MAXIMUM INNER ITERATIONS EXCEEDED')

        if all([convergence_tolerance_cost_function is not None,
                num_inequality_constraints,
                optimization_return_signal != -2]):
            # if inequality
            # calculate new cost and check convergence_tolerance_cost_function
            new_cost_function = cost_function(weights)
            if jit(jnp.abs)(old_cost_function - new_cost_function) <= jit(
                    jnp.abs)(convergence_tolerance_cost_function):
                # converged to convergence_tolerance_cost_function precision
                optimization_return_signal = 2
                convergence_tolerance_cost_function_converged = True
            else:
                # did not converge, update past cost
                old_cost_function = new_cost_function

        if (convergence_tolerance_cost_function is not None and
                convergence_tolerance_cost_function_converged):
            # if convergence reached, break because solution has been found
            break

        if optimization_return_signal == -2:
            # a bad search direction was chosen, terminating
            break

        """        
        if num_outer_iterations >= num_inner_iterations - 1:
            optimization_return_signal = -1
            if verbosity > 0:
                if num_inequality_constraints:
                    print('MAXIMUM OUTER ITERATIONS EXCEEDED')
                else:
                    print('MAXIMUM ITERATIONS EXCEEDED')
            break
        """
        if num_inequality_constraints:
            # update the barrier parameter, calculation: Nocedal & Wright 19.20
            update_value = (num_inequality_constraints
                            * jit(jnp.min)(slacks * lagrange_multipliers[
                                           num_equality_constraints:])
                            / (jit(jnp.dot)(slacks, lagrange_multipliers[
                                            num_equality_constraints:])
                               + minimal_step))
            # calculation: Nocedal & Wright 19.20
            barrier_val = (
                0.1 * jit(jnp.min)(jnp.array([
                    0.05 * (1.0 - update_value)
                    / (update_value + minimal_step), 2.0])) ** 3
                * jit(jnp.dot)(slacks,
                               lagrange_multipliers[num_equality_constraints:])
                / num_inequality_constraints)
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
        elif all([kkt_norm(kkt_weights) <= kkt_tolerance,
                  kkt_norm(kkt_slacks) <= kkt_tolerance,
                  kkt_norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                  kkt_norm(kkt_inequality_lagrange_multipliers)
                  <= kkt_tolerance]):
            msg.append('Converged to KKT tolerance')
        elif (convergence_tolerance_cost_function is not None and
              convergence_tolerance_cost_function_converged):
            msg.append('Converged to '
                       'convergence_tolerance_cost_function tolerance')
        else:
            msg.append('Maximum iterations reached')
            num_outer_iterations = num_outer_iterations
            num_inner_iterations = 0

    # return solution weights, slacks, multipliers, cost, and KKT conditions
    return (weights, slacks, lagrange_multipliers, function_values, kkt_weights,
            kkt_slacks, kkt_equality_lagrange_multipliers,
            kkt_inequality_lagrange_multipliers)
