import numpy as np
import jax.numpy as jnp
from jax import grad, hessian

########################################################
# TODOS:
# test functions
# cost function
# hessian()
# gradient
# #validate()
# #compile()


# tunable constants
verbosity = 1  # needed?
backtracking_line_search_parameter = 0.995
armijo_val = 1.0E-4
weight_precision_tolerance = 0
convergence_tolerance_objective_function = 0.5  # ?
number_inner_iterations = 20
number_outer_iterations = 10
minimal_step = np.finfo(np.float64).eps
barrier_initialization_parameter = 0.2
diagonal_shift_val = 0.0
kkt_tolerance = 0.1  # ?
merit_function_initialization_parameter = np.float64(0.2)  # ?


def solver(equality_constraints, inequality_constraints, weights, slacks=None, lagrange_multipliers=None):
    number_weights = np.size(weights)
    number_equality_constraints = np.size(equality_constraints)
    number_inequality_constraints = np.size(inequality_constraints)

    if (number_weights == 0):
        print("Weights should not be empty")
        return 0

    if number_equality_constraints or number_inequality_constraints:
        if input_lagrange_multipliers is None:
            lagrange_multipliers = init_lagrange_multipliers(weights, number_weights, number_equality_constraints,
                                                             number_inequality_constraints, equality_constraints,
                                                             inequality_constraints)
            if number_inequality_constraints and number_equality_constraints:
                lagrange_multipliers_inequality = lagrange_multipliers[number_equality_constraints:]
                lagrange_multipliers_inequality[lagrange_multipliers_inequality < np.float64(0.0)] = np.float64(
                    kkt_tolerance)
                lagrange_multipliers[number_equality_constraints:] = lagrange_multipliers_inequality
            elif number_inequality_constraints:
                lagrange_multipliers[lagrange_multipliers < np.float64(0.0)] = np.float64(kkt_tolerance)
        else:
            lagrange_multipliers = lagrange_multipliers.astype(np.float64)
    else:
        lagrange_multipliers = np.array([], dtype=np.float64)
        # return print("No optimization necessary without constraints")

    # calculate the initial KKT conditions
    kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = \
        KKT(objective_function, weights, slacks, lagrange_multipliers, number_inequality_constraints, number_equality_constraints)

    # if convergence_tolerance_objective_function is set, calculate the prior cost
    if convergence_tolerance_objective_function is not None:
        old_objective_function = objective_function(weights)

    # init convergences
    convergence_tolerance_KKT_converged = False
    convergence_tolerance_objective_function_converged = False

    # init barrier val
    barrier_val = np.float64(barrier_initialization_parameter)

    # init optimization return signal
    optimization_return_signal = 0

    # calculation: Nocedal & Wright Algorithm 19.2
    for outer_iteration_variable in number_outer_iterations:  # adjusting barrier parameter

        # if current point converged to kkt_tolerance -> solution found
        if all([np.linalg.norm(kkt_weights) <= kkt_tolerance, np.linalg.norm(kkt_slacks) <= kkt_tolerance,
                np.linalg.norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                np.linalg.norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            optimization_return_signal = 1
            convergence_tolerance_KKT_converged = True
            break

        for inner_iteration_variable in number_inner_iterations:
            # check convergence to barrier tolerance precision using the KKT conditions; if True, break from the inner loop
            barrier_tolerance = np.max([kkt_tolerance, barrier_val])
            if all([np.linalg.norm(kkt_weights) <= barrier_tolerance, np.linalg.norm(kkt_slacks) <= barrier_tolerance,
                    np.linalg.norm(kkt_equality_lagrange_multipliers) <= barrier_tolerance,
                    np.linalg.norm(kkt_inequality_lagrange_multipliers) <= barrier_tolerance]):
                if not number_equality_constraints and not number_inequality_constraints:
                    opzimization_return_signal = 1
                    convergence_tolerance_KKT_converged = True
                break

            # compute primal-dual direction (Nocedal & Wright 19.12)
            gradient = gradient_values(objective_function, weights, slacks, lagrange_multipliers)
            # regularize hessian to maintain matrix inertia (Nocedal & Wright 19.25)
            hessian = regularize_hessian(hessian(objective_function)(weights, slacks, lagrange_multipliers),
                        number_weights, number_equality_constraints, number_inequality_constraints,
                        diagonal_shift_val, init_diagonal_shift_val, armijo_val, power_val, barrier_val)
            # TODO: nachprüfen, ob das so klappt
            # calculate search_direction
            search_direction = jnp.linalg.solve(hessian_matrix, gradient.reshape((gradient.size, 1))) \
                .reshape((gradient.size,))  # reshape gradient and result

            if number_inequality_constraints or number_equality_constraints:
                # change sign definition for the multipliers' search direction
                search_direction[number_weights + number_inequality_constraints:] = \
                    -search_direction[number_weights + number_inequality_constraints:]

            if number_inequality_constraints or number_equality_constraints:
                # update the merit function parameter, if necessary
                merit_threshold = np.dot(
                    barrier_cost_gradient(objective_function, weights, slacks, number_inequality_constraints),
                    search_direction[:number_weights + number_inequality_constraints]) / (
                                              1 - update_factor_merit_function_parameter) / np.sum(np.abs(
                    concatenate_constraints(weights, slacks, equality_constraints, inequality_constraints,
                                            number_equality_constraints, number_inequality_constraints)))
                if merit_function_initialization_parameter < merit_threshold:
                    merit_function_initialization_parameter = np.float64(merit_threshold)
                    merit_function_parameter = merit_function_initialization_parameter

            if number_inequality_constraints:
                # use fraction-to-the-boundary rule to make sure slacks and multipliers do not decrease too quickly
                alpha_smax = step(slacks,
                                  search_direction[number_weights:(number_weights + number_inequality_constraints)],
                                  weight_precision_tolerance, backtracking_line_search_parameter)
                alpha_lmax = step(lagrange_multipliers[number_equality_constraints:], search_direction[(
                                  number_weights + number_inequality_constraints + number_equality_constraints):],
                                  weight_precision_tolerance, backtracking_line_search_parameter)
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers, opzimization_return_signal = search(weights, slacks,
                                                                                           lagrange_multipliers,
                                                                                           search_direction,
                                                                                           np.float64(alpha_smax),
                                                                                           np.float64(alpha_lmax),
                                                                                           number_weights,
                                                                                           equality_constraints,
                                                                                           inequality_constraints,
                                                                                           number_equality_constraints,
                                                                                           number_inequality_constraints,
                                                                                           verbosity,
                                                                                           backtracking_line_search_parameter,
                                                                                           armijo_val)
            else:
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers, opzimization_return_signal = search(weights, slacks,
                                                                                           lagrange_multipliers,
                                                                                           search_direction,
                                                                                           np.float64(1.0),
                                                                                           np.float64(1.0),
                                                                                           number_weights,
                                                                                           equality_constraints,
                                                                                           inequality_constraints,
                                                                                           number_equality_constraints,
                                                                                           number_inequality_constraints,
                                                                                           verbosity,
                                                                                           backtracking_line_search_parameter,
                                                                                           armijo_val)

            kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = KKT(
                objective_function, weights, slacks, lagrange_multipliers, number_inequality_constraints,
                number_equality_constraints)  # calculate the updated KKT conditions

            if all([convergence_tolerance_objective_function is not None, not number_inequality_constraints,
                    opzimization_return_signal != -2]):
                # for unconstrained and equality constraints only, calculate new cost and check convergence_tolerance_objective_function
                new_objective_function = objective_function(weights)
                if np.abs(old_objective_function - new_objective_function) <= np.abs(
                        convergence_tolerance_objective_function):
                    # converged to convergence_tolerance_objective_function precision
                    opzimization_return_signal = 2
                    convergence_tolerance_objective_function_converged = True
                    break
                else:
                    # did not converge, update past cost
                    old_objective_function = new_objective_function

            if opzimization_return_signal == -2:  # a bad search direction was chosen, terminating
                break

            if inner >= number_inner_iterations - 1:
                if verbosity > 0 and number_inequality_constraints:
                    print('MAXIMUM INNER ITERATIONS EXCEEDED')

        if all([convergence_tolerance_objective_function is not None, number_inequality_constraints,
                opzimization_return_signal != -2]):
            # when problem has inequality constraints, calculate new cost and check convergence_tolerance_objective_function convergence
            new_objective_function = objective_function(weights)
            if np.abs(old_objective_function - new_objective_function) <= np.abs(
                    convergence_tolerance_objective_function):
                # converged to convergence_tolerance_objective_function precision
                opzimization_return_signal = 2
                convergence_tolerance_objective_function_converged = True
            else:
                # did not converge, update past cost
                old_objective_function = new_objective_function

        if convergence_tolerance_objective_function is not None and convergence_tolerance_objective_function_converged:
            # if convergence_tolerance_objective_function convergence reached, break because solution has been found
            break

        if opzimization_return_signal == -2:
            # a bad search direction was chosen, terminating
            break

        if outer_iterations_number >= number_outer_iterations - 1:
            opzimization_return_signal = -1
            if verbosity > 0:
                if number_inequality_constraints:
                    print('MAXIMUM OUTER ITERATIONS EXCEEDED')
                else:
                    print('MAXIMUM ITERATIONS EXCEEDED')
            break

        if number_inequality_constraints:
            # update the barrier parameter, calculation: Nocedal & Wright 19.20
            update_value = number_inequality_constraints * np.min(
                slacks * lagrange_multipliers[number_equality_constraints:]) / (np.dot(slacks, lagrange_multipliers[
                                                                         number_equality_constraints:]) + minimal_step)
            # calculation: Nocedal & Wright 19.20
            barrier_val = (
                        0.1 * np.min([0.05 * (1.0 - update_value) / (update_value + minimal_step), 2.0]) ** 3 * np.dot(
                    slacks, lagrange_multipliers[number_equality_constraints:]) / number_inequality_constraints)
            if np.float64(barrier_val) < np.float64(0.0):
                barrier_val = 0.0
            barrier_val = np.float64(barrier_val)

    function_values = objective_function(weights)

    if verbosity >= 0:  # nötig?
        msg = []
        if opzimization_return_signal == -2:
            msg.append(
                'Terminated due to bad direction in backtracking line search')
        elif all([np.linalg.norm(kkt_weights) <= kkt_tolerance, np.linalg.norm(kkt_slacks) <= kkt_tolerance,
                  np.linalg.norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                  np.linalg.norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            msg.append('Converged to KKT tolerance')
        elif convergence_tolerance_objective_function is not None and convergence_tolerance_objective_function_converged:
            msg.append('Converged to convergence_tolerance_objective_function tolerance')
        else:
            msg.append('Maximum iterations reached')
            outer_iteration_number = number_outer_iterations
            inner = 0

    # return solution weights, slacks, multipliers, cost, and KKT conditions
    return weights, slacks, lagrange_multipliers, function_values, kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers


def init_lagrange_multipliers(weights, number_weights, number_equality_constraints=0, number_inequality_constraints=0,
                              equality_constraints=None, inequality_constraints=None):
    # TODO: eliminate comments
    # if number_equality_constraints or number_inequality_constraints:
    return np.dot(np.linalg.pinv(
        jacobian_objective_function(weights, number_weights, equality_constraints, inequality_constraints,
                                    number_equality_constraints, number_inequality_constraints)[:number_weights, :]),
                  grad(objective_function, 0)(weights).reshape((number_weights, 1))).reshape(
        (number_equality_constraints + number_inequality_constraints,))


# else: return T.dot(pinv(jacobian_objective_function[:number_weights, :]), df.reshape((number_weights, 1))).reshape((number_equality_constraints + number_inequality_constraints,))
# else really needed?


def gradient(function_to_derive, number_input_arg=1):
    gradient_list = []
    for input_arg_iteration in np.arange(number_input_arg):
        gradient_list.append(grad(function_to_derive, input_arg_iteration))
    return gradient_list


def gradient_values(objective_function, weights, slacks, lagrange_multipliers):
    # objective_function input: weights[0], ..., weights[-1], slacks[0], ..., slacks[-1], lagrange_multipliers[0], ...
    number_input_arg = np.size(weights) + np.size(slacks) + np.size(lagrange_multipliers)  # get number of variables
    derivatives_of_function = gradient(objective_function, number_input_arg)  # derive function to every variable
    results = []  # prepare empty list
    for iterate_variables in np.arange(number_input_arg):  # put variables in every derivative
        results.append(derivatives_of_function[iterate_variables](weights, slacks, lagrange_multipliers))
    return np.array(results)


def KKT(objective_function, weights, slacks, lagrange_multipliers, number_inequality_constraints,
        number_equality_constraints):
    """Calculate the first-order Karush-Kuhn-Tucker conditions. Irrelevant conditions are set to zero."""
    # kkt_weights is the gradient of the Lagrangian with respect to x (weights)
    # kkt_slacks is the gradient of the Lagrangian with respect to s (slack variables)
    # kkt_equality_lagrange_multipliers is the gradient of the Lagrangian with respect to lagrange_multipliers[:number_equality_constraints] (equality
    # constraints Lagrange multipliers)
    # kkt_inequality_lagrange_multipliers is the gradient of the Lagrangian with respect to lagrange_multipliers[number_equality_constraints:] (inequality
    # constraints Lagrange multipliers)

    # objective_function = function + lagrange_multipliers * equality_constraints + slacks * lagrange_multipliers * inequality_constraints
    kkt_results = gradient_values(objective_function, weights, slacks, lagrange_multipliers)
    number_variables = np.size(weights)
    number_slacks = np.size(slacks)
    number_equality_lagrange_multipliers = np.size(lagrange_multipliers)

    if number_inequality_constraints and number_equality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = kkt_results[number_variables:(number_variables + number_slacks)] * slacks
        kkt_equality_lagrange_multipliers = kkt_results[(number_variables + number_slacks):(
                    number_variables + number_slacks + number_equality_lagrange_multipliers)]
        kkt_inequality_lagrange_multipliers = kkt_results[(
                                        number_variables + number_slacks + number_equality_lagrange_multipliers):]
    elif number_equality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = np.float64(0.0)
        kkt_equality_lagrange_multipliers = kkt_results[(number_variables + number_slacks):(
                    number_variables + number_slacks + number_equality_lagrange_multipliers)]
        kkt_inequality_lagrange_multipliers = np.float64(0.0)
    elif number_inequality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = kkt_results[number_variables:(number_variables + number_slacks)] * slacks
        kkt_equality_lagrange_multipliers = np.float64(0.0)
        kkt_inequality_lagrange_multipliers = kkt_results[(
                                       number_variables + number_slacks + number_equality_lagrange_multipliers):]
    else:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = np.float64(0.0)
        kkt_equality_lagrange_multipliers = np.float64(0.0)
        kkt_inequality_lagrange_multipliers = np.float64(0.0)

    return kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers


def regularize_hessian(hessian_matrix, number_weights, number_equality_constraints=0, number_inequality_constraints=0,
                       diagonal_shift_val=0.0, init_diagonal_shift_val=0.5, armijo_val=1.0E-4, power_val=0.4,
                       barrier_val=0.2):
    """Regularize the Hessian to avoid ill-conditioning and to escape saddle points."""
    # Nocedal & Wright 19.25 / Appendix B1, further info: Wächter & Biegler
    # constants are arbitrary but typical choices
    eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
    condition_number = np.min(np.abs(eigenvalues)) / np.max(np.abs(eigenvalues))

    if condition_number <= minimal_step or (number_equality_constraints + number_inequality_constraints) != np.sum(
            eigenvalues < -minimal_step):
        if condition_number <= minimal_step and number_equality_constraints:
            lower_index = number_weights + number_inequality_constraints
            upper_index = lower_index + number_equality_constraints
            regularization_val = np.float64(np.sqrt(minimal_step))
            hessian_matrix[lower_index:upper_index, lower_index:upper_index] -= regularization_val * armijo_val * (
                        barrier_val ** power_val) * np.eye(number_equality_constraints)
        if diagonal_shift_val == 0.0:  # diagonal shift coefficient must not be zero
            diagonal_shift_val = init_diagonal_shift_val
        else:  # diagonal shift coefficient must not be too small
            diagonal_shift_val = np.max([diagonal_shift_val / 2, init_diagonal_shift_val])

        # regularize Hessian with diagonal shift matrix (delta*I) until matrix inertia condition is satisfied
        hessian_matrix[:number_weights, :number_weights] += diagonal_shift_val * np.eye(number_weights)
        eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
        while (number_equality_constraints + number_inequality_constraints) != np.sum(eigenvalues < -minimal_step):
            Hc[:number_weights, :number_weights] -= diagonal_shift_val * np.eye(number_weights)
            diagonal_shift_val *= 10.0
            Hc[:number_weights, :number_weights] += diagonal_shift_val * np.eye(number_weights)
            eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)

    return hessian_matrix


def concatenate_constraints(weights, slacks, equality_constraints=None, inequality_constraints=None,
                            number_equality_constraints=0, number_inequality_constraints=0):
    if number_equality_constraints and number_inequality_constraints:
        return np.concatenate([equality_constraints(weights).reshape((number_equality_constraints,)),
                               (inequality_constraints(weights) - slacks).reshape((number_inequality_constraints,))],
                              axis=0)
    elif number_equality_constraints:
        return equality_constraints(weights).reshape((number_equality_constraints,))
    elif number_inequality_constraints:
        return (inequality_constraints(weights) - slacks).reshape((number_inequality_constraints,))
    return zeros((number_equality_constraints + number_inequality_constraints,))


def step(values, derivative_of_values, weight_precision_tolerance=0, backtracking_line_search_parameter=0.995):
    """Golden section search used to determine the maximum step length for slack variables and Lagrange multipliers
       using the fraction-to-the-boundary rule."""
    GOLD_constant = (np.sqrt(5.0) + 1.0) / 2.0
    weighted_values = (1.0 - backtracking_line_search_parameter) * values  # default: 0.05*values
    small_return = 0.0
    big_return = 1.0

    if np.all(values + big_return * derivative_of_values >= weighted_values):
        return big_return
    else:
        decreasing_variable = big_return - (big_return - small_return) / GOLD_constant
        increasing_variable = small_return + (big_return - small_return) / GOLD_constant
        while np.abs(big_return - small_return) > GOLD_constant * weight_precision_tolerance:
            if np.any(values + increasing_variable * derivative_of_values < weighted_values):
                big_return = np.copy(increasing_variable)
            else:
                small_return = np.copy(increasing_variable)
            if decreasing_variable > small_return:
                if np.any(values + decreasing_variable * derivative_of_values < weighted_values):
                    big_return = np.copy(decreasing_variable)
                else:
                    small_return = np.copy(decreasing_variable)

            decreasing_variable = big_return - (big_return - small_return) / GOLD_constant
            increasing_variable = small_return + (big_return - small_return) / GOLD_constant

        return small_return


def barrier_function(slacks, barrier_value):
    return barrier_value * np.sum(np.log(slacks))


def merit_function(objective_function, weights, slacks, merit_function_parameter, equality_constraints=None,
                   inequality_constraints=None, number_equality_constraints=0, number_inequality_constraints=0):
    if number_equality_constraints and number_inequality_constraints:
        return (objective_function(weights) + merit_function_parameter * (
                    np.sum(np.abs(equality_constraints(weights))) + np.sum(
                np.abs(inequality_constraints(weights) - slacks))) - barrier_function(slacks, barrier_value))
    elif number_equality_constraints:
        return objective_function(weights) + merit_function_parameter * np.sum(np.abs(equality_constraints(weights)))
    elif number_inequality_constraints:
        return objective_function(weights) + merit_function_parameter * np.sum(
            np.abs(inequality_constraints(weights) - slacks)) - barrier_function(slacks, barrier_value)
    else:
        return objective_function(weights)


def jacobian_objective_function(weights, number_weights, equality_constraints=None, inequality_constraints=None,
                                number_equality_constraints=0, number_inequality_constraints=0):
    # all reshapes necessary? TODO
    if number_equality_constraints and number_inequality_constraints:
        # TODO: jacfwd einseitig?
        jacobian_top = np.concatenate([ \
            jax.jacfwd(equality_constraints)(weights) \
                .reshape(number_equality_constraints,number_weights) \
                .reshape(number_weights, number_equality_constraints) \
                .reshape((number_weights, number_equality_constraints)), \
            jax.jacfwd(inequality_constraints)(weights) \
                .reshape(number_inequality_constraints, number_weights) \
                .reshape(number_weights,number_inequality_constraints) \
                .reshape((number_weights, number_inequality_constraints))], axis=1)
        jacobian_bottom = np.concatenate([np.zeros((number_inequality_constraints, number_equality_constraints)),
                                          -np.eye(number_inequality_constraints)], axis=1)
        return np.concatenate([jacobian_top, jacobian_bottom], axis=0)
    elif number_equality_constraints:
        return jax.jacfwd(equality_constraints)(weights) \
            .reshape(number_equality_constraints, number_weights) \
            .reshape(number_weights, number_equality_constraints) \
            .reshape((number_weights, number_equality_constraints))
    elif number_inequality_constraints:
        return np.concatenate([jax.jacfwd(inequality_constraints)(weights) \
                              .reshape(number_inequality_constraints,number_weights) \
                              .reshape(number_weights, number_inequality_constraints) \
                              .reshape((number_weights, number_inequality_constraints)), \
                            -np.eye(number_inequality_constraints)], axis=0)
    else:
        return np.zeros((number_weights + number_inequality_constraints,
                         number_equality_constraints + number_inequality_constraints))


def gradient_merit_function(objective_function, weights, slacks, search_direction, number_weights,
                            equality_constraints=None, inequality_constraints=None, number_equality_constraints=0,
                            number_inequality_constraints=0, barrier_initialization_parameter=0.2, merit_function_parameter=10.0):
    if number_equality_constraints and number_inequality_constraints:
        # derivative of objective function
        # before: gradient(objective_function(weights))
        return (np.dot(grad(objective_function)(weights), search_direction[:number_weights]) \
                - merit_function_parameter * (np.sum(np.abs(equality_constraints(weights))) \
                + np.sum(np.abs(inequality_constraints(weights) - slacks))) \
                - np.dot(barrier_initialization_parameter / (slacks + minimal_step), search_direction[number_weights:]))
    elif number_equality_constraints:  # TODO: nachprüfen, ob das so klappt
        return (np.dot(grad(objective_function)(weights), search_direction[:number_weights]) \
                - merit_function_parameter * np.sum(np.abs(equality_constraints(weights))))
    elif number_inequality_constraints:
        return (np.dot(grad(objective_function)(weights), search_direction[:number_weights]) \
                - merit_function_parameter * np.sum(np.abs(inequality_constraints(weights) - slacks)) \
                - np.dot(barrier_initialization_parameter / (slacks + minimal_step), search_direction[number_weights:]))
    else:
        return np.dot(grad(objective_function)(weights), search_direction[:number_weights])


def barrier_cost_gradient(objective_function, weights, slacks, number_inequality_constraints):
    if number_inequality_constraints:
        return np.concatenate(
            [grad(objective_function, 0)(weights), -barrier_val / (slacks + minimal_value)], axis=0)
    else:
        return grad(objective_function, 0)(weights)  # TODO: to be revisited


def search(weights_0, slacks_0, lagrange_multipliers_0, search_direction, alpha_smax, alpha_lmax, number_weights,
           equality_constraints=None, inequality_constraints=None, number_equality_constraints=0,
           number_inequality_constraints=0, verbosity=1, backtracking_line_search_parameter=0.995, armijo_val=1.0E-4):
    """Backtracking line search to find a solution that leads to a smaller value of the Lagrangian within the confines
       of the maximum step length for the slack variables and Lagrange multipliers found using class function 'step'."""
    # extract search directions along weights, slacks, and multipliers
    search_direction_weights = search_direction[:number_weights]
    if number_inequality_constraints:
        search_direction_slacks = search_direction[number_weights:(number_weights + number_inequality_constraints)]
    if number_equality_constraints or number_inequality_constraints:
        search_direction_lagrange_multipliers = search_direction[(number_weights + number_inequality_constraints):]
    else:
        search_direction_lagrange_multipliers = np.float64(0.0)
        alpha_lmax = np.float64(0.0)
    weights = np.copy(weights_0)
    slacks = np.copy(slacks_0)
    merit_function_result = merit_function(objective_function, weights_0, slacks_0, merit_function_parameter, \
        equality_constraints, inequality_constraints, number_equality_constraints, number_inequality_constraints)
    gradient_merit_function_result = gradient_merit_function(objective_function, weights_0, slacks_0, \
            search_direction[:number_weights + number_inequality_constraints], number_weights, equality_constraints, \
            inequality_constraints, number_equality_constraints, number_inequality_constraints, \
            barrier_initialization_parameter, merit_function_parameter)
    correction = False
    if number_inequality_constraints:
        # step search when there are inequality constraints
        if merit_function(objective_function, weights_0 + alpha_smax * search_direction_weights,
                slacks_0 + alpha_smax * search_direction_slacks, merit_function_parameter, equality_constraints, \
                inequality_constraints, number_equality_constraints, number_inequality_constraints) \
                > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
            # second-order correction
            correction_old = concatenate_constraints(weights_0, slacks_0, equality_constraints, inequality_constraints,
                                                     number_equality_constraints, number_inequality_constraints)
            correction_new = concatenate_constraints(weights_0 + alpha_smax * search_direction_weights,
                                                     slacks_0 + alpha_smax * search_direction_slacks,
                                                     equality_constraints, inequality_constraints,
                                                     number_equality_constraints, number_inequality_constraints)
            if np.sum(np.abs(correction_new)) > np.sum(np.abs(correction_old)):
                # infeasibility has increased, attempt to correct
                jacobian_matrix_objective_function = jacobian_objective_function(weights, number_weights,
                                                                                 equality_constraints,
                                                                                 inequality_constraints,
                                                                                 number_equality_constraints,
                                                                                 number_inequality_constraints)
                try:
                    feasibility_restoration_direction = -jnp.linalg.solve(jacobian_matrix_objective_function, \
                                         correction_new.reshape((number_weights + number_inequality_constraints, 1))) \
                                            .reshape((number_weights + number_inequality_constraints,))
                except:
                    # if the Jacobian is not invertible, find the minimum norm solution instead
                    feasibility_restoration_direction = - \
                    np.linalg.lstsq(jacobian_matrix_objective_function, correction_new, rcond=None)[0]  #jnp or np
                if (merit_function(objective_function, weights_0 + alpha_smax * search_direction_weights \
                                + feasibility_restoration_direction[:number_weights], slacks_0 \
                                + alpha_smax * search_direction_slacks + feasibility_restoration_direction[number_weights:], \
                                   merit_function_parameter, equality_constraints, inequality_constraints, \
                                   number_equality_constraints, number_inequality_constraints) \
                                <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result):
                    alpha_corr = step(slacks_0, alpha_smax * search_direction_slacks \
                                      + feasibility_restoration_direction[number_weights:], \
                                      weight_precision_tolerance, backtracking_line_search_parameter)
                    if (merit_function(objective_function, weights_0 + alpha_corr * (alpha_smax \
                            * search_direction_weights + feasibility_restoration_direction[:number_weights]), \
                            slacks_0 + alpha_corr * (alpha_smax * search_direction_slacks \
                            + feasibility_restoration_direction[number_weights:], merit_function_parameter, \
                            equality_constraints, inequality_constraints, number_equality_constraints, \
                            number_inequality_constraints)) \
                            <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result):
                        if verbosity > 2:
                            print('Second-order feasibility correction accepted')
                        # correction accepted
                        correction = True
            if not correction:
                # infeasibility has not increased, no correction necessary
                alpha_smax *= backtracking_line_search_parameter
                alpha_lmax *= backtracking_line_search_parameter
                while merit_function(objective_function, weights_0 + alpha_smax * search_direction_weights,
                        slacks_0 + alpha_smax * search_direction_slacks, merit_function_parameter, \
                        equality_constraints, inequality_constraints, number_equality_constraints, \
                        number_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                    # backtracking line search
                    if (np.sqrt(np.linalg.norm(alpha_smax * search_direction_weights) ** 2 + np.linalg.norm(
                            alpha_lmax * search_direction_slacks) ** 2) < minimal_step):
                        # search direction is unreliable to machine precision, stop solver
                        if verbosity > 2:
                            print('Search direction is unreliable to machine precision.')
                        optimization_return_signal = -2
                        return weights_0, slacks_0, lagrange_multipliers_0
                    alpha_smax *= backtracking_line_search_parameter
                    alpha_lmax *= backtracking_line_search_parameter
        # update slack variables
        if correction:
            slacks = slacks_0 + alpha_corr * (
                        alpha_smax * search_direction_slacks + feasibility_restoration_direction[number_weights:])
        else:
            slacks = slacks_0 + alpha_smax * search_direction_slacks
    else:
        # step search for only equality constraints or unconstrained problems
        if merit_function(objective_function, weights_0 + alpha_smax * search_direction_weights, slacks_0,
                        merit_function_parameter, equality_constraints, inequality_constraints, \
                        number_equality_constraints, number_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
            if number_equality_constraints:
                # second-order correction
                correction_old = concatenate_constraints(weights_0, slacks_0, equality_constraints,
                                                         inequality_constraints, number_equality_constraints,
                                                         number_inequality_constraints)
                correction_new = concatenate_constraints(weights_0 + alpha_smax * search_direction_weights, slacks_0,
                                                         equality_constraints, inequality_constraints,
                                                         number_equality_constraints, number_inequality_constraints)
                if np.sum(np.abs(correction_new)) > np.sum(np.abs(correction_old)):
                    # infeasibility has increased, attempt to correct
                    jacobian_matrix_objective_function = jacobian_objective_function(weights, number_weights,
                                                                                     equality_constraints,
                                                                                     inequality_constraints,
                                                                                     number_equality_constraints,
                                                                                     number_inequality_constraints)
                    try:
                        # calculate a feasibility restoration direction
                        feasibility_restoration_direction = -jnp.linalg.solve(jacobian_matrix_objective_function, \
                                    correction_new.reshape((number_weights, number_inequality_constraints, 1))) \
                                    .reshape((number_weights + number_inequality_constraints,))
                    except:
                        # if the Jacobian is not invertible, find the minimum norm solution instead
                        feasibility_restoration_direction = - \
                        np.linalg.lstsq(jacobian_matrix_objective_function, correction_new, rcond=None)[0]
                    if merit_function(objective_function,
                                weights_0 + alpha_smax * search_direction_weights + feasibility_restoration_direction,
                                slacks_0, merit_function_parameter, equality_constraints, inequality_constraints, \
                                number_equality_constraints, number_inequality_constraints) \
                                <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                        # correction accepted
                        if verbosity > 2:
                            print('Second-order feasibility correction accepted')
                        alpha_corr = np.float64(1.0)
                        correction = True
            if not correction:
                # infeasibility has not increased, no correction necessary
                alpha_smax *= backtracking_line_search_parameter
                alpha_lmax *= backtracking_line_search_parameter
                while merit_function(objective_function, weights_0 + alpha_smax * search_direction_weights, slacks_0,
                        merit_function_parameter, equality_constraints, inequality_constraints, \
                        number_equality_constraints, number_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                    # backtracking line search
                    if np.linalg.norm(alpha_smax * search_direction_weights) < minimal_step:
                        # search direction is unreliable to machine precision, stop solver
                        if verbosity > 2:
                            print('Search direction is unreliable to machine precision.')
                        optimization_return_signal = -2
                        return weights_0, slacks_0, lagrange_multipliers_0
                    alpha_smax *= backtracking_line_search_parameter
                    alpha_lmax *= backtracking_line_search_parameter
    # update weights
    if correction:
        weights = weights_0 + alpha_corr * (
                    alpha_smax * search_direction_weights + feasibility_restoration_direction[:number_weights])
    else:
        weights = weights_0 + alpha_smax * search_direction_weights
    # update multipliers (if applicable)
    if number_equality_constraints or number_inequality_constraints:
        lagrange_multipliers = lagrange_multipliers_0 + alpha_lmax * search_direction_lagrange_multipliers
    else:
        lagrange_multipliers = np.copy(lagrange_multipliers_0)
    # return updated weights, slacks, and multipliers
    return weights, slacks, lagrange_multipliers, optimization_return_signal
