import numpy as np
import jax.numpy as jnp
from jax import grad, hessian

def solver(inputweights, inputslacks, inputmultipliers):
    # validate()
    # compile()
    # init weights, slacks, multipliers, diag shift coeff(?)
    # KKT(weights, slacks, multipliers)
    # init function_tolerance for convergence
    # init optimisation_return_signal

    for outer_iteration_variable in 12: # adjusting barrier parameter
        # if current point converged to kkt_tolerance -> solution found
        for inner_iteration_variable in 11:
            print("delete me")
            # if convergence <= mu_tolerance (?) -> kkt_tolerance converged -> break
            # calc gradient, hessian
            # regularize_hessian() to maintain matrix inertia
            # calculate search_direction
            # change sign definition for multipliers search direction
            # update merit function parameter
            # use fraction-to-boundary rule -> slacks and multipliers using backtracing line search
            # search() ( step() )
            # KKT(weights, slacks, multipliers)
            # calc new cost for eq constraints and check function_tolerance convergence
            # calc new cost for Ineq constraints and check function_tolerance convergence
            # cost()
            # break if function_tolerance convergence reached
            # update barrier parameter

    weights, slacks, multipliers, cost, kkt_conditions = 0  # overwrite
    return weights, slacks, multipliers, cost, kkt_conditions

# TODOS: start with inner for loop!
# regularize_hessian(), search_direction, merit function,
# fraction-to-boundary/backtracing, step(), search(), calc new cost (?), cost()

# hessian() from jax


def gradient(function_to_derive, number_input_arg=1):
    gradient_list = []
    for input_arg_iteration in np.arange(number_input_arg):
        gradient_list.append(grad(function_to_derive, input_arg_iteration))
    return gradient_list


def gradient_values(function_to_minimize, weights, slacks, lagrange_multipliers):
    # function_to_minimize input: weights[0], ..., weights[-1], slacks[0], ..., slacks[-1], lagrange_multipliers[0], ...
    number_input_arg = np.size(weights) + np.size(slacks) + np.size(lagrange_multipliers) # get number of variables
    derivatives_of_function = gradient(function_to_minimize, number_input_arg)  # derive function to every variable
    results = []                                                                # prepare empty list
    for iterate_variables in np.arange(number_input_arg):                       # put variables in every derivative
        results.append(derivatives_of_function[iterate_variables](weights, slacks, lagrange_multipliers))
    return np.array(results)


def KKT(function_to_minimize, weights, slacks, lagrange_multipliers, number_inequality_constraints=0, number_equality_constraints=0):
    """Calculate the first-order Karush-Kuhn-Tucker conditions. Irrelevant conditions are set to zero."""
    # kkt_weights is the gradient of the Lagrangian with respect to x (weights)
    # kkt_slacks is the gradient of the Lagrangian with respect to s (slack variables)
    # kkt_equality_lagrange_multipliers is the gradient of the Lagrangian with respect to lda[:self.neq] (equality
    # constraints Lagrange multipliers)
    # kkt_inequality_lagrange_multipliers is the gradient of the Lagrangian with respect to lda[self.neq:] (inequality
    # constraints Lagrange multipliers)

    # function_to_minimize = function + lagrange_multipliers * equality_constraints + slacks * lagrange_multipliers * inequality_constraints
    kkt_results = gradient_values(function_to_minimize, weights, slacks, lagrange_multipliers)
    number_variables = np.size(weights)
    number_slacks = np.size(slacks)
    number_equality_lagrange_multipliers = np.size(lagrange_multipliers)

    if number_inequality_constraints and number_equality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = kkt_results[number_variables:(number_variables + number_slacks)] * slacks
        kkt_equality_lagrange_multipliers = kkt_results[(number_variables + number_slacks):(number_variables + number_slacks + number_equality_lagrange_multipliers)]
        kkt_inequality_lagrange_multipliers = kkt_results[(number_variables + number_slacks + number_equality_lagrange_multipliers):]
    elif number_equality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = np.float64(0.0)
        kkt_equality_lagrange_multipliers = kkt_results[(number_variables + number_slacks):(number_variables + number_slacks + number_equality_lagrange_multipliers)]
        kkt_inequality_lagrange_multipliers = np.float64(0.0)
    elif number_inequality_constraints:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = kkt_results[number_variables:(number_variables + number_slacks)] * slacks
        kkt_equality_lagrange_multipliers = np.float64(0.0)
        kkt_inequality_lagrange_multipliers = kkt_results[(number_variables + number_slacks + number_equality_lagrange_multipliers):]
    else:
        kkt_weights = kkt_results[:number_variables]
        kkt_slacks = np.float64(0.0)
        kkt_equality_lagrange_multipliers = np.float64(0.0)
        kkt_inequality_lagrange_multipliers = np.float64(0.0)

    return kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers


def regularize_hessian(hessian_matrix, number_weights, number_equality_constraints=0, number_inequality_constraints=0, diagonal_shift_val=0.0, init_diagonal_shift_val=0.5, armijo_val = 1.0E-4, power_val = 0.4, barrier_val=0.2):
    """Regularize the Hessian to avoid ill-conditioning and to escape saddle points."""
    eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
    condition_number = np.min(np.abs(eigenvalues)) / np.max(np.abs(eigenvalues))
    minimal_step = np.finfo(np.float64).eps
    regularization_val = np.float64(np.sqrt(np.finfo(np.float64).eps))

    if condition_number <= minimal_step or (number_equality_constraints + number_inequality_constraints) != np.sum(eigenvalues < -minimal_step):
        if condition_number <= minimal_step and number_equality_constraints:
            lower_index = number_weights + number_inequality_constraints
            upper_index = lower_index + number_equality_constraints
            hessian_matrix[lower_index:upper_index, lower_index:upper_index] -= regularization_val * armijo_val * (barrier_val ** power_val) * np.eye(number_equality_constraints)
        if diagonal_shift_val == 0.0:                               # diagonal shift coefficient must not be zero
            diagonal_shift_val = init_diagonal_shift_val
        else:                                                       # diagonal shift coefficient must not be too small
            diagonal_shift_val = np.max([diagonal_shift_val / 2, init_diagonal_shift_val])

        # regularize Hessian with diagonal shift matrix (delta*I) until matrix inertia condition is satisfied
        hessian_matrix[:number_weights, :number_weights] += diagonal_shift_val * np.eye(number_weights)
        eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
        while (number_equality_constraints + number_inequality_constraints) != np.sum(eigenvalues < -minimal_step):
            Hc[:number_weights, :number_weights] -= self.delta * np.eye(number_weights)
            diagonal_shift_val *= 10.0
            Hc[:number_weights, :number_weights] += diagonal_shift_val * np.eye(number_weights)
            eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)

    return hessian_matrix


# calculate the search direction
gradient = gradient_values(function_to_minimize, weights, slacks, lagrange_multipliers)
# was macht sym_solve_cmp bzw theanp.function()?
dz = self.sym_solve_cmp(hessian_matrix, gradient.reshape((gradient.size, 1))).reshape((gradient.size,))
self.sym_solve_cmp = theano.function(inputs=[self.M_dev, self.b_dev], outputs=lin_soln,)