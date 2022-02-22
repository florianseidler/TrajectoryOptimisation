import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit

########################################################
# TODOS:
# write tests
# objective_function -> barrier term
# #validate()
# compile() -> jit()

# jax.numpy less precise than numpy (jnp.sqrt less precise than np.sqrt)


'''
def cost_function(weights):
    return weights[0]**2 + 2 * weights[1]**2 + 2 * weights[0] + 8 * weights[1]


def inequality_constraints_1(weights):
    return weights[0] + 2*weights[1] - 10


def inequality_constraints_2(weights):
    return weights[0]


def inequality_constraints_3(weights):
    return weights[1]


equality_constraints = []
inequality_constraints = [inequality_constraints_1, inequality_constraints_2, inequality_constraints_3]
'''

def cost_function(weights):
    return -weights[0] - weights[1]


def equality_constr_1(weights):
    return weights[0]**2 + weights[1]**2 - 1


equality_constraints = [equality_constr_1]
inequality_constraints = []

'''
def cost_function(weights):
    return (weights[0] - 1.0) ** 2 + 2.0 * (weights[1] + 2.0) ** 2 + 3.0 *(weights[2] + 3.0) ** 2


def equality_constraint_1(weights):
    return weights[2] - weights[1] - weights[0] - 1.0


def inequality_constraint_1(weights):
    return weights[2] - weights[0] ** 2


equality_constraints = [equality_constraint_1]
inequality_constraints = [inequality_constraint_1]
'''

# tunable constants
verbosity = 1  # needed?
backtracking_line_search_parameter = 0.995
armijo_val = 1.0E-4
weight_precision_tolerance = jnp.finfo(jnp.float32).eps ### DARF NICHT 0 sein!!
convergence_tolerance_cost_function = 0.00000001
power_val = 0.4
barrier_val = 0.2
init_diagonal_shift_val=0.5
#num_inner_iterations = 20
#num_outer_iterations = 10
minimal_step = jnp.finfo(jnp.float32).eps
barrier_initialization_parameter = 0.2
diagonal_shift_val = 0.0
kkt_tolerance = 1.0E-4
#merit_function_initialization_parameter = jnp.float64(10.0)
update_factor_merit_function_parameter = 0.1
merit_function_parameter = 10.0
optimization_return_signal = 0
#num_weights = jnp.size(weights)
#num_equality_constraints = jnp.size(equality_constraints)
#num_inequality_constraints = jnp.size(inequality_constraints)



def solver(cost_function, equality_constraints, inequality_constraints, input_weights, input_slacks=None, input_lagrange_multipliers=None):
    merit_function_initialization_parameter = jnp.float64(10.0)
    num_inner_iterations = 20
    num_outer_iterations = 10
    #weights = jnp.array([])
    #slacks = jnp.array([])
    lagrange_multipliers = jnp.array([])

    if input_weights is None:
        key = jax.random.PRNGKey(1702)
        weights = random.normal(key, shape=(num_weights,)).astype(jnp.float64)
    else:
        weights = input_weights

    num_weights = jnp.size(weights)
    num_equality_constraints = jnp.size(equality_constraints)
    num_inequality_constraints = jnp.size(inequality_constraints)

    if (num_weights == 0):
        print("Weights should not be empty")
        return 0

    if num_inequality_constraints:
        if input_slacks is None:
            for iter in range(num_inequality_constraints):
                slacks[iter] = kkt_tolerance
        else:
            slacks = input_slacks
    else:
        slacks = None

    if num_equality_constraints or num_inequality_constraints:
        if input_lagrange_multipliers is None:
            lagrange_multipliers = init_lagrange_multipliers(cost_function, weights, num_weights, num_equality_constraints,
                                                             num_inequality_constraints, equality_constraints,
                                                             inequality_constraints)
            if num_inequality_constraints and num_equality_constraints:
                lagrange_multipliers_inequality = lagrange_multipliers[num_equality_constraints:]
                lagrange_multipliers_inequality[lagrange_multipliers_inequality < jnp.float64(0.0)] = jnp.float64(
                    kkt_tolerance)
                lagrange_multipliers[num_equality_constraints:] = lagrange_multipliers_inequality
            elif num_inequality_constraints:
                lagrange_multipliers[lagrange_multipliers < jnp.float64(0.0)] = jnp.float64(kkt_tolerance)
        else:
            lagrange_multipliers = input_lagrange_multipliers.astype(jnp.float64)
    else:
        lagrange_multipliers = jnp.array([], dtype=jnp.float64)
        # return print("No optimization necessary without constraints")

    # init barrier val
    if num_inequality_constraints:
        barrier_val = 0.2
    else:
        barrier_val = kkt_tolerance
        barrier_initialization_parameter = jnp.float64(barrier_val)

    # calculate the initial KKT conditions
    kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = \
        KKT(objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints, num_equality_constraints, barrier_val)

    # if convergence_tolerance_cost_function is set, calculate the prior cost
    if convergence_tolerance_cost_function is not None:
        old_cost_function = cost_function(weights)

    # init convergences
    convergence_tolerance_KKT_converged = False
    convergence_tolerance_cost_function_converged = False

    # init optimization return signal
    optimization_return_signal = 0


    print(objective_function(weights, slacks, lagrange_multipliers))

    # calculation: Nocedal & Wright Algorithm 19.2
    for outer_iteration_variable in range(num_outer_iterations):  # adjusting barrier parameter
        # if current point converged to kkt_tolerance -> solution found
        if all([jnp.linalg.norm(kkt_weights) <= kkt_tolerance, jnp.linalg.norm(kkt_slacks) <= kkt_tolerance,
                jnp.linalg.norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                jnp.linalg.norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            optimization_return_signal = 1
            convergence_tolerance_KKT_converged = True
            break

        for inner_iteration_variable in range(num_inner_iterations):
            # check convergence to barrier tolerance precision using the KKT conditions; if True, break from the inner loop
            barrier_tolerance = jnp.max(jnp.array([kkt_tolerance, barrier_val]))
            if all([jnp.linalg.norm(kkt_weights) <= barrier_tolerance, jnp.linalg.norm(kkt_slacks) <= barrier_tolerance,
                    jnp.linalg.norm(kkt_equality_lagrange_multipliers) <= barrier_tolerance,
                    jnp.linalg.norm(kkt_inequality_lagrange_multipliers) <= barrier_tolerance]):
                if not num_equality_constraints and not num_inequality_constraints:
                    optimization_return_signal = 1
                    convergence_tolerance_KKT_converged = True
                break

            print(outer_iteration_variable+1,'. Outer iteration     ', inner_iteration_variable + 1,'. Inner iteration')
            print(weights)
            print(slacks)
            print(lagrange_multipliers)
            # compute primal-dual direction (Nocedal & Wright 19.12)
            gradient = gradient_np_ary(objective_function, weights, slacks, lagrange_multipliers, barrier_val)
            # regularize hessian to maintain matrix inertia (Nocedal & Wright 19.25)
            hessian_numpy = hessian_np_ary(objective_function, weights, slacks, lagrange_multipliers)
            hessian_matrix = regularize_hessian(hessian_numpy, num_weights, num_equality_constraints, num_inequality_constraints, \
                                                diagonal_shift_val, init_diagonal_shift_val, armijo_val, power_val, barrier_val)
            # calculate search_direction
            search_direction = jnp.array(jnp.linalg.solve(hessian_matrix, -gradient.reshape((gradient.size, 1))) \
                                        .reshape((gradient.size,)))  # reshape gradient and result

            if num_inequality_constraints or num_equality_constraints:
                # change sign definition for the multipliers' search direction
                search_direction = search_direction.at[num_weights + num_inequality_constraints:].set(-search_direction[num_weights + num_inequality_constraints:])

            if num_inequality_constraints or num_equality_constraints:
                # update the merit function parameter, if necessary
                if num_inequality_constraints:
                    barrier_gradient = jnp.concatenate([grad(cost_function)(weights), -barrier_val / (slacks + minimal_step)])
                else:
                    barrier_gradient = grad(cost_function)(weights)
                direction_gradient = jnp.dot(barrier_gradient, search_direction[:num_weights + num_inequality_constraints])
                sum_weights_slacks = jnp.sum(jnp.abs(concatenate_constraints(weights, slacks, equality_constraints, inequality_constraints,
                                                                           num_equality_constraints, num_inequality_constraints)))

                merit_threshold =  direction_gradient / (1 - update_factor_merit_function_parameter) / sum_weights_slacks

                if merit_function_initialization_parameter < merit_threshold:
                    merit_function_initialization_parameter = jnp.float64(merit_threshold)
                    merit_function_parameter = merit_function_initialization_parameter

            if num_inequality_constraints:
                # use fraction-to-the-boundary rule to make sure slacks and multipliers do not decrease too quickly
                alpha_smax = step(slacks,
                                  search_direction[num_weights:(num_weights + num_inequality_constraints)],
                                  weight_precision_tolerance, backtracking_line_search_parameter)
                alpha_lmax = step(lagrange_multipliers[num_equality_constraints:],
                                  search_direction[(num_weights + num_inequality_constraints + num_equality_constraints):],
                                  weight_precision_tolerance, backtracking_line_search_parameter)
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers = search(cost_function,
                                                               weights, slacks,
                                                               lagrange_multipliers,
                                                               search_direction,
                                                               jnp.float64(alpha_smax),
                                                               jnp.float64(alpha_lmax),
                                                               num_weights,
                                                               barrier_val,
                                                               equality_constraints,
                                                               inequality_constraints,
                                                               num_equality_constraints,
                                                               num_inequality_constraints,
                                                               verbosity,
                                                               backtracking_line_search_parameter,
                                                               armijo_val)
            else:
                # use a backtracking line search to update weights, slacks, and multipliers
                weights, slacks, lagrange_multipliers = search(cost_function,
                                                               weights, slacks,
                                                               lagrange_multipliers,
                                                               search_direction,
                                                               jnp.float64(1.0),
                                                               jnp.float64(1.0),
                                                               num_weights,
                                                               barrier_val,
                                                               equality_constraints,
                                                               inequality_constraints,
                                                               num_equality_constraints,
                                                               num_inequality_constraints,
                                                               verbosity,
                                                               backtracking_line_search_parameter,
                                                               armijo_val)

            kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = KKT(
                objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints,
                num_equality_constraints, barrier_val)  # calculate the updated KKT conditions

            if all([convergence_tolerance_cost_function is not None, not num_inequality_constraints,
                    optimization_return_signal != -2]):
                # for unconstrained and equality constraints only, calculate new cost and check convergence_tolerance_cost_function
                new_cost_function = cost_function(weights)
                if jnp.abs(old_cost_function - new_cost_function) <= jnp.abs(
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
            # when problem has inequality constraints, calculate new cost and check convergence_tolerance_cost_function convergence
            new_cost_function = cost_function(weights)
            if jnp.abs(old_cost_function - new_cost_function) <= jnp.abs(
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
                           * jnp.min(slacks * lagrange_multipliers[num_equality_constraints:]) \
                           / (jnp.dot(slacks, lagrange_multipliers[num_equality_constraints:]) + minimal_step)
            # calculation: Nocedal & Wright 19.20
            barrier_val = (0.1 * jnp.min(jnp.array([0.05 * (1.0 - update_value) \
                                         / (update_value + minimal_step), 2.0])) ** 3 * jnp.dot(slacks, lagrange_multipliers[num_equality_constraints:]) \
                           / num_inequality_constraints)
            if jnp.float64(barrier_val) < jnp.float64(0.0):
                barrier_val = 0.0
            barrier_initialization_parameter = jnp.float64(barrier_val) ######TODO !!!! mu_dev != mu_host

    function_values = cost_function(weights)

    if verbosity >= 0:  # nötig?
        msg = []
        if optimization_return_signal == -2:
            msg.append(
                'Terminated due to bad direction in backtracking line search')
        elif all([jnp.linalg.norm(kkt_weights) <= kkt_tolerance, jnp.linalg.norm(kkt_slacks) <= kkt_tolerance,
                  jnp.linalg.norm(kkt_equality_lagrange_multipliers) <= kkt_tolerance,
                  jnp.linalg.norm(kkt_inequality_lagrange_multipliers) <= kkt_tolerance]):
            msg.append('Converged to KKT tolerance')
        elif convergence_tolerance_cost_function is not None and convergence_tolerance_cost_function_converged:
            msg.append('Converged to convergence_tolerance_cost_function tolerance')
        else:
            msg.append('Maximum iterations reached')
            num_outer_iterations = num_outer_iterations
            num_inner_iterations = 0

    # return solution weights, slacks, multipliers, cost, and KKT conditions
    return weights, slacks, lagrange_multipliers, function_values, kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers


def equality_constraints_fct(weights, lagrange_multipliers):
    num_equality_constraints = jnp.size(equality_constraints)
    return_sum = 0
    iter = 0
    for iter in range(num_equality_constraints):
        return_sum += equality_constraints[iter](weights) * lagrange_multipliers[iter]
    return return_sum


def inequality_constraints_fct(weights, slacks, lagrange_multipliers):
    num_inequality_constraints = jnp.size(inequality_constraints)
    num_equality_constraints = jnp.size(equality_constraints)
    iter = 0
    return_sum = 0
    for iter in range(num_inequality_constraints):
        return_sum += (inequality_constraints[iter](weights) - slacks[iter]) * lagrange_multipliers[iter + num_equality_constraints]
    return return_sum


def objective_function(weights, slacks, lagrange_multipliers):
    return cost_function(weights) - equality_constraints_fct(weights, lagrange_multipliers) - inequality_constraints_fct(weights, slacks, lagrange_multipliers)


def init_lagrange_multipliers(cost_function, weights, num_weights, num_equality_constraints=0, \
                              num_inequality_constraints=0, equality_constraints=None, inequality_constraints=None):

    jacobian_constraints = jacobian_of_constraints(weights, num_weights, equality_constraints,
                inequality_constraints, num_equality_constraints, num_inequality_constraints)[:num_weights, :]
    gradient_cost_function = grad(cost_function, 0)(weights).reshape((num_weights, 1))
    return jnp.dot(jnp.linalg.pinv(jacobian_constraints, gradient_cost_function)) \
                .reshape((num_equality_constraints + num_inequality_constraints,))


def KKT(objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints,
        num_equality_constraints, barrier_val):
    """Calculate the first-order Karush-Kuhn-Tucker conditions. Irrelevant conditions are set to zero."""
    # kkt_weights is the gradient of the Lagrangian with respect to x (weights)
    # kkt_slacks is the gradient of the Lagrangian with respect to s (slack variables)
    # kkt_equality_lagrange_multipliers is the gradient of the Lagrangian with respect to lagrange_multipliers[:num_equality_constraints] (equality
    # constraints Lagrange multipliers)
    # kkt_inequality_lagrange_multipliers is the gradient of the Lagrangian with respect to lagrange_multipliers[num_equality_constraints:] (inequality
    # constraints Lagrange multipliers)

    # objective_function = function + lagrange_multipliers * equality_constraints + slacks * lagrange_multipliers * inequality_constraints
    kkt_results = gradient_np_ary(objective_function, weights, slacks, lagrange_multipliers, barrier_val)
    num_weights = jnp.size(weights)

    if num_inequality_constraints and num_equality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = kkt_results[num_weights:(num_weights + num_inequality_constraints)] * slacks.reshape(num_inequality_constraints, 1)
        kkt_equality_lagrange_multipliers = kkt_results[(num_weights + num_inequality_constraints):(
                    num_weights + num_inequality_constraints + num_equality_constraints)]
        kkt_inequality_lagrange_multipliers = kkt_results[(
                                        num_weights + num_inequality_constraints + num_equality_constraints):]
    elif num_equality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = jnp.float64(0.0)
        kkt_equality_lagrange_multipliers = kkt_results[(num_weights + num_inequality_constraints):(
                    num_weights + num_inequality_constraints + num_equality_constraints)]
        kkt_inequality_lagrange_multipliers = jnp.float64(0.0)
    elif num_inequality_constraints:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = kkt_results[num_weights:(num_weights + num_inequality_constraints)] * slacks
        kkt_equality_lagrange_multipliers = jnp.float64(0.0)
        kkt_inequality_lagrange_multipliers = kkt_results[(
                                       num_weights + num_inequality_constraints + num_equality_constraints):]
    else:
        kkt_weights = kkt_results[:num_weights]
        kkt_slacks = jnp.float64(0.0)
        kkt_equality_lagrange_multipliers = jnp.float64(0.0)
        kkt_inequality_lagrange_multipliers = jnp.float64(0.0)

    return kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers


def gradient_np_ary(objective_function, weights, slacks, lagrange_multipliers, barrier_val):
    # TODO: check, ASK
    # calculate gradient and convert from tensor to numpy-array
    gradient_jnp = (grad(objective_function, [0, 1, 2])(weights, slacks, lagrange_multipliers))
    gradient_weights = jnp.asarray(gradient_jnp[0])
    if slacks == None:
        gradient_slacks = jnp.array([])
    else:
        gradient_slacks = jnp.asarray(gradient_jnp[1])
        num_slacks = jnp.size(slacks)
        iter = 0
        for iter in range(num_slacks):
            gradient_slacks = gradient_slacks.at[iter].add(-(barrier_val / slacks[iter])) # barrier parameter initialization
    gradient_lagrange_multipliers = jnp.asarray(gradient_jnp[2])
    gradient_np = jnp.concatenate([gradient_weights, gradient_slacks, -gradient_lagrange_multipliers], axis=0)
    return gradient_np


def hessian_np_ary(obj_fct, weights, slacks, lagrange_multipliers):
    # TODO: check, ASK
    hessian_jnp = (hessian(obj_fct, [0, 1, 2])(weights, slacks, lagrange_multipliers))
    upper_left = jnp.asarray(hessian_jnp[0][0])
    upper_right = jnp.asarray(hessian_jnp[0][2])
    lower_left = jnp.asarray(hessian_jnp[2][0])
    lower_right = jnp.asarray(hessian_jnp[2][2])
    if slacks == None:
        upper_part = jnp.concatenate([upper_left, -upper_right], axis=1)
        lower_part = jnp.concatenate([-lower_left, lower_right], axis=1)
        hessian_np = jnp.concatenate([upper_part, lower_part], axis=0)
    else:
        upper_middle = jnp.asarray(hessian_jnp[0][1])
        middle_left = jnp.asarray(hessian_jnp[1][0])
        middle = jnp.asarray(hessian_jnp[1][1])
        var = jnp.asarray(middle).shape[0]
        middle_ = jnp.zeros((var, var)) ### ADDED
        for i in range(var):
            #middle_[i][i] = (barrier_initialization_parameter / (slacks[i] ** 2))
            middle_ = middle_.at[i, i].add((lagrange_multipliers[i] / slacks[i])) # Wieso auch immer..
        middle_right = jnp.asarray(hessian_jnp[1][2])
        lower_middle = jnp.asarray(hessian_jnp[2][1])
        upper_part = jnp.concatenate([upper_left, upper_middle, -upper_right], axis=1)
        middle_part = jnp.concatenate([middle_left, middle_, -middle_right], axis=1)
        lower_part = jnp.concatenate([-lower_left, -lower_middle, lower_right], axis=1)
        hessian_np = jnp.concatenate([upper_part, middle_part, lower_part], axis=0)
    return hessian_np


def regularize_hessian(hessian_matrix, num_weights, num_equality_constraints=0, num_inequality_constraints=0,
                       diagonal_shift_val=0.0, init_diagonal_shift_val=0.5, armijo_val=1.0E-4, power_val=0.4,
                       barrier_val=0.2):
    """Regularize the Hessian to avoid ill-conditioning and to escape saddle points."""
    # Nocedal & Wright 19.25 / Appendix B1, further info: Wächter & Biegler
    # constants are arbitrary but typical choices
    eigenvalues = jnp.linalg.eigvalsh(hessian_matrix)#, jnp.eye(hessian_matrix.shape[0]))
    condition_num = jnp.min(jnp.abs(eigenvalues)) / jnp.max(jnp.abs(eigenvalues))
    if condition_num <= minimal_step or (num_equality_constraints + num_inequality_constraints) != jnp.sum(
            eigenvalues < -minimal_step):
        if condition_num <= minimal_step and num_equality_constraints:
            lower_index = num_weights + num_inequality_constraints
            upper_index = lower_index + num_equality_constraints
            regularization_val = jnp.float64(jnp.sqrt(minimal_step))
            hessian_matrix[lower_index:upper_index, lower_index:upper_index] -= regularization_val * armijo_val * (
                        barrier_val ** power_val) * jnp.eye(num_equality_constraints)
        if diagonal_shift_val == 0.0:  # diagonal shift coefficient must not be zero
            diagonal_shift_val = init_diagonal_shift_val
        else:  # diagonal shift coefficient must not be too small
            diagonal_shift_val = jnp.max([diagonal_shift_val / 2, init_diagonal_shift_val])

        # regularize Hessian with diagonal shift matrix (delta*I) until matrix inertia condition is satisfied
        hessian_matrix.at[:num_weights, :num_weights].add(diagonal_shift_val * jnp.eye(num_weights))
        eigenvalues = jnp.linalg.eigvalsh(hessian_matrix)
        while (num_equality_constraints + num_inequality_constraints) != jnp.sum(eigenvalues < -minimal_step):
            hessian_matrix = hessian_matrix.at[:num_weights, :num_weights].add(-diagonal_shift_val * jnp.eye(num_weights))
            diagonal_shift_val *= 10.0
            hessian_matrix = hessian_matrix.at[:num_weights, :num_weights].add(diagonal_shift_val * jnp.eye(num_weights))
            eigenvalues = jnp.linalg.eigvalsh(hessian_matrix)

    return hessian_matrix


def concatenate_constraints(weights, slacks, equality_constraints=None, inequality_constraints=None,
                            num_equality_constraints=0, num_inequality_constraints=0):
    concatenated_ary = jnp.zeros((num_equality_constraints + num_inequality_constraints))
    iter_equality = 0
    iter_inequality = 0
    for iter_equality in range(num_equality_constraints):
        concatenated_ary = concatenated_ary.at[iter_equality].set(equality_constraints[iter_equality](weights))
    for iter_inequality in range(num_inequality_constraints):
        inequality_val = inequality_constraints[iter_inequality](weights) - slacks[iter_inequality]
        concatenated_ary = concatenated_ary.at[iter_equality + iter_inequality].set(inequality_val)
    return concatenated_ary


def step(values, derivative_of_values, weight_precision_tolerance=jnp.finfo(jnp.float32).eps, backtracking_line_search_parameter=0.995):
    """Golden section search used to determine the maximum step length for slack variables and Lagrange multipliers
       using the fraction-to-the-boundary rule."""
    GOLD_constant = (jnp.sqrt(5.0) + 1.0) / 2.0
    weighted_values = (1.0 - backtracking_line_search_parameter) * values  # default: 0.05*values
    small_return = 0.0
    big_return = 1.0

    if jnp.all(values + big_return * derivative_of_values >= weighted_values):
        return big_return
    else:
        decreasing_variable = big_return - (big_return - small_return) / GOLD_constant
        increasing_variable = small_return + (big_return - small_return) / GOLD_constant
        while jnp.abs(big_return - small_return) > GOLD_constant * weight_precision_tolerance:
            if jnp.any(values + increasing_variable * derivative_of_values < weighted_values):
                big_return = increasing_variable
            else:
                small_return = increasing_variable
            if decreasing_variable > small_return:
                if jnp.any(values + decreasing_variable * derivative_of_values < weighted_values):
                    big_return = decreasing_variable
                else:
                    small_return = decreasing_variable

            decreasing_variable = big_return - (big_return - small_return) / GOLD_constant
            increasing_variable = small_return + (big_return - small_return) / GOLD_constant

        return small_return


def barrier_function(slacks, barrier_val):
    return barrier_val * jnp.sum(jnp.log(slacks))


def sum_inequality_values(inequality_constraints, weights, slacks, num_inequality_constraints):
    sum = 0
    for iter_var in range(num_inequality_constraints):
        sum += jnp.abs(inequality_constraints[iter_var](weights) - slacks[iter_var])
    return sum


def sum_equality_values(equality_constraints, weights, num_equality_constraints):
    sum = 0
    for iter_var in range(num_equality_constraints):
        sum += jnp.abs(equality_constraints[iter_var](weights))
    return sum


def merit_function(cost_function, weights, slacks, merit_function_parameter, barrier_val, equality_constraints=None,
                   inequality_constraints=None, num_equality_constraints=0, num_inequality_constraints=0):
    if num_equality_constraints and num_inequality_constraints:
        equality_sum = sum_equality_values(equality_constraints, weights, num_equality_constraints)
        inequality_sum = sum_inequality_values(inequality_constraints, weights, slacks, num_inequality_constraints)
        constraints_val = merit_function_parameter * (equality_sum + inequality_sum)
        return cost_function(weights) + constraints_val - barrier_function(slacks, barrier_val)

    elif num_equality_constraints:
        equality_sum = sum_equality_values(equality_constraints, weights, num_equality_constraints)
        return cost_function(weights) + merit_function_parameter * equality_sum

    elif num_inequality_constraints:
        inequality_sum = sum_inequality_values(inequality_constraints, weights, slacks, num_inequality_constraints)
        return cost_function(weights) + merit_function_parameter * inequality_sum - barrier_function(slacks, barrier_val)
    else:
        return cost_function(weights)


def jacobian_of_constraints(weights, num_weights, equality_constraints=None, inequality_constraints=None,
                                num_equality_constraints=0, num_inequality_constraints=0):
    # TODO alle pfade prüfen
    if num_equality_constraints and num_inequality_constraints:
        grad_equality_ary = jnp.zeros((num_equality_constraints, num_weights))
        for iter1 in range(num_equality_constraints):
            grad_equality_ary[iter1] = jnp.array(grad(equality_constraints[iter1])(weights))
        grad_inequality_ary = jnp.zeros((num_inequality_constraints, num_weights))
        for iter2 in range(num_inequality_constraints):
            grad_inequality_ary[iter2] = jnp.array(grad(inequality_constraints[iter2])(weights))
        jacobian_top = jnp.concatenate([grad_equality_ary.reshape((num_weights, num_equality_constraints)), \
                                       grad_inequality_ary.reshape((num_weights, num_inequality_constraints))], axis=1)
        jacobian_bottom = jnp.concatenate([jnp.zeros((num_inequality_constraints, num_equality_constraints)),
                                          -jnp.eye(num_inequality_constraints)], axis=1)
        return jnp.concatenate([jacobian_top, jacobian_bottom], axis=0)

    elif num_equality_constraints:
        empt_ary = jnp.zeros((num_equality_constraints, num_weights))
        for iter in range(num_equality_constraints):
            empt_ary[iter] = jnp.array(grad(equality_constraints[iter])(weights))
        return empt_ary.T ### ÄNDERUNG TODO

    elif num_inequality_constraints:### HIer PROBLEM
        empt_ary = jnp.zeros((num_inequality_constraints, num_weights))
        for iter in range(num_inequality_constraints):
            empt_ary[iter] = jnp.array(grad(inequality_constraints[iter])(weights))
        return (jnp.concatenate([empt_ary.T, -jnp.eye(num_inequality_constraints)], axis=0)).T
    else:
        return jnp.zeros((num_weights + num_inequality_constraints,
                         num_equality_constraints + num_inequality_constraints))


def gradient_merit_function(cost_function, weights, slacks, search_direction, num_weights, barrier_val,
                            equality_constraints=None, inequality_constraints=None, num_equality_constraints=0,
                            num_inequality_constraints=0, merit_function_parameter=10.0):
    if num_equality_constraints and num_inequality_constraints:
        equality_sum = sum_equality_values(equality_constraints, weights, num_equality_constraints)
        inequality_sum = sum_inequality_values(inequality_constraints, weights, slacks, num_inequality_constraints)
        return (jnp.dot(grad(cost_function)(weights), search_direction[:num_weights]) \
                - merit_function_parameter * (equality_sum + inequality_sum) \
                - jnp.dot(barrier_val / (slacks + minimal_step), search_direction[num_weights:]))
    elif num_equality_constraints:  # TODO: nachprüfen, ob das so klappt
        equality_sum = sum_equality_values(equality_constraints, weights, num_equality_constraints)
        return (jnp.dot(grad(cost_function)(weights), search_direction[:num_weights]) \
                - merit_function_parameter * equality_sum)
    elif num_inequality_constraints:
        inequality_sum = sum_inequality_values(inequality_constraints, weights, slacks, num_inequality_constraints)
        return (jnp.dot(grad(cost_function)(weights), search_direction[:num_weights]) \
                - merit_function_parameter * inequality_sum \
                - jnp.dot(barrier_val / (slacks + minimal_step), search_direction[num_weights:]))
    else:
        return jnp.dot(grad(cost_function)(weights), search_direction[:num_weights])


def barrier_cost_gradient(objective_function, weights, slacks, num_inequality_constraints, barrier_val):
    if num_inequality_constraints:
        return jnp.concatenate(
            [gradient_np_ary(objective_function, weights, slacks, lagrange_multipliers, barrier_val)[:num_weights], -barrier_val / (slacks + minimal_value)], axis=0)
    else:
        return grad(objective_function, 0)(weights)  # TODO: to be revisited


def search(cost_function, weights_0, slacks_0, lagrange_multipliers_0, search_direction, alpha_smax, alpha_lmax, num_weights,
           barrier_val, equality_constraints=None, inequality_constraints=None, num_equality_constraints=0,
           num_inequality_constraints=0, verbosity=1, backtracking_line_search_parameter=0.995, armijo_val=1.0E-4):
    """Backtracking line search to find a solution that leads to a smaller value of the Lagrangian within the confines
       of the maximum step length for the slack variables and Lagrange multipliers found using class function 'step'."""
    ### TDOD : HIER IWO FEHLER, MINIMIERUNG SCHeitert --> weights, slacks

    # extract search directions along weights, slacks, and multipliers
    optimization_return_signal = 0
    search_direction_weights = search_direction[:num_weights]
    if num_inequality_constraints:
        search_direction_slacks = search_direction[num_weights:(num_weights + num_inequality_constraints)]
    if num_equality_constraints or num_inequality_constraints:
        search_direction_lagrange_multipliers = search_direction[(num_weights + num_inequality_constraints):]
    else:
        search_direction_lagrange_multipliers = jnp.float64(0.0)
        alpha_lmax = jnp.float64(0.0)
    weights = weights_0 # left np.copy away
    slacks = slacks_0
    merit_function_result = merit_function(cost_function, weights_0, slacks_0, merit_function_parameter, barrier_val,  \
        equality_constraints, inequality_constraints, num_equality_constraints, num_inequality_constraints)
    gradient_merit_function_result = gradient_merit_function(cost_function, weights_0, slacks_0, \
            search_direction[:num_weights + num_inequality_constraints], num_weights, barrier_val, equality_constraints, \
            inequality_constraints, num_equality_constraints, num_inequality_constraints, merit_function_parameter)
    correction = False
    if num_inequality_constraints:
        # step search when there are inequality constraints
        new_merit_val = merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights,
                       slacks_0 + alpha_smax * search_direction_slacks, merit_function_parameter, barrier_val, equality_constraints, \
                       inequality_constraints, num_equality_constraints, num_inequality_constraints)
        old_merit_val = merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result
        if new_merit_val > old_merit_val:
            # second-order correction
            correction_old = concatenate_constraints(weights_0, slacks_0, equality_constraints, inequality_constraints,
                                                     num_equality_constraints, num_inequality_constraints)
            correction_new = concatenate_constraints(weights_0 + alpha_smax * search_direction_weights,
                                                     slacks_0 + alpha_smax * search_direction_slacks,
                                                     equality_constraints, inequality_constraints,
                                                     num_equality_constraints, num_inequality_constraints)
            if jnp.sum(jnp.abs(correction_new)) > jnp.sum(jnp.abs(correction_old)):
                # infeasibility has increased, attempt to correct
                jacobian_matrix_of_constraints = jacobian_of_constraints(weights, num_weights,
                                                                                 equality_constraints,
                                                                                 inequality_constraints,
                                                                                 num_equality_constraints,
                                                                                 num_inequality_constraints).T
                try:
                    feasibility_restoration_direction = -jnp.linalg.solve(jacobian_matrix_of_constraints, \
                                         correction_new.reshape((num_weights + num_inequality_constraints, 1))) \
                                            .reshape((num_weights + num_inequality_constraints,))
                except:
                    # if the Jacobian is not invertible, find the minimum norm solution instead
                    feasibility_restoration_direction = - \
                    jnp.linalg.lstsq(jacobian_matrix_of_constraints, correction_new, rcond=None)[0]  #jnp or np
                if (merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights \
                                + feasibility_restoration_direction[:num_weights], slacks_0 \
                                + alpha_smax * search_direction_slacks + feasibility_restoration_direction[num_weights:], \
                                   merit_function_parameter, barrier_val, equality_constraints, inequality_constraints, \
                                   num_equality_constraints, num_inequality_constraints) \
                                <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result):
                    alpha_corr = step(slacks_0, alpha_smax * search_direction_slacks \
                                      + feasibility_restoration_direction[num_weights:], \
                                      weight_precision_tolerance, backtracking_line_search_parameter)
                    merit_val = merit_function(cost_function,
                                            weights_0 + alpha_corr * (alpha_smax  * search_direction_weights + feasibility_restoration_direction[:num_weights]), \
                                            slacks_0 + alpha_corr * (alpha_smax * search_direction_slacks + feasibility_restoration_direction[num_weights:]), \
                                            merit_function_parameter, barrier_val, equality_constraints, inequality_constraints, \
                                            num_equality_constraints, num_inequality_constraints)
                    if (merit_val <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result):
                        if verbosity > 2:
                            print('Second-order feasibility correction accepted')
                        # correction accepted
                        correction = True
            if not correction:
                # infeasibility has not increased, no correction necessary
                alpha_smax *= backtracking_line_search_parameter
                alpha_lmax *= backtracking_line_search_parameter
                while merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights,
                        slacks_0 + alpha_smax * search_direction_slacks, merit_function_parameter, barrier_val, \
                        equality_constraints, inequality_constraints, num_equality_constraints, \
                        num_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                    # backtracking line search
                    if (jnp.sqrt(jnp.linalg.norm(alpha_smax * search_direction_weights) ** 2 + jnp.linalg.norm(
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
                        alpha_smax * search_direction_slacks + feasibility_restoration_direction[num_weights:])
        else:
            slacks = slacks_0 + alpha_smax * search_direction_slacks
    else:
        # step search for only equality constraints or unconstrained problems
        if merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights, slacks_0,
                        merit_function_parameter, barrier_val, equality_constraints, inequality_constraints, \
                        num_equality_constraints, num_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
            if num_equality_constraints:
                # second-order correction
                correction_old = concatenate_constraints(weights_0, slacks_0, equality_constraints,
                                                         inequality_constraints, num_equality_constraints,
                                                         num_inequality_constraints)
                correction_new = concatenate_constraints(weights_0 + alpha_smax * search_direction_weights, slacks_0,
                                                         equality_constraints, inequality_constraints,
                                                         num_equality_constraints, num_inequality_constraints)
                if jnp.sum(jnp.abs(correction_new)) > jnp.sum(jnp.abs(correction_old)):
                    # infeasibility has increased, attempt to correct
                    jacobian_matrix_of_constraints = jacobian_of_constraints(weights, num_weights,
                                                                                     equality_constraints,
                                                                                     inequality_constraints,
                                                                                     num_equality_constraints,
                                                                                     num_inequality_constraints).T
                    try:
                        # calculate a feasibility restoration direction
                        feasibility_restoration_direction = -jnp.linalg.solve(jacobian_matrix_of_constraints, \
                                    correction_new.reshape((num_weights, num_inequality_constraints, 1))) \
                                    .reshape((num_weights + num_inequality_constraints,))
                    except:
                        # if the Jacobian is not invertible, find the minimum norm solution instead
                        feasibility_restoration_direction = - \
                        jnp.linalg.lstsq(jacobian_matrix_of_constraints, correction_new, rcond=None)[0]
                    if merit_function(cost_function,
                                weights_0 + alpha_smax * search_direction_weights + feasibility_restoration_direction,
                                slacks_0, merit_function_parameter, barrier_val, equality_constraints, inequality_constraints, \
                                num_equality_constraints, num_inequality_constraints) \
                                <= merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                        # correction accepted
                        if verbosity > 2:
                            print('Second-order feasibility correction accepted')
                        alpha_corr = jnp.float64(1.0)
                        correction = True
            if not correction:
                # infeasibility has not increased, no correction necessary
                alpha_smax *= backtracking_line_search_parameter
                alpha_lmax *= backtracking_line_search_parameter
                while merit_function(cost_function, weights_0 + alpha_smax * search_direction_weights, slacks_0,
                        merit_function_parameter, barrier_val, equality_constraints, inequality_constraints, \
                        num_equality_constraints, num_inequality_constraints) \
                        > merit_function_result + alpha_smax * armijo_val * gradient_merit_function_result:
                    # backtracking line search
                    if jnp.linalg.norm(alpha_smax * search_direction_weights) < minimal_step:
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
                    alpha_smax * search_direction_weights + feasibility_restoration_direction[:num_weights])
    else:
        weights = weights_0 + alpha_smax * search_direction_weights
    # update multipliers (if applicable)
    if num_equality_constraints or num_inequality_constraints:
        lagrange_multipliers = lagrange_multipliers_0 + alpha_lmax * search_direction_lagrange_multipliers
    else:
        lagrange_multipliers = lagrange_multipliers_0
    # return updated weights, slacks, and multipliers
    return weights, slacks, lagrange_multipliers


