import jax.numpy as jnp
from jax import grad, jit, vmap, hessian, jacfwd
from KKT_calculations import KKT


def initialization(cost_function, equality_constraints, inequality_constraints, input_weights=None, input_slacks=None,
                   input_lagrange_multipliers=None):

    """

    Parameters
    ----------
    cost_function
    equality_constraints
    inequality_constraints
    input_weights
    input_slacks
    input_lagrange_multipliers

    Returns
    -------
    verbosity, backtracking_line_search_parameter, armijo_val, weight_precision_tolerance, \
        convergence_tolerance_cost_function, power_val, init_diagonal_shift_val, minimal_step, diagonal_shift_val, \
        kkt_tolerance, update_factor_merit_function_parameter, merit_function_parameter, barrier_val, \
        merit_function_initialization_parameter, num_inner_iterations, num_outer_iterations, objective_function, \
        weights, slacks, lagrange_multipliers, num_weights, num_inequality_constraints, num_equality_constraints, \
        kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers, \
        old_cost_function, convergence_tolerance_kkt_converged, convergence_tolerance_cost_function_converged, \
        optimization_return_signal

    """

    # tunable constants
    verbosity = 1
    backtracking_line_search_parameter = 0.995
    armijo_val = 1.0E-4
    weight_precision_tolerance = jnp.finfo(jnp.float32).eps  # cannot be 0!
    convergence_tolerance_cost_function = 0.00000001
    power_val = 0.4
    init_diagonal_shift_val = 0.5
    minimal_step = jnp.finfo(jnp.float32).eps
    diagonal_shift_val = 0.0
    kkt_tolerance = 1.0E-4
    update_factor_merit_function_parameter = 0.1
    merit_function_parameter = 10.0
    merit_function_initialization_parameter = jnp.float32(10.0)
    num_inner_iterations = 20
    num_outer_iterations = 10


    @jit
    def equality_constraints_fct(weights, lagrange_multipliers):
        num_equality_constraints = jnp.size(equality_constraints)
        return_sum = 0
        iter = 0
        for iter in range(num_equality_constraints):
            return_sum += equality_constraints[iter](weights) * lagrange_multipliers[iter]
        return return_sum


    @jit
    def inequality_constraints_fct(weights, slacks, lagrange_multipliers):
        num_inequality_constraints = jnp.size(inequality_constraints)
        num_equality_constraints = jnp.size(equality_constraints)
        iter = 0
        return_sum = 0
        for iter in range(num_inequality_constraints):
            return_sum += (inequality_constraints[iter](weights) - slacks[iter]) * lagrange_multipliers[
                iter + num_equality_constraints]
        return return_sum


    @jit
    def objective_function(weights, slacks, lagrange_multipliers):
        return jnp.asarray(cost_function(weights) - equality_constraints_fct(weights, lagrange_multipliers) -
                           inequality_constraints_fct(weights, slacks, lagrange_multipliers))


    if input_weights is None:
        key = jax.random.PRNGKey(1702)
        weights = random.normal(key, shape=(num_weights,)).astype(jnp.float32)
    else:
        weights = input_weights

    num_weights = jnp.size(weights)
    num_equality_constraints = jnp.size(equality_constraints)
    num_inequality_constraints = jnp.size(inequality_constraints)

    if num_weights == 0:
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
            lagrange_multipliers = \
                init_lagrange_multipliers(cost_function, weights, num_weights, num_equality_constraints,
                                          num_inequality_constraints, equality_constraints, inequality_constraints)
            if num_inequality_constraints and num_equality_constraints:
                lagrange_multipliers_inequality = lagrange_multipliers[num_equality_constraints:]
                lagrange_multipliers_inequality[lagrange_multipliers_inequality < jnp.float32(0.0)] = \
                    jnp.float32(kkt_tolerance)
                lagrange_multipliers[num_equality_constraints:] = lagrange_multipliers_inequality
            elif num_inequality_constraints:
                lagrange_multipliers[lagrange_multipliers < jnp.float32(0.0)] = jnp.float32(kkt_tolerance)
        else:
            lagrange_multipliers = input_lagrange_multipliers.astype(jnp.float32)
    else:
        lagrange_multipliers = jnp.array([], dtype=jnp.float32)
        # return print("No optimization necessary without constraints")

    # init barrier val
    if num_inequality_constraints:
        barrier_val = 0.2
    else:
        barrier_val = kkt_tolerance

    # calculate the initial KKT conditions
    kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers = \
        KKT(objective_function, weights, slacks, lagrange_multipliers, num_inequality_constraints,
            num_equality_constraints, barrier_val)

    # if convergence_tolerance_cost_function is set, calculate the prior cost
    if convergence_tolerance_cost_function is not None:
        old_cost_function = cost_function(weights)

    # init convergences
    convergence_tolerance_kkt_converged = False
    convergence_tolerance_cost_function_converged = False

    # init optimization return signal
    optimization_return_signal = 0

    return verbosity, backtracking_line_search_parameter, armijo_val, weight_precision_tolerance, \
        convergence_tolerance_cost_function, power_val, init_diagonal_shift_val, minimal_step, diagonal_shift_val, \
        kkt_tolerance, update_factor_merit_function_parameter, merit_function_parameter, barrier_val, \
        merit_function_initialization_parameter, num_inner_iterations, num_outer_iterations, objective_function, \
        weights, slacks, lagrange_multipliers, num_weights, num_inequality_constraints, num_equality_constraints, \
        kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers, \
        old_cost_function, convergence_tolerance_kkt_converged, convergence_tolerance_cost_function_converged, \
        optimization_return_signal


def init_lagrange_multipliers(cost_function, weights, num_weights, num_equality_constraints=0, \
                              num_inequality_constraints=0, equality_constraints=None, inequality_constraints=None):
    """

    Parameters
    ----------
    cost_function
    weights
    num_weights
    num_equality_constraints
    num_inequality_constraints
    equality_constraints
    inequality_constraints

    Returns
    -------
    lagrange_multipliers

    """

    jacobian_constraints = \
        jacobian_of_constraints(weights, num_weights, equality_constraints, inequality_constraints,
                                num_equality_constraints, num_inequality_constraints)[:num_weights, :]

    gradient_cost_function = grad(cost_function, 0)(weights).reshape((num_weights, 1))

    return jnp.dot(jnp.linalg.pinv(jacobian_constraints, gradient_cost_function))\
        .reshape((num_equality_constraints + num_inequality_constraints,))
