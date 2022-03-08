import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit


minimal_step = jnp.finfo(jnp.float32).eps


def gradient_analytic(objective_function, weights, slacks, lagrange_multipliers, barrier_val):
    """ Calculate gradient with analytic calculation for the barrier term. """
    gradient_jnp = jit(grad(objective_function, [0, 1, 2]))(weights, slacks, lagrange_multipliers)
    gradient_weights = jnp.asarray(gradient_jnp[0])
    if slacks == None:
        gradient_slacks = jnp.array([])
    else:
        gradient_slacks = jnp.asarray(gradient_jnp[1])
        num_slacks = jnp.size(slacks)
        iter_slacks = 0
        for iter_slacks in range(num_slacks):  # barrier parameter init
            gradient_slacks = gradient_slacks.at[iter_slacks].add(-(barrier_val / slacks[iter_slacks]))
    gradient_lagrange_multipliers = jnp.asarray(gradient_jnp[2])
    analytic_gradient = jnp.concatenate([gradient_weights, gradient_slacks, -gradient_lagrange_multipliers], axis=0)
    return analytic_gradient


def hessian_approximation(obj_fct, weights, slacks, lagrange_multipliers):
    """ Calculate hessian with approximation for the barrier term. """
    hessian_jnp = jit(hessian(obj_fct, [0, 1, 2]))(weights, slacks, lagrange_multipliers)
    upper_left = jnp.asarray(hessian_jnp[0][0])
    upper_right = jnp.asarray(hessian_jnp[0][2])
    lower_left = jnp.asarray(hessian_jnp[2][0])
    lower_right = jnp.asarray(hessian_jnp[2][2])
    if slacks == None:
        upper_part = jnp.concatenate([upper_left, -upper_right], axis=1)
        lower_part = jnp.concatenate([-lower_left, lower_right], axis=1)
        approximated_hessian = jnp.concatenate([upper_part, lower_part], axis=0)
    else:
        upper_middle = jnp.asarray(hessian_jnp[0][1])
        middle_left = jnp.asarray(hessian_jnp[1][0])
        middle = jnp.asarray(hessian_jnp[1][1])
        dimension_middle = jnp.asarray(middle).shape[0]
        middle_ = jnp.zeros((dimension_middle, dimension_middle))
        for iter_middle in range(dimension_middle):  # approximation of hessian
            middle_ = middle_.at[iter_middle, iter_middle].add((lagrange_multipliers[iter_middle] / slacks[iter_middle]))
        middle_right = jnp.asarray(hessian_jnp[1][2])
        lower_middle = jnp.asarray(hessian_jnp[2][1])
        upper_part = jnp.concatenate([upper_left, upper_middle, -upper_right], axis=1)
        middle_part = jnp.concatenate([middle_left, middle_, -middle_right], axis=1)
        lower_part = jnp.concatenate([-lower_left, -lower_middle, lower_right], axis=1)
        approximated_hessian = jnp.concatenate([upper_part, middle_part, lower_part], axis=0)
    return approximated_hessian


def regularize_hessian(hessian_matrix, num_weights, num_equality_constraints=0, num_inequality_constraints=0,
                       diagonal_shift_val=0.0, init_diagonal_shift_val=0.5, armijo_val=1.0E-4, power_val=0.4,
                       barrier_val=0.2):

    """Regularize the Hessian to avoid ill-conditioning and to escape saddle points
    and to maintain matrix inertia.
    The constants are arbitrary but choosen of typical choices.
    Source: Nocedal & Wright 19.25 / Appendix B1, further info: Wächter & Biegler"""

    eigenvalues = jit(jnp.linalg.eigvalsh)(hessian_matrix)#, jnp.eye(hessian_matrix.shape[0]))
    condition_num = jnp.min(jnp.abs(eigenvalues)) / jnp.max(jnp.abs(eigenvalues))
    if condition_num <= minimal_step or (num_equality_constraints + num_inequality_constraints) != jit(jnp.sum)(
            eigenvalues < -minimal_step):

        if condition_num <= minimal_step and num_equality_constraints:
            lower_index = num_weights + num_inequality_constraints
            upper_index = lower_index + num_equality_constraints
            regularization_val = jnp.float32(jnp.sqrt(minimal_step))
            hessian_matrix.at[lower_index:upper_index, lower_index:upper_index].add(-regularization_val * armijo_val * (
                        barrier_val ** power_val) * jnp.eye(num_equality_constraints))

        if diagonal_shift_val == 0.0:  # diagonal shift coefficient must not be zero
            diagonal_shift_val = init_diagonal_shift_val
        else:  # diagonal shift coefficient must not be too small
            diagonal_shift_val = jit(jnp.max)(jnp.array([diagonal_shift_val / 2, init_diagonal_shift_val]))

        # regularize Hessian with diagonal shift matrix (delta*I) until matrix inertia condition is satisfied
        hessian_matrix.at[:num_weights, :num_weights].add(diagonal_shift_val * jnp.eye(num_weights))
        eigenvalues = jit(jnp.linalg.eigvalsh)(hessian_matrix)
        while (num_equality_constraints + num_inequality_constraints) != jit(jnp.sum)(eigenvalues < -minimal_step):
            hessian_matrix = hessian_matrix.at[:num_weights, :num_weights].add(-diagonal_shift_val * jnp.eye(num_weights))
            diagonal_shift_val *= 10.0
            hessian_matrix = hessian_matrix.at[:num_weights, :num_weights].add(diagonal_shift_val * jnp.eye(num_weights))
            eigenvalues = jit(jnp.linalg.eigvalsh)(hessian_matrix)

    return hessian_matrix, diagonal_shift_val
