import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit


def jacobian_of_constraints(weights, num_weights, equality_constraints=None, inequality_constraints=None,
                                num_equality_constraints=0, num_inequality_constraints=0):

    """ Calculates the Jacobian of the constraints by concatenating the gradients of the constraints
    and filling up the matrix. """

    if num_equality_constraints and num_inequality_constraints:
        grad_equality_ary = jnp.zeros((num_equality_constraints, num_weights))
        for iter1 in range(num_equality_constraints):
            grad_equality_ary[iter1] = jnp.array(grad(equality_constraints[iter1])(weights))
        grad_inequality_ary = jnp.zeros((num_inequality_constraints, num_weights))
        for iter2 in range(num_inequality_constraints):
            grad_inequality_ary[iter2] = jnp.array(grad(inequality_constraints[iter2])(weights))
        jacobian_top = jnp.concatenate([grad_equality_ary.reshape((num_weights, num_equality_constraints)),
                                       grad_inequality_ary.reshape((num_weights, num_inequality_constraints))], axis=1)
        jacobian_bottom = jnp.concatenate([jnp.zeros((num_inequality_constraints, num_equality_constraints)),
                                          -jnp.eye(num_inequality_constraints)], axis=1)
        return jnp.concatenate([jacobian_top, jacobian_bottom], axis=0)

    elif num_equality_constraints:
        jacobian = jnp.zeros((num_equality_constraints, num_weights))
        for iter in range(num_equality_constraints):
            jacobian[iter] = jnp.array(grad(equality_constraints[iter])(weights))
        return jacobian.T

    elif num_inequality_constraints:
        jacobian = jnp.zeros((num_inequality_constraints, num_weights))
        for iter in range(num_inequality_constraints):
            jacobian[iter] = jnp.array(grad(inequality_constraints[iter])(weights))
        return (jnp.concatenate([jacobian.T, -jnp.eye(num_inequality_constraints)], axis=0)).T

    else:
        return jnp.zeros((num_weights + num_inequality_constraints,
                         num_equality_constraints + num_inequality_constraints))
