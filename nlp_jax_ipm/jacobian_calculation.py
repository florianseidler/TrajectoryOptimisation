import jax.numpy as jnp
from jax import grad


def jacobian_of_constraints(weights, num_weights, equality_constraints=None,
                            inequality_constraints=None,
                            num_equality_constraints=0,
                            num_inequality_constraints=0):

    """
    Calculates the Jacobian of the constraints by concatenating the
    gradients of the constraints and filling up the matrix.

    Parameters
    ----------
    weights
    num_weights
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints

    Returns
    -------
    jacobian_constraints: Jacobian of the constraints
    """

    if num_equality_constraints and num_inequality_constraints:

        grad_equality_ary = jnp.zeros((num_equality_constraints, num_weights))
        for equality_constraint in range(num_equality_constraints):
            grad_equality_ary = grad_equality_ary.at[equality_constraint].set(jnp.array(
                grad(equality_constraints[equality_constraint])(weights)))

        grad_inequality_ary = jnp.zeros(
            (num_inequality_constraints, num_weights))
        for inequality_constraint in range(num_inequality_constraints):
            grad_inequality_ary = grad_inequality_ary.at[inequality_constraint].set(jnp.array(
                grad(inequality_constraints[inequality_constraint])(weights)))
            grad_equality_ary = grad_equality_ary.reshape(
                (num_weights, num_equality_constraints))
            grad_inequality_ary = grad_inequality_ary.reshape(
                (num_weights, num_inequality_constraints))

        jacobian_top = jnp.concatenate(
            [grad_equality_ary, grad_inequality_ary], axis=1)
        jacobian_bottom = jnp.concatenate(
            [jnp.zeros((num_inequality_constraints, num_equality_constraints)),
             - jnp.eye(num_inequality_constraints)], axis=1)

        jacobian_constraints = jnp.concatenate([jacobian_top, jacobian_bottom],
                                               axis=0)
        return jacobian_constraints

    elif num_equality_constraints:
        jacobian_constraints = jnp.zeros(
            (num_equality_constraints, num_weights))
        for equality_constraint in range(num_equality_constraints):
            jacobian_constraints.at[equality_constraint].set(jnp.array(
                grad(equality_constraints[equality_constraint])(weights)))

        return jacobian_constraints.T

    elif num_inequality_constraints:
        jacobian = jnp.zeros((num_inequality_constraints, num_weights))
        for inequality_constraint in range(num_inequality_constraints):
            jacobian.at[inequality_constraint].set(jnp.array(
                grad(inequality_constraints[inequality_constraint])(weights)))
            jacobian_constraints = (
                jnp.concatenate([jacobian.T,
                                 - jnp.eye(num_inequality_constraints)],
                                axis=0)).T

        return jacobian_constraints

    else:
        jacobian_constraints = (
            jnp.zeros((num_weights + num_inequality_constraints,
                       num_equality_constraints + num_inequality_constraints)))
        return jacobian_constraints
