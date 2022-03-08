import jax.numpy as jnp
from jax import grad, hessian, jacfwd, lax, jit


def concatenate_constraints(weights, slacks, equality_constraints=None, inequality_constraints=None,
                            num_equality_constraints=0, num_inequality_constraints=0):

    """Returns array out of equality and inequality constraints with applied weights"""

    concatenated_ary = jnp.zeros((num_equality_constraints + num_inequality_constraints))
    iter_equality = 0
    iter_inequality = 0
    for iter_equality in range(num_equality_constraints):
        concatenated_ary = concatenated_ary.at[iter_equality].set(equality_constraints[iter_equality](weights))
    for iter_inequality in range(num_inequality_constraints):
        inequality_val = inequality_constraints[iter_inequality](weights) - slacks[iter_inequality]
        concatenated_ary = concatenated_ary.at[iter_equality + iter_inequality].set(inequality_val)
    return concatenated_ary
