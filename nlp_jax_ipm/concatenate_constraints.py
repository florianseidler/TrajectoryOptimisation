import jax.numpy as jnp


def concatenate_constraints(weights, slacks, equality_constraints=None,
                            inequality_constraints=None,
                            num_equality_constraints=0,
                            num_inequality_constraints=0):
    """
    Returns array out of equality and inequality constraints with
    applied weights

    Parameters
    ----------
    weights
    slacks
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints

    Returns
    -------
    concatenated_ary: equality and inequality constraints with inserted values
    """

    num_zeros = num_equality_constraints + num_inequality_constraints
    concatenated_ary = jnp.zeros(num_zeros)
    equality_iter = 0

    for equality_iter in range(num_equality_constraints):
        concatenated_ary = (
            concatenated_ary.at[equality_iter]
            .set(equality_constraints[equality_iter](weights)))
    inequality_iter = 0
    for inequality_iter in range(num_inequality_constraints):
        inequality_val = (inequality_constraints[inequality_iter](weights)
                          - slacks[inequality_iter])
        inequality_position = equality_iter + inequality_iter
        concatenated_ary = (concatenated_ary.at[inequality_position]
                            .set(inequality_val))

    return concatenated_ary
