import jax.numpy as jnp
from jax import jit


def merit_function(cost_function, weights, slacks, merit_function_parameter,
                   barrier_val, equality_constraints=None,
                   inequality_constraints=None, num_equality_constraints=0,
                   num_inequality_constraints=0):
    """
    Calculates merit function used for step search to minimize Lagrangian.

    Parameters
    ----------
    cost_function
    weights
    slacks
    merit_function_parameter
    barrier_val
    equality_constraints
    inequality_constraints
    num_equality_constraints
    num_inequality_constraints

    Returns
    -------
    result of the merit function
    """

    if num_equality_constraints and num_inequality_constraints:
        equality_sum = sum_equality_values(
            equality_constraints, weights, num_equality_constraints)
        inequality_sum = sum_inequality_values(
            inequality_constraints, weights, slacks, num_inequality_constraints)
        constraints_val = merit_function_parameter * (equality_sum
                                                      + inequality_sum)
        return (cost_function(weights) + constraints_val
                - barrier_function(slacks, barrier_val))

    elif num_equality_constraints:
        equality_sum = sum_equality_values(
            equality_constraints, weights, num_equality_constraints)
        return cost_function(weights) + merit_function_parameter * equality_sum

    elif num_inequality_constraints:
        inequality_sum = sum_inequality_values(
            inequality_constraints, weights, slacks, num_inequality_constraints)
        return (cost_function(weights)
                + merit_function_parameter * inequality_sum
                - barrier_function(slacks, barrier_val))
    else:
        return cost_function(weights)


@jit
def barrier_function(slacks, barrier_val):
    """
    Calculates the barrier function.

    Parameters
    ----------
    slacks
    barrier_val

    Returns
    -------
    barrier function
    """

    return barrier_val * jnp.sum(jnp.log(slacks))


def sum_inequality_values(inequality_constraints, weights, slacks,
                          num_inequality_constraints):
    """
    Calculates the sum of the inequality constraints with inserted weights
    and slacks.

    Parameters
    ----------
    inequality_constraints
    weights
    slacks
    num_inequality_constraints

    Returns
    -------
    sum of the inequalities with inserted weights and slacks
    """

    loss = 0
    for iter_var in range(num_inequality_constraints):
        loss += jnp.abs(inequality_constraints[iter_var](weights)
                        - slacks[iter_var])
    return loss


def sum_equality_values(equality_constraints, weights,
                        num_equality_constraints):
    """
    Calculates the sum of the equality constraints with inserted weights.

    Parameters
    ----------
    equality_constraints
    weights
    num_equality_constraints

    Returns
    -------
    sum of the equalities with inserted weights
    """
    total_loss = 0
    for equality_constraint in equality_constraints:
        loss = equality_constraint(weights)
        loss = jnp.abs(loss)
        total_loss = total_loss - loss

    return total_loss
