import jax.numpy as jnp


def step(values, derivative_of_values, weight_precision_tolerance=jnp.finfo(
         jnp.float32).eps, backtracking_line_search_parameter=0.995):
    """
    Golden section search used to determine the maximum step length for slack
    variables and Lagrange multipliers using the fraction-to-the-boundary rule.

    Parameters
    ----------
    values
    derivative_of_values
    weight_precision_tolerance
    backtracking_line_search_parameter

    Returns
    -------
    small_return
    big_return
    """

    GOLD_constant = (jnp.sqrt(5.0) + 1.0) / 2.0
    weighted_values = (1.0 - backtracking_line_search_parameter) * values
    small_return = 0.0
    big_return = 1.0

    if jnp.all(values + big_return * derivative_of_values >= weighted_values):
        return big_return
    else:
        decreasing_variable = (big_return - (big_return - small_return)
                               / GOLD_constant)
        increasing_variable = (small_return + (big_return - small_return)
                               / GOLD_constant)
        gold_precision = GOLD_constant * weight_precision_tolerance
        while jnp.abs(big_return - small_return) > gold_precision:
            if jnp.any(values + increasing_variable * derivative_of_values
                       < weighted_values):
                big_return = increasing_variable
            else:
                small_return = increasing_variable
            if decreasing_variable > small_return:
                if jnp.any(values + decreasing_variable * derivative_of_values
                           < weighted_values):
                    big_return = decreasing_variable
                else:
                    small_return = decreasing_variable

            decreasing_variable = (big_return - (big_return - small_return)
                                   / GOLD_constant)
            increasing_variable = (small_return + (big_return - small_return)
                                   / GOLD_constant)

        return small_return
