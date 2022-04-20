import jax.numpy as jnp
from jax import random
from nlp_jax_ipm import solve


def acceleration(control, angle, angle_speed):
    # from dynamics
    return ((1.0 / (cart_mass + pole_mass
                    - pole_mass * (jnp.cos(angle) ** 2)))
            * (control + gravity * pole_mass * jnp.cos(angle)
            * jnp.sin(angle) + pole_length * pole_mass
            * jnp.sin(angle) * angle_speed ** 2))


def angle_acceleration(control, angle, angle_speed):
    # from dynamics
    return (-(1.0 / (cart_mass + pole_mass - pole_mass * (jnp.cos(angle) ** 2))
            * (jnp.cos(angle) * control
            + gravity * cart_mass * jnp.sin(angle)
            + pole_length * pole_mass * jnp.cos(angle)
            * jnp.sin(angle) * angle_speed ** 2)))


def control_cost_function(control):
    num_knots = jnp.size(control) - 1
    control_sum = 0
    for iter in range(num_knots):
        control_sum += 0.5 * timestep * (control[iter] ** 2 + control[iter + 1] ** 2)
    return control_sum


# trapezoidal constraints
# end of trapezoidal constraints


# limit constraints
# end of limit constraints

# defined values
# end of defined values


# weights, slacks, multipliers
number_weights = num_knot_points * 5 - 8  # num_knot_points >! 2
key = random.PRNGKey(1702)
weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
slacks = random.normal(key, shape=(num_knot_points * 2,)).astype(jnp.float32)
lagrange_multipliers = random.normal(key, shape=(num_knot_points - 1,)).astype(jnp.float32)

# call solver

(weights, slacks, lagrange_multipliers, function_values, kkt_weights,
 kkt_slacks, kkt_equality_lagrange_multipliers,
 kkt_inequality_lagrange_multipliers) = (
        solve(control_cost_function, equality_constraints,
              inequality_constraints, weights, slacks, lagrange_multipliers))
