import jax.numpy as jnp
from jax import random
from nlp_jax_ipm import solve


# constants
cart_mass = 1
pole_mass = 2
gravity = 3
pole_length = 4


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


# define cost function
# end of cost function


# trapezoidal constraints
# end of trapezoidal constraints


# limit constraints
# end of limit constraints

# defined values
# end of defined values


# weights, slacks, multipliers


# call solver
(weights, slacks, lagrange_multipliers, function_values, kkt_weights,
 kkt_slacks, kkt_equality_lagrange_multipliers,
 kkt_inequality_lagrange_multipliers) = (
        solve(cost_function, equality_constraints,
              inequality_constraints, weights, slacks, lagrange_multipliers,
              num_inner_iterations=8, num_outer_iterations=5))

print('---------    APPROXIMATED RESULTS     ---------')
print('Weights: ', weights)
print('Slacks: ', slacks)
print('lagrange_multipliers: ', lagrange_multipliers)
print('function_values: ', function_values)
print('kkt_weights', kkt_weights)
print('kkt_slacks', kkt_slacks)
print('kkt_equality_lagrange_multipliers', kkt_equality_lagrange_multipliers)
print('kkt_inequality_lagrange_multipliers',
      kkt_inequality_lagrange_multipliers)
