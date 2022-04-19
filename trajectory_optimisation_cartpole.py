import jax.numpy as jnp
#import nlp_jax_ipm


def cart_pole_dynamics(angle, velocity, angle_speed, control, cart_mass,
                       pole_mass, gravity, pole_length):
    """

    Parameters
    ----------
    angle
    velocity
    angle_speed
    control
    cart_mass
    pole_mass
    gravity
    pole_length

    Returns
    -------
    velocity
    angle_speed
    acceleration
    angle_acceleration
    """

    # acceleration = derivative_derivative_position
    acceleration = ((1.0 / (cart_mass + pole_mass
                            - pole_mass * (jnp.cos(angle) ** 2)))
                    * (control
                       + gravity * pole_mass * jnp.cos(angle)
                        * jnp.sin(angle)
                       + pole_length * pole_mass
                        * jnp.sin(angle) * angle_speed ** 2))

    # angle_acceleration = derivative_derivative_angle
    angle_acceleration = (-(1.0 / (cart_mass + pole_mass
                                   - pole_mass * (jnp.cos(angle) ** 2))
                            * (jnp.cos(angle) * control
                          + gravity * cart_mass * jnp.sin(angle)
                          + pole_length * pole_mass * jnp.cos(angle)
                            * jnp.sin(angle) * angle_speed ** 2)))

    derivative_of_state = jnp.array(velocity, angle_speed, acceleration,
                                    angle_acceleration)
    return derivative_of_state

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

# start and end values
pos[0] = 0
vel[0] = 0
angle[0] = 0
angle_speed[0] = 0
pos[num_knot_points - 1] = pos_end
vel[num_knot_points - 1] = 0
angle[num_knot_points - 1] = angle_end
angle_speed[num_knot_points - 1] = 0

# weights, slacks, multipliers
number_weights = num_knot_points * 5 - 8
key = random.PRNGKey(1702)
weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
slacks = random.normal(key, shape=(num_knot_points * 2,)).astype(jnp.float32)
lagrange_multipliers = random.normal(key, shape=(num_knot_points - 1,)).astype(jnp.float32)

# call solver
"""
(weights, slacks, lagrange_multipliers, function_values, kkt_weights,
 kkt_slacks, kkt_equality_lagrange_multipliers,
 kkt_inequality_lagrange_multipliers) = (
        solve(cost_function, equality_constraints,
              inequality_constraints, weights, slacks, lagrange_multipliers))
"""