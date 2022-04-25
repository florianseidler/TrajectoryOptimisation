# constants
damping_norm = 0.1
gravity_norm = 1


def angle_acceleration(control, angle, angle_speed):
    # from dynamics
    return (- damping_norm * angle_speed -
            gravity_norm * jnp.sin(angle) + control)

