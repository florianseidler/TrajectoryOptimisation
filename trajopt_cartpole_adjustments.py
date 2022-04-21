from generate_constraints import (constraints_generator, adjust_parameter,
                                  define_value, define_cost_function,
                                  define_weights_slacks_lagrange_multipliers)

num_knot_points = 4
timestep = 2
num_defined_values = 8

constraints_generator(targetfile="trajopt_cartpole_template.py", num_knot_points=num_knot_points, control_min=1, control_max=5)

define_value("angle_speed_end", 0)
define_value("angle_end", 180)
define_value("velocity_end", 0)
define_value("position_end", 1)
define_value("angle_speed_start", 0)
define_value("angle_start", 0)
define_value("velocity_start", 0)
define_value("position_start", 0)
define_value("timestep", timestep)
define_value("num_knot_points", num_knot_points)

adjust_parameter("trapezoidal_collocation_constraints.py", "position[0]", "position_start")
adjust_parameter("trapezoidal_collocation_constraints.py", "velocity[0]", "velocity_start")
adjust_parameter("trapezoidal_collocation_constraints.py", "angle[0]", "angle_start")
adjust_parameter("trapezoidal_collocation_constraints.py", "angle_speed[0]", "angle_speed_start")
adjust_parameter("trapezoidal_collocation_constraints.py", "position[" + str(num_knot_points - 1) + "]", "position_end")
adjust_parameter("trapezoidal_collocation_constraints.py", "velocity[" + str(num_knot_points - 1) + "]", "velocity_end")
adjust_parameter("trapezoidal_collocation_constraints.py", "angle[" + str(num_knot_points - 1) + "]", "angle_end")
adjust_parameter("trapezoidal_collocation_constraints.py", "angle_speed[" + str(num_knot_points - 1) + "]", "angle_speed_end")

#weights = jnp.concatenate([position[1:-1], angle[1:-1], vecolcity[1:-1], angle_speed[1:-1], control, angle_accel])

for k in range(0, num_knot_points - 2):
    adjust_parameter("trapezoidal_collocation_constraints.py", "position[" + str(k + 1) + "]", "weights[" + str(k) + "]")
for l in range(k + 1, num_knot_points - 1 + k):
    adjust_parameter("trapezoidal_collocation_constraints.py", "angle[" + str(l - (k + 1) + 1) + "]", "weights[" + str(l) + "]")
for m in range(l + 1, num_knot_points - 1 + l):
    adjust_parameter("trapezoidal_collocation_constraints.py", "velocity[" + str(m - (l + 1) + 1) + "]", "weights[" + str(m) + "]")
for n in range(m + 1, num_knot_points - 1 + m):
    adjust_parameter("trapezoidal_collocation_constraints.py", "angle_speed[" + str(n - (m + 1) + 1) + "]", "weights[" + str(n) + "]")
for o in range(n + 1, num_knot_points + n + 1):
    adjust_parameter("trapezoidal_collocation_constraints.py", "control[" + str(o - n - 1) + "]", "weights[" + str(o) + "]")
for p in range(o + 1, num_knot_points + o + 1):
    adjust_parameter("trapezoidal_collocation_constraints.py", "control[" + str(p - o - 1) + "]", "weights[" + str(p) + "]")

#adjust_parameter("trapezoidal_collocation_constraints.py", "control", "weights")
define_cost_function(num_defined_values)
define_weights_slacks_lagrange_multipliers(num_knot_points, num_defined_values)
