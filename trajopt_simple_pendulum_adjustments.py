from trajopt_simple_pendulum_generate_functions import (
    adjust_parameter,
    define_value, define_cost_function,
    define_weights_slacks_lagrange_multipliers, copy_file_line_by_line,
    define_equality_constraints, define_inequality_constraints, call_solver,
    split_results)

time = 2.5
num_knot_points = 25
timestep = time / num_knot_points
num_defined_values = 4

copy_file_line_by_line(targetfile="trajopt_example_simple_pendulum.py")

define_cost_function(num_knot_points, num_defined_values)

define_equality_constraints(num_knot_points=num_knot_points)
define_inequality_constraints(num_knot_points=num_knot_points, control_min=-5, control_max=5)

define_value("num_knot_points", num_knot_points)
define_value("timestep", timestep)
define_value("angle_start", 0)
define_value("angle_speed_start", 0)
define_value("angle_end", "jnp.pi")
define_value("angle_speed_end", 0)

adjust_parameter("trajopt_non_linear_problem.py", "angle[0]", "angle_start")
adjust_parameter("trajopt_non_linear_problem.py", "angle_speed[0]", "angle_speed_start")
adjust_parameter("trajopt_non_linear_problem.py", "angle[" + str(num_knot_points - 1) + "]", "angle_end")
adjust_parameter("trajopt_non_linear_problem.py", "angle_speed[" + str(num_knot_points - 1) + "]", "angle_speed_end")

#weights = jnp.concatenate([position[1:-1], angle[1:-1], vecolcity[1:-1], angle_speed[1:-1], control, angle_accel])

for k in range(0, num_knot_points - 2):
    adjust_parameter("trajopt_non_linear_problem.py", "angle[" + str(k + 1) + "]", "weights[" + str(k) + "]")
for l in range(k + 1, num_knot_points - 1 + k):
    adjust_parameter("trajopt_non_linear_problem.py", "angle_speed[" + str(l - (k + 1) + 1) + "]", "weights[" + str(l) + "]")
for m in range(l + 1, num_knot_points + l + 1):
    adjust_parameter("trajopt_non_linear_problem.py", "control[" + str(m - l - 1) + "]", "weights[" + str(m) + "]")


#adjust_parameter("trapezoidal_collocation_constraints.py", "control", "weights")

define_weights_slacks_lagrange_multipliers(num_knot_points, num_defined_values)

call_solver()
split_results(k, l)


