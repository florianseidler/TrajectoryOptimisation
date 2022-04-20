from generate_constraints import (constraints_generator, adjust_parameter,
                                  define_value)

num_knot_points = 4
timestep = 2

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