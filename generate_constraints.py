def constraints_generator_orig(targetfile, num_knot_points, control_min,
                               control_max):
    fout = open('trapezoidal_collocation_constraints.py', 'w')
    fin = open(targetfile)

    for line in fin:
        fout.write(line)
        if line == '# trapezoidal constraints\n':
            next_line = next(fin)
            if next_line == '# end of trapezoidal constraints\n':
                for k in range(1, num_knot_points):
                    fout.write("def equality_constraint_position_" + str(
                        k) + "(position, velocity):\n")
                    fout.write("    return position[" +
                               str(k) + "] - position[" + str(k - 1) +
                               "] - 0.5 * timestep * (velocity[" +
                               str(k - 1) + "] + velocity[" + str(k) + "])\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_angle_" + str(
                        k) + "(angle, angle_speed):\n")
                    fout.write("    return angle[" + str(k) + "] - angle[" +
                               str(k - 1) + "] - 0.5 * timestep * (angle_speed["
                               + str(k - 1) + "] + angle_speed[" +
                               str(k) + "])\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_velocity_" +
                               str(k) + "(velocity, control, angle, "
                                        "angle_speed):\n")
                    fout.write("    return velocity[" + str(k) +
                               "] - velocity[" + str(k - 1) +
                               "] - 0.5 * timestep * (acceleration(control[" +
                               str(k - 1) + "], angle[" + str(k - 1) +
                               "], angle_speed[" + str(k - 1) +
                               "]) + acceleration(control[" + str(k) +
                               "], angle[" + str(k) + "], angle_speed[" +
                               str(k) + "]))\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_angle_speed_" + str(
                        k) + "(angle_speed, control, angle):\n")
                    fout.write("    return angle_speed[" + str(
                        k) + "] - angle_speed[" + str(k - 1) +
                               "] - 0.5 * timestep * (angle_acceleration("
                               "control[" + str(k - 1) + "], angle[" +
                               str(k - 1) + "], angle_speed[" +
                               str(k - 1) + "]) + angle_acceleration(control[" +
                               str(k) + "], angle[" +
                               str(k) + "], angle_speed[" +
                               str(k) + "]))\n")
                    fout.write("\n\n")
                fout.write("equality_constraints = [")
                for k in range(1, num_knot_points):
                    if k != 1:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("equality_constraint_position_" + str(k))
                    fout.write(", equality_constraint_angle_" + str(k))
                    fout.write(", equality_constraint_velocity_" + str(k))
                    fout.write(", equality_constraint_angle_speed_" + str(k))
                fout.write("]\n")  # \n\t\t\t\t\t\t
            fout.write(next_line)
        if line == '# limit constraints\n':
            next_line2 = next(fin)
            if next_line2 == '# end of limit constraints\n':
                for k in range(1, num_knot_points + 1):
                    fout.write("def inequality_constraint_" + str(
                        k) + "(control):\n")
                    fout.write("    return control[" + str(
                        k - 1) + "] - " + str(control_min) + "\n")
                    fout.write("\n\n")
                for l in range(k + 1, num_knot_points + k + 1):
                    fout.write("def inequality_constraint_" + str(
                        l) + "(control):\n")
                    fout.write("    return - control[" + str(
                        l - 1 - k) + "] + " + str(control_max) + "\n")
                    fout.write("\n\n")
                fout.write("inequality_constraints = [")
                for m in range(1, num_knot_points + k + 1):
                    if m != 1:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("inequality_constraint_" + str(m))
                fout.write("]\n\n")  # \n\t\t\t\t\t\t
            fout.write(next_line2)
    fin.close()
    fout.close()
    return


def constraints_generator(targetfile, num_knot_points, control_min,
                          control_max):
    fout = open('trapezoidal_collocation_constraints.py', 'w')
    fin = open(targetfile)

    for line in fin:
        fout.write(line)
        if line == '# trapezoidal constraints\n':
            next_line = next(fin)
            if next_line == '# end of trapezoidal constraints\n':
                for k in range(1, num_knot_points):
                    fout.write("def equality_constraint_position_" + str(
                        k) + "(weights):\n")
                    fout.write("    return position[" +
                               str(k) + "] - position[" + str(k - 1) +
                               "] - 0.5 * timestep * (velocity[" +
                               str(k - 1) + "] + velocity[" + str(k) + "])\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_angle_" + str(
                        k) + "(weights):\n")
                    fout.write("    return angle[" + str(k) + "] - angle[" +
                               str(k - 1) + "] - 0.5 * timestep * (angle_speed["
                               + str(k - 1) + "] + angle_speed[" +
                               str(k) + "])\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_velocity_" +
                               str(k) + "(weights):\n")
                    fout.write("    return velocity[" + str(k) +
                               "] - velocity[" + str(k - 1) +
                               "] - 0.5 * timestep * (acceleration(control[" +
                               str(k - 1) + "], angle[" + str(k - 1) +
                               "], angle_speed[" + str(k - 1) +
                               "]) + acceleration(control[" + str(k) +
                               "], angle[" + str(k) + "], angle_speed[" +
                               str(k) + "]))\n")
                    fout.write("\n\n")
                    fout.write("def equality_constraint_angle_speed_" + str(
                        k) + "(weights):\n")
                    fout.write("    return angle_speed[" + str(
                        k) + "] - angle_speed[" + str(k - 1) +
                               "] - 0.5 * timestep * (angle_acceleration("
                               "control[" + str(k - 1) + "], angle[" +
                               str(k - 1) + "], angle_speed[" +
                               str(k - 1) + "]) + angle_acceleration(control[" +
                               str(k) + "], angle[" +
                               str(k) + "], angle_speed[" +
                               str(k) + "]))\n")
                    fout.write("\n\n")
                fout.write("equality_constraints = [")
                for k in range(1, num_knot_points):
                    if k != 1:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("equality_constraint_position_" + str(k))
                    fout.write(", equality_constraint_angle_" + str(k))
                    fout.write(", equality_constraint_velocity_" + str(k))
                    fout.write(", equality_constraint_angle_speed_" + str(k))
                fout.write("]\n")  # \n\t\t\t\t\t\t
            fout.write(next_line)
        if line == '# limit constraints\n':
            next_line2 = next(fin)
            if next_line2 == '# end of limit constraints\n':
                for k in range(1, num_knot_points + 1):
                    fout.write("def inequality_constraint_" + str(
                        k) + "(weights):\n")
                    fout.write("    return control[" + str(
                        k - 1) + "] - " + str(control_min) + "\n")
                    fout.write("\n\n")
                for l in range(k + 1, num_knot_points + k + 1):
                    fout.write("def inequality_constraint_" + str(
                        l) + "(weights):\n")
                    fout.write("    return - control[" + str(
                        l - 1 - k) + "] + " + str(control_max) + "\n")
                    fout.write("\n\n")
                fout.write("inequality_constraints = [")
                for m in range(1, num_knot_points + k + 1):
                    if m != 1:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("inequality_constraint_" + str(m))
                fout.write("]\n\n")  # \n\t\t\t\t\t\t
            fout.write(next_line2)
    fin.close()
    fout.close()
    return


def define_value(parameter, parameter_value):
    file = open('trapezoidal_collocation_constraints.py', "r")
    replacement = ""
    for line in file:
        replacement = replacement + line
        if line == '# defined values\n':
            define = (parameter + " = " + str(parameter_value) + "\n")
            replacement = replacement + define
    file.close()
    file = open('trapezoidal_collocation_constraints.py', "w")
    file.write(replacement)
    file.close()
    return


def adjust_parameter(targetfile, parameter, renamed_parameter):
    file = open(targetfile, "r")
    replacement = ""
    for line in file:
        changes = line.replace(parameter, renamed_parameter)
        replacement = replacement + changes
    file.close()
    fout = open(targetfile, "w")
    fout.write(replacement)
    fout.close()
    return


def define_cost_function(num_defined_values):
    file = open('trapezoidal_collocation_constraints.py', "r")
    replacement = ""
    for line in file:
        replacement = replacement + line
        if line == '# define cost function\n':
            offset = num_defined_values
            cost_function = ("def cost_function(weights):\n"
                             "\tnum_knots = jnp.size(weights) - 1\n"
                             "\tcontrol_sum = 0\n" \
                             "\tfor iter in range(" + str(offset) +
                             ", num_knots + " + str(offset) + "):\n" 
                             "\t\tcontrol_sum += 0.5 * timestep * "
                             "(weights[iter] ** 2 + weights[iter + 1] ** 2)\n" 
                             "\treturn control_sum")
            replacement = replacement + cost_function
    file.close()
    file = open('trapezoidal_collocation_constraints.py', "w")
    file.write(replacement)
    file.close()
    return


def define_weights_slacks_lagrange_multipliers(num_knot_points,
                                               num_defined_values):

    file = open('trapezoidal_collocation_constraints.py', "r")
    replacement = ""
    for line in file:
        replacement = replacement + line
        if line == '# weights, slacks, multipliers\n':
            variables = (
                    "number_weights = " +
                     str(num_knot_points * 6 - num_defined_values) + "\n"
                     "key = random.PRNGKey(1702)\n"
                     "weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)\n"
                     "number_slacks = jnp.size(inequality_constraints)\n"                                                
                     "slacks = random.normal(key, shape=(number_slacks,)).astype(jnp.float32)\n"
                     "number_lagrange_multipliers = jnp.size(equality_constraints) + jnp.size(inequality_constraints)\n" 
                     "lagrange_multipliers = random.normal(key, shape=(number_lagrange_multipliers,)).astype(jnp.float32)\n")
            replacement = replacement + variables
    file.close()
    file = open('trapezoidal_collocation_constraints.py', "w")
    file.write(replacement)
    file.close()
    return
