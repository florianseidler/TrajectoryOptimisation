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
