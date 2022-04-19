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
                    fout.write("def equality_constraint_" + str(
                        k) + "(state, dynamics):\n")
                    fout.write("    return state[" + str(
                        k) + "] - state[" + str(k - 1) +
                               "] - 0.5 * timestep * (dynamics[" + str(
                        k - 1) + "] + "
                             "dynamics[" + str(k) + "])\n")
                    fout.write("\n\n")
                fout.write("equality_constraints = [")
                for k in range(num_knot_points - 1):
                    if k != 0:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("equality_constraint_" + str(k))
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
                        l - 1) + "] + " + str(control_max) + "\n")
                    fout.write("\n\n")
                fout.write("inequality_constraints = [")
                for m in range(1, num_knot_points + k + 1):
                    if m != 1:
                        fout.write(", ")  # \n\t\t\t\t\t\t
                    fout.write("inequality_constraint_" + str(m))
                fout.write("]\n")  # \n\t\t\t\t\t\t
            fout.write(next_line2)
    fin.close()
    fout.close()
    return
