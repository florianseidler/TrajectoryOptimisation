def define_equality_constraints(num_knot_points):
    file = open('trajopt_non_linear_problem.py', "a")
    replacement = ""
    replacement = replacement +  "# trapezoidal constraints\n"
    for k in range(1, num_knot_points):
        angle_constr = generate_single_trapezoidal_constraint(
            "equality_constraint", "angle", "angle_speed",
            "weights", k)
        replacement = replacement + (angle_constr + "\n\n")
        angle_speed_constr = (
                "def equality_constraint_angle_speed_" + str(k) +
                "(weights):\n"
                "\treturn angle_speed[" + str(k) + "] - "
                                                   "angle_speed[" + str(
            k - 1) +
                "] - 0.5 * timestep * (angle_acceleration(control["
                + str(k - 1) + "], angle[" + str(k - 1) + "], "
                                                          "angle_speed[" + str(
            k - 1) + "]) + "
                     "angle_acceleration(control[" + str(
            k) + "], angle["
                + str(k) + "], angle_speed[" + str(k) + "]))\n")
        replacement = replacement + (angle_speed_constr + "\n\n")
    replacement = replacement + "equality_constraints = ["
    for k in range(1, num_knot_points):
        if k != 1:
            replacement = replacement + ", "
        replacement = replacement + ("equality_constraint_angle_" +
                                     str(k))
        replacement = replacement + (
                ", equality_constraint_angle_speed_" + str(k))
    replacement = replacement + "]\n\n"
    file.write(replacement)
    file.close()
    return


def define_inequality_constraints(num_knot_points, control_min, control_max):
    file = open('trajopt_non_linear_problem.py', "a")
    replacement = ""
    replacement = replacement + "\n# limit constraints\n"
    for k in range(1, num_knot_points + 1):
        min_constr = ("def inequality_constraint_" + str(
            k) + "(weights):\n"
                 "\treturn control[" + str(
            k - 1) + "] - " + str(control_min) + "\n")
        replacement = replacement + (min_constr + "\n\n")
    for l in range(k + 1, num_knot_points + k + 1):
        max_constr = ("def inequality_constraint_" + str(
            l) + "(weights):\n"
                 "\treturn - control[" + str(
            l - 1 - k) + "] + " + str(control_max) + "\n")
        replacement = replacement + (max_constr + "\n\n")
    replacement = replacement + "inequality_constraints = ["
    for m in range(1, num_knot_points + k + 1):
        if m != 1:
            replacement = replacement + (", ")
        replacement = replacement + ("inequality_constraint_" +
                                     str(m))
    replacement = replacement + ("]\n\n")
    file.write(replacement)
    file.close()
    return


def generate_single_trapezoidal_constraint(constraint_type, var_1, var_2, args,
                                           k):
    return ("def " + constraint_type + "_" + var_1 + "_" + str(k) +
            "(" + args + "):\n\treturn angle[" + str(k) + "] - " + var_1 + "["
            + str(k - 1) + "] - 0.5 * timestep * (" + var_2 + "[" + str(k - 1)
            + "] + " + var_2 + "[" + str(k) + "])\n")


def define_value(parameter, parameter_value):
    file = open('trajopt_non_linear_problem.py', "a")
    define = (parameter + " = " + str(parameter_value) + "\n")
    file.write(define)
    file.close()
    return


def copy_file_line_by_line(targetfile):
    file = open(targetfile, "r")
    content = ""
    for line in file:
        content = content + line
    file.close()
    file = open('trajopt_non_linear_problem.py', "a+")
    libraries = (
        "import jax.numpy as jnp\n"
        "from jax import random\n"
        "from nlp_jax_ipm import solve\n\n\n")
    file.write(libraries)
    file.write(content)
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


def define_cost_function(num_knot_points, num_defined_values):
    file = open('trajopt_non_linear_problem.py', "a")
    offset = 2 * num_knot_points - num_defined_values
    weight_string = ""
    for iter in range(num_knot_points):
        if iter != 0:
            weight_string = (weight_string + " + ")
        weight_string = (weight_string + "weights[" +
                         str(iter + offset) + "] ** 2")
    cost_function = ("\n# define cost function\n"
                     "def cost_function(weights):\n"
                     "\treturn (" + weight_string + ")\n\n\n")
    file.write(cost_function)
    file.close()
    return


def define_weights_slacks_lagrange_multipliers(num_knot_points,
                                               num_defined_values):

    file = open('trajopt_non_linear_problem.py', "a")
    variables = ("\n# weights, slacks, multipliers\n"
        "number_weights = " + str(num_knot_points * 3 -
                                  num_defined_values) + "\n"
        "key = random.PRNGKey(1702)\n"
        "weights = random.normal(key, shape=(number_weights,))"
        ".astype(jnp.float32)\n"
        "number_slacks = jnp.size(inequality_constraints)\n"                                                
        "slacks = random.normal(key, shape=(number_slacks,))"
        ".astype(jnp.float32)\n"
        "number_lagrange_multipliers = jnp.size(equality_constraints) "
        "+ jnp.size(inequality_constraints)\n" 
        "lagrange_multipliers = random.normal(key, "
        "shape=(number_lagrange_multipliers,)).astype(jnp.float32)\n")
    file.write(variables)
    file.close()
    return


def call_solver():
    file = open('trajopt_non_linear_problem.py', "a")
    call = ("\n# call solver\n"
            "(weights, slacks, lagrange_multipliers, function_values, "
            "kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, "
            "kkt_inequality_lagrange_multipliers) = (\nsolve(cost_function, "
            "equality_constraints, inequality_constraints, weights, slacks, "
            "lagrange_multipliers, num_inner_iterations=8, "
            "num_outer_iterations=5))\n")
    file.write(call)
    file.close()
    return


def split_results(idx_angle, idx_angle_speed):
    file = open('trajopt_non_linear_problem.py', "a")
    splitted_results = (
        "\nangle = weights[:" + str(idx_angle) + "]\n"
        "angle_speed = weights[" + str(idx_angle) + ":" + str(idx_angle_speed) + "]\n"
        "control = weights[" + str(idx_angle_speed) + ":]\n")
    file.write(splitted_results)
    file.close()
    return


def define_libraries():
    file = open('trajopt_non_linear_problem.py', "a")
    libraries = (
        "import jax.numpy as jnp\n"
        "from jax import random\n"
        "from nlp_jax_ipm import solve\n\n\n")
    file.write(libraries)
    file.close()
    return

