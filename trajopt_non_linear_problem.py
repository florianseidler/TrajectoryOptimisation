import jax.numpy as jnp
from jax import random
from nlp_jax_ipm import solve


# constants
damping_norm = 0.1
gravity_norm = 1


def angle_acceleration(control, angle, angle_speed):
    # from dynamics
    return (- damping_norm * angle_speed -
            gravity_norm * jnp.sin(angle) + control)


# define cost function
def cost_function(weights):
	return (weights[46] ** 2 + weights[47] ** 2 + weights[48] ** 2 + weights[49] ** 2 + weights[50] ** 2 + weights[51] ** 2 + weights[52] ** 2 + weights[53] ** 2 + weights[54] ** 2 + weights[55] ** 2 + weights[56] ** 2 + weights[57] ** 2 + weights[58] ** 2 + weights[59] ** 2 + weights[60] ** 2 + weights[61] ** 2 + weights[62] ** 2 + weights[63] ** 2 + weights[64] ** 2 + weights[65] ** 2 + weights[66] ** 2 + weights[67] ** 2 + weights[68] ** 2 + weights[69] ** 2 + weights[70] ** 2)


# trapezoidal constraints
def equality_constraint_angle_1(weights):
	return weights[0] - angle_start - 0.5 * timestep * (angle_speed_start + weights[23])


def equality_constraint_angle_speed_1(weights):
	return weights[23] - angle_speed_start - 0.5 * timestep * (angle_acceleration(weights[46], angle_start, angle_speed_start) + angle_acceleration(weights[47], weights[0], weights[23]))


def equality_constraint_angle_2(weights):
	return weights[1] - weights[0] - 0.5 * timestep * (weights[23] + weights[24])


def equality_constraint_angle_speed_2(weights):
	return weights[24] - weights[23] - 0.5 * timestep * (angle_acceleration(weights[47], weights[0], weights[23]) + angle_acceleration(weights[48], weights[1], weights[24]))


def equality_constraint_angle_3(weights):
	return weights[2] - weights[1] - 0.5 * timestep * (weights[24] + weights[25])


def equality_constraint_angle_speed_3(weights):
	return weights[25] - weights[24] - 0.5 * timestep * (angle_acceleration(weights[48], weights[1], weights[24]) + angle_acceleration(weights[49], weights[2], weights[25]))


def equality_constraint_angle_4(weights):
	return weights[3] - weights[2] - 0.5 * timestep * (weights[25] + weights[26])


def equality_constraint_angle_speed_4(weights):
	return weights[26] - weights[25] - 0.5 * timestep * (angle_acceleration(weights[49], weights[2], weights[25]) + angle_acceleration(weights[50], weights[3], weights[26]))


def equality_constraint_angle_5(weights):
	return weights[4] - weights[3] - 0.5 * timestep * (weights[26] + weights[27])


def equality_constraint_angle_speed_5(weights):
	return weights[27] - weights[26] - 0.5 * timestep * (angle_acceleration(weights[50], weights[3], weights[26]) + angle_acceleration(weights[51], weights[4], weights[27]))


def equality_constraint_angle_6(weights):
	return weights[5] - weights[4] - 0.5 * timestep * (weights[27] + weights[28])


def equality_constraint_angle_speed_6(weights):
	return weights[28] - weights[27] - 0.5 * timestep * (angle_acceleration(weights[51], weights[4], weights[27]) + angle_acceleration(weights[52], weights[5], weights[28]))


def equality_constraint_angle_7(weights):
	return weights[6] - weights[5] - 0.5 * timestep * (weights[28] + weights[29])


def equality_constraint_angle_speed_7(weights):
	return weights[29] - weights[28] - 0.5 * timestep * (angle_acceleration(weights[52], weights[5], weights[28]) + angle_acceleration(weights[53], weights[6], weights[29]))


def equality_constraint_angle_8(weights):
	return weights[7] - weights[6] - 0.5 * timestep * (weights[29] + weights[30])


def equality_constraint_angle_speed_8(weights):
	return weights[30] - weights[29] - 0.5 * timestep * (angle_acceleration(weights[53], weights[6], weights[29]) + angle_acceleration(weights[54], weights[7], weights[30]))


def equality_constraint_angle_9(weights):
	return weights[8] - weights[7] - 0.5 * timestep * (weights[30] + weights[31])


def equality_constraint_angle_speed_9(weights):
	return weights[31] - weights[30] - 0.5 * timestep * (angle_acceleration(weights[54], weights[7], weights[30]) + angle_acceleration(weights[55], weights[8], weights[31]))


def equality_constraint_angle_10(weights):
	return weights[9] - weights[8] - 0.5 * timestep * (weights[31] + weights[32])


def equality_constraint_angle_speed_10(weights):
	return weights[32] - weights[31] - 0.5 * timestep * (angle_acceleration(weights[55], weights[8], weights[31]) + angle_acceleration(weights[56], weights[9], weights[32]))


def equality_constraint_angle_11(weights):
	return weights[10] - weights[9] - 0.5 * timestep * (weights[32] + weights[33])


def equality_constraint_angle_speed_11(weights):
	return weights[33] - weights[32] - 0.5 * timestep * (angle_acceleration(weights[56], weights[9], weights[32]) + angle_acceleration(weights[57], weights[10], weights[33]))


def equality_constraint_angle_12(weights):
	return weights[11] - weights[10] - 0.5 * timestep * (weights[33] + weights[34])


def equality_constraint_angle_speed_12(weights):
	return weights[34] - weights[33] - 0.5 * timestep * (angle_acceleration(weights[57], weights[10], weights[33]) + angle_acceleration(weights[58], weights[11], weights[34]))


def equality_constraint_angle_13(weights):
	return weights[12] - weights[11] - 0.5 * timestep * (weights[34] + weights[35])


def equality_constraint_angle_speed_13(weights):
	return weights[35] - weights[34] - 0.5 * timestep * (angle_acceleration(weights[58], weights[11], weights[34]) + angle_acceleration(weights[59], weights[12], weights[35]))


def equality_constraint_angle_14(weights):
	return weights[13] - weights[12] - 0.5 * timestep * (weights[35] + weights[36])


def equality_constraint_angle_speed_14(weights):
	return weights[36] - weights[35] - 0.5 * timestep * (angle_acceleration(weights[59], weights[12], weights[35]) + angle_acceleration(weights[60], weights[13], weights[36]))


def equality_constraint_angle_15(weights):
	return weights[14] - weights[13] - 0.5 * timestep * (weights[36] + weights[37])


def equality_constraint_angle_speed_15(weights):
	return weights[37] - weights[36] - 0.5 * timestep * (angle_acceleration(weights[60], weights[13], weights[36]) + angle_acceleration(weights[61], weights[14], weights[37]))


def equality_constraint_angle_16(weights):
	return weights[15] - weights[14] - 0.5 * timestep * (weights[37] + weights[38])


def equality_constraint_angle_speed_16(weights):
	return weights[38] - weights[37] - 0.5 * timestep * (angle_acceleration(weights[61], weights[14], weights[37]) + angle_acceleration(weights[62], weights[15], weights[38]))


def equality_constraint_angle_17(weights):
	return weights[16] - weights[15] - 0.5 * timestep * (weights[38] + weights[39])


def equality_constraint_angle_speed_17(weights):
	return weights[39] - weights[38] - 0.5 * timestep * (angle_acceleration(weights[62], weights[15], weights[38]) + angle_acceleration(weights[63], weights[16], weights[39]))


def equality_constraint_angle_18(weights):
	return weights[17] - weights[16] - 0.5 * timestep * (weights[39] + weights[40])


def equality_constraint_angle_speed_18(weights):
	return weights[40] - weights[39] - 0.5 * timestep * (angle_acceleration(weights[63], weights[16], weights[39]) + angle_acceleration(weights[64], weights[17], weights[40]))


def equality_constraint_angle_19(weights):
	return weights[18] - weights[17] - 0.5 * timestep * (weights[40] + weights[41])


def equality_constraint_angle_speed_19(weights):
	return weights[41] - weights[40] - 0.5 * timestep * (angle_acceleration(weights[64], weights[17], weights[40]) + angle_acceleration(weights[65], weights[18], weights[41]))


def equality_constraint_angle_20(weights):
	return weights[19] - weights[18] - 0.5 * timestep * (weights[41] + weights[42])


def equality_constraint_angle_speed_20(weights):
	return weights[42] - weights[41] - 0.5 * timestep * (angle_acceleration(weights[65], weights[18], weights[41]) + angle_acceleration(weights[66], weights[19], weights[42]))


def equality_constraint_angle_21(weights):
	return weights[20] - weights[19] - 0.5 * timestep * (weights[42] + weights[43])


def equality_constraint_angle_speed_21(weights):
	return weights[43] - weights[42] - 0.5 * timestep * (angle_acceleration(weights[66], weights[19], weights[42]) + angle_acceleration(weights[67], weights[20], weights[43]))


def equality_constraint_angle_22(weights):
	return weights[21] - weights[20] - 0.5 * timestep * (weights[43] + weights[44])


def equality_constraint_angle_speed_22(weights):
	return weights[44] - weights[43] - 0.5 * timestep * (angle_acceleration(weights[67], weights[20], weights[43]) + angle_acceleration(weights[68], weights[21], weights[44]))


def equality_constraint_angle_23(weights):
	return weights[22] - weights[21] - 0.5 * timestep * (weights[44] + weights[45])


def equality_constraint_angle_speed_23(weights):
	return weights[45] - weights[44] - 0.5 * timestep * (angle_acceleration(weights[68], weights[21], weights[44]) + angle_acceleration(weights[69], weights[22], weights[45]))


def equality_constraint_angle_24(weights):
	return angle_end - weights[22] - 0.5 * timestep * (weights[45] + angle_speed_end)


def equality_constraint_angle_speed_24(weights):
	return angle_speed_end - weights[45] - 0.5 * timestep * (angle_acceleration(weights[69], weights[22], weights[45]) + angle_acceleration(weights[70], angle_end, angle_speed_end))


equality_constraints = [equality_constraint_angle_1, equality_constraint_angle_speed_1, equality_constraint_angle_2, equality_constraint_angle_speed_2, equality_constraint_angle_3, equality_constraint_angle_speed_3, equality_constraint_angle_4, equality_constraint_angle_speed_4, equality_constraint_angle_5, equality_constraint_angle_speed_5, equality_constraint_angle_6, equality_constraint_angle_speed_6, equality_constraint_angle_7, equality_constraint_angle_speed_7, equality_constraint_angle_8, equality_constraint_angle_speed_8, equality_constraint_angle_9, equality_constraint_angle_speed_9, equality_constraint_angle_10, equality_constraint_angle_speed_10, equality_constraint_angle_11, equality_constraint_angle_speed_11, equality_constraint_angle_12, equality_constraint_angle_speed_12, equality_constraint_angle_13, equality_constraint_angle_speed_13, equality_constraint_angle_14, equality_constraint_angle_speed_14, equality_constraint_angle_15, equality_constraint_angle_speed_15, equality_constraint_angle_16, equality_constraint_angle_speed_16, equality_constraint_angle_17, equality_constraint_angle_speed_17, equality_constraint_angle_18, equality_constraint_angle_speed_18, equality_constraint_angle_19, equality_constraint_angle_speed_19, equality_constraint_angle_20, equality_constraint_angle_speed_20, equality_constraint_angle_21, equality_constraint_angle_speed_21, equality_constraint_angle_22, equality_constraint_angle_speed_22, equality_constraint_angle_23, equality_constraint_angle_speed_23, equality_constraint_angle_24, equality_constraint_angle_speed_24]


# limit constraints
def inequality_constraint_1(weights):
	return weights[46] - -5


def inequality_constraint_2(weights):
	return weights[47] - -5


def inequality_constraint_3(weights):
	return weights[48] - -5


def inequality_constraint_4(weights):
	return weights[49] - -5


def inequality_constraint_5(weights):
	return weights[50] - -5


def inequality_constraint_6(weights):
	return weights[51] - -5


def inequality_constraint_7(weights):
	return weights[52] - -5


def inequality_constraint_8(weights):
	return weights[53] - -5


def inequality_constraint_9(weights):
	return weights[54] - -5


def inequality_constraint_10(weights):
	return weights[55] - -5


def inequality_constraint_11(weights):
	return weights[56] - -5


def inequality_constraint_12(weights):
	return weights[57] - -5


def inequality_constraint_13(weights):
	return weights[58] - -5


def inequality_constraint_14(weights):
	return weights[59] - -5


def inequality_constraint_15(weights):
	return weights[60] - -5


def inequality_constraint_16(weights):
	return weights[61] - -5


def inequality_constraint_17(weights):
	return weights[62] - -5


def inequality_constraint_18(weights):
	return weights[63] - -5


def inequality_constraint_19(weights):
	return weights[64] - -5


def inequality_constraint_20(weights):
	return weights[65] - -5


def inequality_constraint_21(weights):
	return weights[66] - -5


def inequality_constraint_22(weights):
	return weights[67] - -5


def inequality_constraint_23(weights):
	return weights[68] - -5


def inequality_constraint_24(weights):
	return weights[69] - -5


def inequality_constraint_25(weights):
	return weights[70] - -5


def inequality_constraint_26(weights):
	return - weights[46] + 5


def inequality_constraint_27(weights):
	return - weights[47] + 5


def inequality_constraint_28(weights):
	return - weights[48] + 5


def inequality_constraint_29(weights):
	return - weights[49] + 5


def inequality_constraint_30(weights):
	return - weights[50] + 5


def inequality_constraint_31(weights):
	return - weights[51] + 5


def inequality_constraint_32(weights):
	return - weights[52] + 5


def inequality_constraint_33(weights):
	return - weights[53] + 5


def inequality_constraint_34(weights):
	return - weights[54] + 5


def inequality_constraint_35(weights):
	return - weights[55] + 5


def inequality_constraint_36(weights):
	return - weights[56] + 5


def inequality_constraint_37(weights):
	return - weights[57] + 5


def inequality_constraint_38(weights):
	return - weights[58] + 5


def inequality_constraint_39(weights):
	return - weights[59] + 5


def inequality_constraint_40(weights):
	return - weights[60] + 5


def inequality_constraint_41(weights):
	return - weights[61] + 5


def inequality_constraint_42(weights):
	return - weights[62] + 5


def inequality_constraint_43(weights):
	return - weights[63] + 5


def inequality_constraint_44(weights):
	return - weights[64] + 5


def inequality_constraint_45(weights):
	return - weights[65] + 5


def inequality_constraint_46(weights):
	return - weights[66] + 5


def inequality_constraint_47(weights):
	return - weights[67] + 5


def inequality_constraint_48(weights):
	return - weights[68] + 5


def inequality_constraint_49(weights):
	return - weights[69] + 5


def inequality_constraint_50(weights):
	return - weights[70] + 5


inequality_constraints = [inequality_constraint_1, inequality_constraint_2, inequality_constraint_3, inequality_constraint_4, inequality_constraint_5, inequality_constraint_6, inequality_constraint_7, inequality_constraint_8, inequality_constraint_9, inequality_constraint_10, inequality_constraint_11, inequality_constraint_12, inequality_constraint_13, inequality_constraint_14, inequality_constraint_15, inequality_constraint_16, inequality_constraint_17, inequality_constraint_18, inequality_constraint_19, inequality_constraint_20, inequality_constraint_21, inequality_constraint_22, inequality_constraint_23, inequality_constraint_24, inequality_constraint_25, inequality_constraint_26, inequality_constraint_27, inequality_constraint_28, inequality_constraint_29, inequality_constraint_30, inequality_constraint_31, inequality_constraint_32, inequality_constraint_33, inequality_constraint_34, inequality_constraint_35, inequality_constraint_36, inequality_constraint_37, inequality_constraint_38, inequality_constraint_39, inequality_constraint_40, inequality_constraint_41, inequality_constraint_42, inequality_constraint_43, inequality_constraint_44, inequality_constraint_45, inequality_constraint_46, inequality_constraint_47, inequality_constraint_48, inequality_constraint_49, inequality_constraint_50]

num_knot_points = 25
timestep = 0.1
angle_start = 0
angle_speed_start = 0
angle_end = jnp.pi
angle_speed_end = 0

# weights, slacks, multipliers
number_weights = 71
key = random.PRNGKey(1702)
weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
number_slacks = jnp.size(inequality_constraints)
slacks = random.normal(key, shape=(number_slacks,)).astype(jnp.float32)
number_lagrange_multipliers = jnp.size(equality_constraints) + jnp.size(inequality_constraints)
lagrange_multipliers = random.normal(key, shape=(number_lagrange_multipliers,)).astype(jnp.float32)

# call solver
(weights, slacks, lagrange_multipliers, function_values, kkt_weights, kkt_slacks, kkt_equality_lagrange_multipliers, kkt_inequality_lagrange_multipliers) = (
solve(cost_function, equality_constraints, inequality_constraints, weights, slacks, lagrange_multipliers, num_inner_iterations=8, num_outer_iterations=5))

angle = weights[:22]
angle_speed = weights[22:45]
control = weights[45:]
