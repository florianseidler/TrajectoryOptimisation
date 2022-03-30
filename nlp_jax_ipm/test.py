import jax.numpy as jnp
from jax import random
from solver import solve

"""
Adjust prob to test different cases:
1: optimization with an equality-constraint
2: optimization with inequality-constraints
3: optimization with equality-constraints and inequality-constraints
4: optimization without constraints
5: optimization of 2D-Rosenbrock Matrix
"""

prob = 6

if prob == 1:

    def cost_function(weights):
        return jnp.asarray(-weights[0] - weights[1])

    def equality_constr_1(weights):
        return jnp.asarray(weights[0]**2 + weights[1]**2 - 1)

    number_weights = 2
    number_equality_constraints = 1
    number_inequality_constraints = 0
    equality_constraints = [equality_constr_1]
    inequality_constraints = []
    weights = jnp.array([1., 2.])
    slacks = jnp.array([])
    lagrange_multipliers = jnp.array([1.])

if prob == 2:

    def cost_function(weights):
        return (weights[0]**2 + 2 * weights[1]**2 + 2 * weights[0] +
                8 * weights[1])

    def inequality_constraints_1(weights):
        return weights[0] + 2 * weights[1] - 10

    def inequality_constraints_2(weights):
        return weights[0]

    def inequality_constraints_3(weights):
        return weights[1]

    number_weights = 2
    number_equality_constraints = 0
    number_inequality_constraints = 3
    equality_constraints = []
    inequality_constraints = [inequality_constraints_1,
                              inequality_constraints_2,
                              inequality_constraints_3]
    weights = jnp.array([1., 2.])
    slacks = jnp.array([1., 1., 1.])
    lagrange_multipliers = jnp.array([1., 2., 3.])

if prob == 3:

    def cost_function(weights):
        return jnp.asarray(- weights[0] * jnp.log(weights[0])
                           - weights[1] * jnp.log(weights[1])
                           - weights[2] * jnp.log(weights[2])
                           - weights[3] * jnp.log(weights[3])
                           - weights[4] * jnp.log(weights[4])
                           - weights[5] * jnp.log(weights[5]))

    def equality_constr_1(weights):
        return jnp.asarray(weights[0] + weights[1] + weights[2] +
                           weights[3] + weights[4] + weights[5] - 1)

    def inequality_constraints_1(weights):
        return jnp.asarray(weights[0])

    def inequality_constraints_2(weights):
        return jnp.asarray(weights[1])

    def inequality_constraints_3(weights):
        return jnp.asarray(weights[2])

    def inequality_constraints_4(weights):
        return jnp.asarray(weights[3])

    def inequality_constraints_5(weights):
        return jnp.asarray(weights[4])

    def inequality_constraints_6(weights):
        return jnp.asarray(weights[5])

    number_weights = 6
    number_equality_constraints = 1
    number_inequality_constraints = 6
    equality_constraints = [equality_constr_1]
    inequality_constraints = [inequality_constraints_1,
                              inequality_constraints_2,
                              inequality_constraints_3,
                              inequality_constraints_4,
                              inequality_constraints_5,
                              inequality_constraints_6]
    weights = jnp.array([1., 1., 1., 1., 1., 1.])
    slacks = jnp.array([1., 1., 1., 1., 1., 1.])
    lagrange_multipliers = jnp.array([1., 1., 1., 1., 1., 1., 1.])

if prob == 4:

    def cost_function(weights):
        return jnp.asarray(weights[0] ** 2 - 4 * weights[0]
                           + (weights[1] ** 2 - weights[1])
                           - (weights[0] * weights[1]))

    number_weights = 2
    number_equality_constraints = 0
    number_inequality_constraints = 0
    equality_constraints = []
    inequality_constraints = []
    weights = jnp.array([1., 2.])
    slacks = None
    lagrange_multipliers = None

if prob == 5:

    def cost_function(weights):
        return jnp.asarray(100 * (weights[1] - weights[0] ** 2) ** 2
                           + (1 - weights[0]) ** 2)

    number_weights = 2
    number_equality_constraints = 0
    number_inequality_constraints = 0
    equality_constraints = []
    inequality_constraints = []
    key = random.PRNGKey(1702)
    weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
    slacks = None
    lagrange_multipliers = None

if prob == 6:

    def cost_function(weights):
        return jnp.asarray((weights[0] - 1) ** 2
                           + 2 * (weights[1] + 2) ** 2
                           + 3 * (weights[2] + 3) ** 2)

    def equality_constr_1(weights):
        return jnp.asarray(weights[2] - weights[1] - weights[0] - 1.0)

    def inequality_constraints_1(weights):
        return jnp.asarray(weights[2] - weights[0] ** 2)

    number_weights = 3
    number_equality_constraints = 1
    number_inequality_constraints = 1
    equality_constraints = [equality_constr_1]
    inequality_constraints = [inequality_constraints_1]
    key = random.PRNGKey(1702)
    weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
    slacks = jnp.array([1.0])
    lagrange_multipliers = jnp.array([1.0, 2.0])

(weights, slacks, lagrange_multipliers, function_values, kkt_weights,
 kkt_slacks, kkt_equality_lagrange_multipliers,
 kkt_inequality_lagrange_multipliers) = (
    solve(cost_function, equality_constraints,
          inequality_constraints, weights, slacks, lagrange_multipliers))

if prob == 1:
    print('Ground Truth: [0.7071 0.7071]')
elif prob == 2:
    print('Ground Truth: [4.000 3.000]')
elif prob == 3:
    print('Ground Truth: [0.16666667 0.16666667 0.16666667 '
          '0.16666667 0.16666667 0.16666667 ]')
elif prob == 4:
    print('Ground Truth: [3.000 2.000]')
elif prob == 5:
    print('Ground Truth: [1.000 1.000]')
elif prob == 6:
    print('Ground Truth: [0.12288 -1.1078 0.0151]')

print('---------    APPROXIMATED RESULTS     ---------')
print('Weights: ', weights)
print('Slacks: ', slacks)
print('lagrange_multipliers: ', lagrange_multipliers)
print('function_values: ', function_values)
print('kkt_weights', kkt_weights)
print('kkt_slacks', kkt_slacks)
print('kkt_equality_lagrange_multipliers', kkt_equality_lagrange_multipliers)
print('kkt_inequality_lagrange_multipliers',
      kkt_inequality_lagrange_multipliers)
