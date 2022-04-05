import jax.numpy as jnp
from jax import random
from solver import solve

"""
Adjust prob to test different cases:
1: maximize with an equality-constraint
2: minimize with inequality-constraints
3: maximize with equality-constraints and inequality-constraints
4: minimize without constraints
5: minimize 2D-Rosenbrock Matrix
6: minimize with equality-constraints and inequality-constraints
7: maximize with equality-constraint
8: maximize with equality-constraints and inequality-constraints
9: minimize with equality-constraints
10:minimize with inequality-constraints
"""

prob = 10

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

    print('maximize f(x, y) = x + y subject to x**2 + y**2 = 1')

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

    print('minimize f(x, y) = x**2 + 2*y**2 + 2*x + 8*y '
          'subject to -x - 2*y + 10 <= 0, x >= 0, y >= 0')

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

    print('Find the maximum entropy distribution of a six-sided die:')
    print('maximize f(x) = -sum(x*log(x)) subject to sum(x) = 1 '
          'and x >= 0 (x.size == 6)')

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

    print('minimize f(x, y) = x**2 - 4*x + y**2 - y - x*y')

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

    print('minimize f(x, y) = 100*(y - x**2)**2 + (1 - x)**2')

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

    print('minimize f(x, y, z) = (x - 1)**2 + 2*(y + 2)**2 + 3*(z + 3)**2 '
          'subject to z - y - x = 1, z - x**2 >= 0')

if prob == 7:

    def cost_function(weights):
        return jnp.asarray(-(weights[0] ** 2) * weights[1])

    def equality_constr_1(weights):
        return jnp.asarray(weights[1] ** 2 + weights[0] ** 2 - 3.0)

    number_weights = 2
    number_equality_constraints = 1
    number_inequality_constraints = 0
    equality_constraints = [equality_constr_1]
    inequality_constraints = []
    key = random.PRNGKey(1702)
    weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
    slacks = None
    lagrange_multipliers = jnp.array([1.0])

    print('maximize f(x, y) = -(x**2)*y subject to x**2 + y**2 = 3')

if prob == 8:

    def cost_function(weights):
        return jnp.asarray(- weights[0] * weights[1] * weights[2])

    def equality_constr_1(weights):
        return jnp.asarray(weights[0] + weights[1] + weights[2] - 1.0)

    def inequality_constraints_1(weights):
        return jnp.asarray(weights[0])

    def inequality_constraints_2(weights):
        return jnp.asarray(weights[1])

    def inequality_constraints_3(weights):
        return jnp.asarray(weights[2])

    number_weights = 3
    number_equality_constraints = 1
    number_inequality_constraints = 3
    equality_constraints = [equality_constr_1]
    inequality_constraints = [inequality_constraints_1,
                              inequality_constraints_2,
                              inequality_constraints_3]
    key = random.PRNGKey(1702)
    weights = random.normal(key, shape=(number_weights,)).astype(jnp.float32)
    slacks = jnp.array([1., 1., 1.])
    lagrange_multipliers = jnp.array([1., 2., 3., 4.])

    print('maximize f(x, y, z) = x*y*z subject to '
          'x + y + z = 1, x >= 0, y >= 0, z >= 0')

if prob == 9:

    def cost_function(weights):
        return jnp.asarray(4 * weights[1] - 2 * weights[2])

    def equality_constr_1(weights):
        return jnp.asarray(2 * weights[0] - weights[1] - weights[2] - 2.)

    def equality_constr_2(weights):
        return jnp.asarray(weights[0]**2 + weights[1]**2 - 1.)

    number_weights = 3
    number_equality_constraints = 2
    number_inequality_constraints = 0
    equality_constraints = [equality_constr_1, equality_constr_2]
    inequality_constraints = []
    weights = jnp.array([1., 2., 3.])
    slacks = None
    lagrange_multipliers = jnp.array([1., 2.])

    print('minimize f(x,y,z) = 4*y - 2*z subject to 2*x - y - z = 2, '
          'x**2 + y**2 = 1')

if prob == 10:

    def cost_function(weights):
        return (weights[0] - 2) ** 2 + 2 * (weights[1] - 1) ** 2

    def inequality_constraints_1(weights):
        return - weights[0] - 4 * weights[1] + 3

    def inequality_constraints_2(weights):
        return weights[0] - weights[1]

    number_weights = 2
    number_equality_constraints = 0
    number_inequality_constraints = 2
    equality_constraints = []
    inequality_constraints = [inequality_constraints_1,
                              inequality_constraints_2]
    weights = jnp.array([1., 2.])
    slacks = jnp.array([1., 1.])
    lagrange_multipliers = jnp.array([1., 2.])

    print('minimize f(x, y) = (x - 2)**2 + 2*(y - 1)**2 '
          'subject to x + 4*y <= 3, x >= y')

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
elif prob == 7:
    print('Ground Truth: [1.41 1.00]')
elif prob == 8:
    print('Ground Truth: [0.333 0.333 0.333]')
elif prob == 9:
    print('Ground Truth: [0.5547 -0.8321 -0.0585]')
elif prob == 10:
    print('Ground Truth: [1.667 0.333]')

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
