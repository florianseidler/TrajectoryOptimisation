import numpy as np
import jax.numpy as jnp
from jax import grad


# goal: minimize costfunction to calc the smallest
# torque-integral needed for trajectory optimisation

# example:
# f(x) = 3 * x² + 2-> min       # template: f(x) = amplitude[0] * x^order + offset
# x <= 4                        # template: x <= constantOfIneqConstraint


# QUESTIONS: integrationconstant, numerical, boundaries

# TODO: check if costfunction is convex
# TODO: programm costfunction as integration of torquefunction
# TODO: boundaries t0=starttime, tf=endtime, x0=startplace, xf=endplace

# IDEA: Use integrationgaussquad5 to minimize torquefunction


def nlpsolver(order, amplitude, offset, constantOfIneqConstraint):

    ineqvar = 0                                                                     # 4. constraint passed
    kkt_x = thirdconstraintparameter(order, amplitude, offset)
    if (checkinequalityconstraint(kktpoint, constantOfIneqConstraint) == 1):        # 2. constraint
        return kkt_x

    kkt_x = -constantOfIneqConstraint
    ineqvar = thirdconstraintineqvar(kkt_x, order, amplitude, offset)
    if (ineqvar >= 1):                                                              # 4. constraint
        if (checkinequalityconstraint(kktpoint, constantOfIneqConstraint) == 1):    # 2. constraint
            return kkt_x


### ----- subfunctions ----- ###


def torquefunction(x, exp, ampli, offs):
    result = 0
    for i in range(exp, 0, -1):
        result += ampli[order - i] * x**i
    result += offs
    return result


def nullfunction(exp, ampli, offs):                                                 # Newton’s Method to approx zero
    iterations = 10
    initialguess = 1.4
    approximation = initialguess - (torquefunction(initialguess, exp, ampli, offs) / grad(torquefunction)(initialguess, exp, ampli, offs))
    for i in range(iterations):
        approximation = approximation - (torquefunction(approximation, exp, ampli, offs) / grad(torquefunction)(approximation, exp, ampli, offs))
    return approximation


def integrationgaussquad5(fct, lowerlimit, upperlimit):                             # gaussian 5-point quadrature
    x5point = np.array([-0.90618, -0.538469, 0, 0.538469, 0.90618])
    w5point = np.array([0.236927, 0.478629, 0.568889, 0.478629, 0.236927])
    val = (upperlimit - lowerlimit) * (x5point + 1) / 2.0 + lowerlimit
    return (upperlimit - lowerlimit) / 2.0 * np.sum(w5point * fct(val))


def thirdconstraintparameter(ampli, exp, offs):
    return nullfunction(ampli, exp, offs)                                           # grad(f) + 0 = 0 = torquefct


def thirdconstraintineqvar(parameter, exp, ampli, offs):                            # calculate value for ineqvar
    return jnp.linalg.solve(1, grad(torquefunction)(parameter, exp, ampli, offs))


def checkinequalityconstraint(kktpoint, constOfIneqConstraint):                     # 2. constraint function
    if(kktpoint > constOfIneqConstraint):
        return 0
    return 1
