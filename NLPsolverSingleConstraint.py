import numpy as np
import jax.numpy as jnp
from jax import grad


# goal: minimize costfunction to calc the smallest
# torque-integral needed for trajectory optimisation

# questions:
# integrationconstant, numerical, boundaries

# TODO: check if costfunction is convex
# TODO: programm costfunction as integration of torquefunction
# TODO: boundaries t0=starttime, tf=endtime, x0=startplace, xf=endplace

# example:
# f(x) = 3 * x² + 2-> min       # template: f(x) = amplitude[0] * x^order + offset
# x <= 4                        # template: x <= constantOfIneqConstraint


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


def nullfunction(exp, ampli, offs):
    return (offs) / ampli[exp - 1]


def thirdconstraintparameter(ampli, exp, offs):
    return nullfunction(ampli, exp, offs)                                           # grad(f) + 0 = 0 = torquefct


def thirdconstraintineqvar(parameter, exp, ampli, offs):                            # calculate value for ineqvar
    return jnp.linalg.solve(1, grad(torquefunction)(parameter, exp, ampli, offs))


def checkinequalityconstraint(kktpoint, constOfIneqConstraint):                     # 2. constraint function
    if(kktpoint > constOfIneqConstraint):
        return 0
    return 1
