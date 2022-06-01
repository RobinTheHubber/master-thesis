from marginal_engine.marginal_model import FilterEquation, MarginalObject
import numpy as np
from marginal_engine.distributions import gaussian, student_t, skewed_normal, skewed_t
from scipy.optimize import LinearConstraint as lin_constraint
from utility.mapping import *

constant_mean_equation = FilterEquation(
    update_function=lambda par, x, y: y,
    initial_values_function=lambda par: par,
    transformation_function=lambda x, backwards=None: x,
    get_initial_parameters_function=lambda data: [np.mean(data)],
    npar=1)

constant_vol_equation = FilterEquation(
    update_function=lambda par, x, y: y,
    initial_values_function=lambda par: par,
    transformation_function=lambda x, backwards=None: map_positive(x, backwards),
    get_initial_parameters_function=lambda data: [np.var(data)],
    npar=1)

ar1_equation = FilterEquation(
    update_function=lambda par, y, mu: par[0] + par[1] * y,
    initial_values_function=lambda par: par[0] / (1 - par[1]),  # mu0 := unconditional mean
    transformation_function=lambda x, backwards=None: [x[0], map_logistic(x[1], 0, 1, backwards)],
    get_initial_parameters_function=lambda data: [np.mean(data) * (1 - 0.85), 0.85],
    npar=2)

garch_11_equation = FilterEquation(
    update_function=lambda par, y, sig2: par[0] + par[1] * y ** 2 + par[2] * sig2,
    initial_values_function=lambda par: par[0] / (1 - par[1] - par[2]),  # sigma0 := unconditional variance
    transformation_function=lambda x, backwards=None: [map_positive(x[0], backwards),
                                                       x[1],
                                                       x[2]],
    get_initial_parameters_function=lambda data: [np.var(data) * (1 - 0.1 - 0.85), 0.1, 0.85],
    npar=3)

eq_cons_garch1 = {'type': 'ineq', 'fun': lambda x: x[-2] + x[-1]}
eq_cons_garch2 = {'type': 'ineq', 'fun': lambda x: -(x[-2] + x[-1]) + 1}
eq_cons_garch3 = {'type': 'ineq', 'fun': lambda x: x[-1]}
eq_cons_garch4 = {'type': 'ineq', 'fun': lambda x: x[-2]}
# eq_cons_garch5 = {'type':'ineq', 'fun': lambda x: -x[-1] + 1}
# eq_cons_garch6 = {'type':'ineq', 'fun': lambda x: -x[-2] + 1}


# import matplotlib.pyplot as plt
# plt.plot(garch_model.data)
# plt.hist(garch_model.PITs)
# print(garch_model.fit_result.message)
# print(garch_model.estimated_parameters)
# plt.show()
#

#
# M=1000
# m,v=[0]*M, [0]*M
# for i in range(M):
#     garch_model.simulate_from_garch(u)
#
#     m[i]=garch_model.data[-1]
#
# np.mean(m)
# np.var(m)
