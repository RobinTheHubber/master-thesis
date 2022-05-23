import numpy as np
import pyvinecopulib as pyv
from copy import deepcopy as dc
from scipy.stats import norm
from mapping import map_logistic
from garch_model import *
from simulate_vine_algorithm import get_vine_data, sample_from_dynamic_vine


def simulate_from_vine_copula(m, T, n, copula_type, distribution_marginal, cpar_equation=None):

    if copula_type == 'gaussian':
        list_parameters_tree1 = [[0.5], [0.75], [0.5], [0.25]]
        list_parameters_tree2 = [[0.2], [0.1], [-0.2]]
        list_parameters_tree3 = [[-0.5], [0.5]]
        list_parameters_tree4 = [[0.1]]

    if copula_type == 'student_t':
        list_parameters_tree1 = np.array([[0.5, 4], [0.75, 3], [0.5, 4], [0.25,4]]).reshape((4,2,1))
        list_parameters_tree2 = np.array([[0.2,3], [0.1,4], [-0.2,3]]).reshape((3,2,1))
        list_parameters_tree3 = np.array([[-0.5,4], [0.5,5]]).reshape((2,2,1))
        list_parameters_tree4 = np.array([[0.1,4]]).reshape((1, 2, 1))

    # rho13 = rho13_2 * np.sqrt(1-rho12**2) * np.sqrt(1-rho23**2) + rho12*rho23 # 0.2318
    # vine_structure = pyv.DVineStructure(order=np.arange(1, n+1))
    # list_copula_tree_1 = [pyv.Bicop(family=fam, parameters=list_parameters_tree1[i]) for i in range(4)]
    # list_copula_tree_2 = [pyv.Bicop(family=fam, parameters=list_parameters_tree2[i]) for i in range(3)]
    # list_copula_tree_3 = [pyv.Bicop(family=fam, parameters=list_parameters_tree3[i]) for i in range(2)]
    # list_copula_tree_4 = [pyv.Bicop(family=fam, parameters=list_parameters_tree4[i]) for i in range(1)]
    # vine = pyv.Vinecop(structure=vine_structure, pair_copulas=[list_copula_tree_1, list_copula_tree_2, list_copula_tree_3, list_copula_tree_4])
    # mU = vine.simulate(n=T, seeds=np.array([m+20])) # simulate dependent uniform random numbers from vine copula model

    np.random.seed(1991)
    print('start simulation')
    if cpar_equation is None:
        mU = get_vine_data(copula_type, n, T, m)
    else:
        mU = sample_from_dynamic_vine(distribution='gaussian', n=5, T=T, m=None)

    print('end simulation\n')
    list_marginal_objects = get_garch_data_and_models(n=n, mU=mU, distribution=distribution_marginal, mean_equation=ar1_equation, volatility_equation=garch_11_equation)

    return list_marginal_objects, list(np.array(list_parameters_tree1).flatten()) +\
                     list(np.array(list_parameters_tree2).flatten()) +\
                     list(np.array(list_parameters_tree3).flatten()) +\
                     list(np.array(list_parameters_tree4).flatten()) # return first copula parameter for every tree



def get_garch_data_and_models(n, mU, distribution, mean_equation, volatility_equation):
    list_marginal_objects = []
    if distribution == 'gaussian':
        parameters = [.5, 0.6, .05, .08, .9]
        distribution_module = gaussian
    if distribution == 'student_t':
        parameters = [4, .5, 0.6, .05, .08, .9]
        distribution_module = student_t

    for j in range(n):
        garch_model = MarginalObject(distribution_module_epsilon=distribution_module, volatility_equation=volatility_equation,
                                     mean_equation=mean_equation, parameters=parameters)

        garch_model.set_constraints([eq_cons_garch1, eq_cons_garch2])

        vU = mU[:, j]
        garch_model.simulate_from_garch(vU)
        list_marginal_objects.append(garch_model)

    return list_marginal_objects