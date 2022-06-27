import pickle

import numpy as np
from marginal_engine.distributions import *
from marginal_engine.garch_model import *
from vine_copula_engine.copula_functions import *


def loop_over_run_parameters(perform_fun, vine_choices, copulas,*args):
    # make sure to run every configuration even if it means overwriting current stored results
    vine_types = {'static': (True, True), 'dynamic': (False, True), 'realized': (False, False)}

    # run each configuration: different marginal distribution setups and different vine model setups
    for cop in copulas:
        RunParameters.copula_type = cop
        marginal_distribution = 'realEGARCG'
        RunParameters.distribution = marginal_distribution
        for vine_type_string, vine_type in vine_types.items():
            if vine_choices[vine_type_string]:
                RunParameters.estimate_static_vine, RunParameters.skip_realized = vine_type
                perform_fun(*args)

class RunParameters():
    copula_type = 'clayton'
    distribution = 'realEGARCH'
    ldist = {1: student_t, 2: skewed_t, 3:skewed_t, 4:student_t, 5:student_t}
    weighting = 'constant'
    estimate_static_vine = False
    skip_realized = False
    run_model = True
    run_var = True
    Q = [0.9, 0.95, 0.99, 0.995]
    optimization_method = 'L-BFGS-B'
    evolution_type = 'simple'
    cpar_equation = 'alt'
    N_mm = 1000
    mean_equation = constant_mean_equation
    nvar = 5
    estimation_window = 1500
    N = 10000
    equal_weighting = False
    weights = np.array([.2, .2, .2, .2, .2]).reshape((-1, 1))
    index = (0,0)
    with open('../PKL files/parameters_realEGARCG_clayton_alt_non_realized_static.pkl', 'rb') as f:
        par_static_estimated = pickle.load(f)

    @staticmethod
    def get_marginal_distribution():
        if RunParameters.distribution == 'gaussian':
            marginal_distribution = gaussian
        elif RunParameters.distribution == 'student_t':
            marginal_distribution = student_t
        elif RunParameters.distribution == 'skewed_t':
            marginal_distribution = skewed_t
        else:
            marginal_distribution = None

        return marginal_distribution



    @staticmethod
    def get_evolution_npar():
        if RunParameters.evolution_type == 'simple':
            npar_evolution = 3
        elif RunParameters.evolution_type == 'HAR':
            npar_evolution = 4
        elif RunParameters.evolution_type == 'simple_ar':
            npar_evolution = 3

        return npar_evolution

    @staticmethod
    def get_pre_obs_est():
        if RunParameters.evolution_type == 'simple':
            obs_pre_est = 1
        elif RunParameters.evolution_type == 'HAR':
            obs_pre_est = 22
        elif RunParameters.evolution_type == 'simple_ar':
            obs_pre_est = 1

        return obs_pre_est

    @staticmethod
    def get_copula_density():
        if RunParameters.copula_type == 'gaussian':
            return copula_density_gaussian
        elif RunParameters.copula_type == 'student_t':
            return copula_density_student_t

    @staticmethod
    def get_copula_h_function():
        if RunParameters.copula_type == 'gaussian':
            return h_function_gaussian
        elif RunParameters.copula_type == 'archemedian':
            return h_function_clayton


    @staticmethod
    def get_copula_h_inv_function():
        if RunParameters.copula_type == 'gaussian':
            return h_function_inv_gaussian
        elif RunParameters.copula_type == 'student_t':
            return h_function_inv_clayton

    @staticmethod
    def get_run_parameters():
        return RunParameters.evolution_type, RunParameters.copula_type, RunParameters.cpar_equation, RunParameters.run_model, RunParameters.run_var, RunParameters.skip_realized, RunParameters.estimation_window, RunParameters.N, RunParameters.weights, RunParameters.nvar


def create_filename_var_output():
    realized_str = 'realized'
    if RunParameters.skip_realized:
        realized_str = 'non_realized'
    dynamic_str = '_static' * RunParameters.estimate_static_vine

    ew = (1-RunParameters.equal_weighting) * 'MV_weights_'
    filename_var_output = '../PKL files/VaR_' + ew + RunParameters.distribution + '_' + RunParameters.copula_type + '_' + RunParameters.cpar_equation + '_' + realized_str + dynamic_str + '.pkl'
    return filename_var_output