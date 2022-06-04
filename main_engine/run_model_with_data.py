# todo 0: compare dynamic gaussian vine copula garch VaR forecasts with simple GARCH parametric forecasts
# (active) todo 1: expand with dynamic t copula
# todo 2: expand with the realized volatility measures for the marginals and realized covariance measures for the evolution equations
# todo 2 to 3: probably discuss results somewhere around this point
# todo 3: try out a couple of different garch marginal or even a totally different marginal
import pandas as pd
from forecasting_VaR.forecasting import *
from main_engine.realized_measure import get_realized_measure
from marginal_engine.garch_model import *
from marginal_engine.distributions import *
from vine_copula_engine.vine_copula_estimation import get_vine_stuff, estimate_vine_sequentially


def load_data(distribution, train_idx, mean_equation):
    nvar = 5
    daily_returns_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    daily_returns = daily_returns_data.iloc[:, 1:1+nvar]
    realized_covariances_data = pd.read_csv('../datasets/10_dim_realized_covar.csv')
    count = 0
    dictionary_realized_measure = {}
    list_sigma2 = {}
    for j in range(nvar):
        list_sigma2[j+1] = realized_covariances_data.values[:, count+1].astype('float')

        for i, col_indx in enumerate(range(count + 1, count + nvar - j)):
            variable_pair_key = (j+1, i+j+2)
            vine_copula_key = (i + 1, j + 1)
            dictionary_realized_measure[vine_copula_key] = realized_covariances_data.values[:, col_indx+1].astype('float')

        count += 10 - j


    T = daily_returns.shape[0]
    n = daily_returns.shape[1]

    marginal_models_list = []
    for j in range(n):
        garch_model = MarginalObject(distribution_module_epsilon=distribution, volatility_equation=garch_11_equation,
                                     mean_equation=mean_equation)

        bounds = [[-np.inf, np.inf], [-np.inf, np.inf], [0, 1], [0, 1]]
        if distribution == student_t:
            bounds = [[-np.inf, np.inf]] + bounds
        if mean_equation == ar1_equation:
            bounds = [[-np.inf, np.inf]] + bounds

        garch_model.set_bounds(bounds)
        garch_model.set_constraints([eq_cons_garch2])
        garch_model.set_data(daily_returns.iloc[:train_idx, j])
        marginal_models_list.append(garch_model)

    return marginal_models_list, n, T, daily_returns, dictionary_realized_measure, list_sigma2


def estimate_model(marginal_models_list, n, copula_type, cpar_equation, list_sigma2, training_idx, dictionary_theta=None, daily_realized_cov=None):
    dynamic = True
    dictionary_transformation_functions, dictionary_copula_h_functions, \
    dictionary_copula_densities, dictionary_parameter_initial_values = get_vine_stuff(n=n, copula_type=copula_type,
                                                                                      dynamic=dynamic)

    #### Estimate marginal models
    if dictionary_theta is None:
        dictionary_v, dictionary_theta = estimate_marginals(marginal_models_list)
    else:
        dictionary_v = {}
        for i, marginal_model in enumerate(marginal_models_list):
            dictionary_v[(0, i + 1)] = get_PITs_with_estimated_parameters(marginal_model.data, dictionary_theta[i + 1],
                                                                          marginal_model)

    ### retrieve realized measures
    if daily_realized_cov is not None:
        realized_measure = get_realized_measure(n, daily_realized_cov, list_sigma2, (0, training_idx))
    else:
        realized_measure = {}
        for j in range(1, n):
            for i in range(1, n-j):
                realized_measure[(j, 1)] = None

    #### Estimate the vine (sequentially)
    dicParameters_seq, filtered_rho = estimate_vine_sequentially(dictionary_v, dictionary_theta,
                                                                 dictionary_transformation_functions,
                                                                 dictionary_copula_h_functions,
                                                                 dictionary_copula_densities,
                                                                 dictionary_parameter_initial_values, n, cpar_equation, realized_measure)
    return dicParameters_seq, filtered_rho

