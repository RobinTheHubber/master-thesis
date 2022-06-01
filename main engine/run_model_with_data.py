# todo: load in data and transform for appropriate use in dynamic vine estimation
# todo: expand with the realized volatility measures for the marginals and realized covariance measures for the evolution equations

import pandas as pd
from forecasting import *
from marginal_engine.garch_model import *
from marginal_engine.distributions import *
from vine_copula_engine.vine_copula_estimation import get_vine_stuff, estimate_vine_sequentially
from datasets.estimated_parameters import *


def load_data(distribution, train_idx, mean_equation):
    daily_returns_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    daily_returns = daily_returns_data.iloc[:, 1:6]
    realized_covariances_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    realized_covariances = realized_covariances_data.iloc[:, 1:6]
    stock_names = [name[:-7] for name in daily_returns.columns]

    T = daily_returns.shape[0]
    n = daily_returns.shape[1]

    marginal_models_list = []
    for j in range(n):
        garch_model = MarginalObject(distribution_module_epsilon=distribution, volatility_equation=garch_11_equation,
                                     mean_equation=mean_equation)

        bounds = [[-np.inf, np.inf], [-np.inf,np.inf], [0, 1], [0, 1]]
        if distribution == student_t:
            bounds = [[-np.inf, np.inf]] + bounds
        if mean_equation == ar1_equation:
            bounds = [[-np.inf, np.inf]] + bounds

        garch_model.set_bounds(bounds)
        garch_model.set_constraints([eq_cons_garch2])
        garch_model.set_data(daily_returns.iloc[:train_idx, j])
        marginal_models_list.append(garch_model)

    return marginal_models_list, n, T, daily_returns

def estimate_model(marginal_models_list, n, copula_type, cpar_equation, dictionary_theta=None):
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
             dictionary_v[(0, i+1)] = get_PITs_with_estimated_parameters(marginal_model.data, dictionary_theta[i+1], marginal_model)

    #### Estimate the vine (sequentially)
    dicParameters_seq, filtered_rho = estimate_vine_sequentially(dictionary_v, dictionary_theta, dictionary_transformation_functions, dictionary_copula_h_functions,
                               dictionary_copula_densities, dictionary_parameter_initial_values, n, cpar_equation)
    return dicParameters_seq, filtered_rho

def main():
    train_idx = 1500
    copula_type = 'gaussian'
    cpar_equation = 'difference'
    est_par = [dic_estimated_parameters_marginal_gaussian_constant_mean,
     dic_estimated_parameters_marginal_student_t_constant_mean]
    dist_module = [gaussian, student_t]

    for i in range(2):
        dic_estimated_parameters_marginal = est_par[i]
        distribution_marginal = dist_module[i]
        mean_equation = constant_mean_equation
        # print('\nstart..:' + str(mean_equation) + ' & ' + str(distribution_marginal))
        marginal_models_list, n, T, daily_returns = load_data(distribution_marginal, train_idx, mean_equation)
        test_set_size = T - train_idx
        dicParameters_estimated, filtered_rho = estimate_model(marginal_models_list, n, copula_type, cpar_equation, dic_estimated_parameters_marginal_gaussian_constant_mean)

    return
    N = 1000
    value_at_risk_output = compute_value_at_risk(daily_returns, cpar_equation, copula_type, marginal_models_list, dic_estimated_parameters_all_gaussian, test_set_size, N=N)
    value_at_risk_test_output = compute_value_at_risk_test_output(daily_returns[train_idx:], value_at_risk_output)
    for k, v in value_at_risk_test_output.items():
        print((k, v))

if __name__ == '__main__':
    main()
















