from scipy.stats import t

from datasets.estimated_parameters import dic_estimated_parameters_marginal_gaussian_constant_mean, \
    dic_estimated_parameters_marginal_student_t_constant_mean
from forecasting_VaR.forecasting import compute_value_at_risk, dic_estimated_parameters_all_gaussian, \
    compute_value_at_risk_test_output
from main_engine.run_model_with_data import load_data, estimate_model
from marginal_engine.distributions import *
from marginal_engine.garch_model import constant_mean_equation


def main():
    train_idx = 1500
    copula_type = 'gaussian'
    cpar_equation = 'difference'
    marginal_models_estimated = {gaussian:dic_estimated_parameters_marginal_gaussian_constant_mean,
     student_t:dic_estimated_parameters_marginal_student_t_constant_mean}
    mean_equation = constant_mean_equation

    ## estimate vine copula given vine copula type for each estimated marginal model
    for distribution_marginal, dictionary_estimated_parameters_marginal in marginal_models_estimated.items():
        marginal_models_list, n, T, daily_returns = load_data(distribution_marginal, train_idx, mean_equation)
        test_set_size = T - train_idx
        dicParameters_estimated, filtered_rho = estimate_model(marginal_models_list, n, copula_type, cpar_equation,
                                                               dictionary_estimated_parameters_marginal)


        ## comute value-at-risk output
        filtered_rho0 = dict(zip(list(filtered_rho.keys()), [v[:, -1] for k, v in filtered_rho.items()]))
        value_at_risk_output = compute_value_at_risk(daily_returns, cpar_equation, copula_type, marginal_models_list,
                                                     dicParameters_estimated, filtered_rho0, test_set_size, N=1000)
        ## test value-at-risk output
        value_at_risk_test_output = compute_value_at_risk_test_output(daily_returns[train_idx:], value_at_risk_output)
        for k, v in value_at_risk_test_output.items():
            print((k, v))


if __name__ == '__main__':
    main()
