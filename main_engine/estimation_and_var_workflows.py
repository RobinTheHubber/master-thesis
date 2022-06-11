import pickle

from datasets.estimated_parameters import dic_estimated_parameters_marginal_skewed_t_constant_mean
from forecasting_VaR.forecasting import value_at_risk_workflow
from forecasting_VaR.forecasting_HAR import HAR_estimation_workflow
from main_engine.realized_measure import get_realized_measure
from main_engine.run_model_with_data import estimate_model
from main_engine.get_data import load_return_data, load_realized_cov
from main_engine.run_parameters import RunParameters, create_filename_var_output
from marginal_engine.distributions import skewed_t
from marginal_engine.garch_model import constant_mean_equation


def estimation_and_var_workflow():
    evolution_type, copula_type, cpar_equation, run_model, run_var, skip_realized, train_idx, N, weights, n = RunParameters.get_run_parameters()
    marginal_models_estimated = {
        # gaussian:dic_estimated_parameters_marginal_gaussian_constant_mean}
     # student_t:dic_estimated_parameters_marginal_student_t_constant_mean}
        skewed_t: dic_estimated_parameters_marginal_skewed_t_constant_mean}
    mean_equation = constant_mean_equation

    ## estimate vine copula given vine copula type for each estimated marginal model
    for distribution_marginal, dictionary_estimated_parameters_marginal in marginal_models_estimated.items():
        marginal_models_list, T, daily_returns = load_return_data(distribution_marginal, train_idx, mean_equation, n)
        daily_realized_cov, list_sigma2 = load_realized_cov(n)
        test_set_size = T - train_idx

        dictionary_parameter_estimates, dictionary_filtered_rho0 = retrieve_model_estimation_output(distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                         train_idx, dictionary_estimated_parameters_marginal, daily_realized_cov, n,
                                         list_sigma2, run_model)

        ## set value-at-risk parameters
        portfolio_returns = (daily_returns.values @ weights).reshape((-1,))
        realized_measure = get_realized_measure(n, daily_realized_cov, list_sigma2, (train_idx, T))
        _, h_ij_hat_test = HAR_estimation_workflow(from_=train_idx, to_=T+1)

        retrieve_VaR_estimation_output(portfolio_returns, distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                       train_idx, dictionary_estimated_parameters_marginal, h_ij_hat_test, weights,
                                       N,
                                       test_set_size, daily_returns, dictionary_parameter_estimates,
                                       dictionary_filtered_rho0, run_var)

        # value_at_risk_test_workflow(portfolio_returns, train_idx, value_at_risk_output_garch, value_at_risk_output)


def retrieve_VaR_estimation_output(portfolio_returns, distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                   train_idx, dictionary_estimated_parameters_marginal, realized_measure, weights, N,
                                   test_set_size, daily_returns, dictionary_parameter_estimates, dictionary_filtered_rho0, run_model=False):


    filename_var_output = create_filename_var_output()
    try:
        # with open('../PKL files/VaR_GARCH_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'rb') as f:
        #     value_at_risk_output_garch = pickle.load(f)

        with open(filename_var_output, 'rb') as f:
            value_at_risk_output = pickle.load(f)

    except:
        run_model = True

    if run_model:
        #### Estimate the vine (sequentially)
        value_at_risk_output = value_at_risk_workflow(marginal_models_list, test_set_size,
                                                                                  dictionary_estimated_parameters_marginal,
                                                                                  daily_returns,
                                                                                  weights, N, portfolio_returns,
                                                                                  train_idx, cpar_equation, copula_type,
                                                                                  dictionary_parameter_estimates,
                                                                                  dictionary_filtered_rho0,
                                                                                  realized_measure)

        with open(filename_var_output, 'wb') as f:
            pickle.dump(value_at_risk_output, f)

    return value_at_risk_output


def retrieve_model_estimation_output(distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                     train_idx, dictionary_estimated_parameters_marginal, daily_realized_cov, n, list_sigma2, run_model=False):

    marginal_distribution = RunParameters.distribution
    realized_str = '_realized'
    if RunParameters.skip_realized:
        realized_str = '_non_realized'

    dynamic_str = '_static' * RunParameters.estimate_static_vine
    filename_rho = '../PKL files/rho0_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + realized_str +  dynamic_str + '.pkl'
    filename_parameter = '../PKL files/parameters_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + realized_str + dynamic_str + '.pkl'

    try:
        with open(filename_rho, 'rb') as f:
            dictionary_filtered_rho0 = pickle.load(f)

        with open(filename_parameter, 'rb') as f:
            dictionary_parameter_estimates = pickle.load(f)

    except:
        run_model = True

    if run_model:
        dictionary_parameter_estimates, filtered_rho = estimate_model(marginal_models_list, n, copula_type,
                                                                      cpar_equation, list_sigma2, train_idx,
                                                                      dictionary_estimated_parameters_marginal,
                                                                      daily_realized_cov)
        pre_obs_est = RunParameters.get_pre_obs_est()
        dictionary_filtered_rho0 = dict(zip(list(filtered_rho.keys()), [v[:, -pre_obs_est:] for k, v in filtered_rho.items()]))


        with open(filename_rho, 'wb') as f:
            pickle.dump(dictionary_filtered_rho0, f)

        with open(filename_parameter, 'wb') as f:
            pickle.dump(dictionary_parameter_estimates, f)

    return dictionary_parameter_estimates, dictionary_filtered_rho0