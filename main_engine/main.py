from datasets.estimated_parameters import dic_estimated_parameters_marginal_gaussian_constant_mean, \
    dic_estimated_parameters_marginal_student_t_constant_mean
from main_engine.realized_measure import get_realized_measure
from main_engine.run_model_with_data import load_data, estimate_model
from main_engine.run_parameters import RunParameters
from marginal_engine.distributions import *
from marginal_engine.garch_model import constant_mean_equation
import pickle
from forecasting_VaR.forecasting import  *


def plot_filtered_parameters_vine_copula(filtered_rho):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax1, ax2 = axs
    for k, v in filtered_rho.items():
        ax1.plot(v[0,:], label=k)
        ax2.plot(v[1,:], label=k)
    plt.legend()
    plt.show()


def retrieve_model_estimation_output(distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                     train_idx, dictionary_estimated_parameters_marginal, daily_realized_cov, n, list_sigma2, run_model=False):
    if distribution_marginal == gaussian:
        marginal_distribution = 'gaussian'
    else:
        marginal_distribution = 'student_t'

    try:
        with open('rho0_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'rb') as f:
            dictionary_filtered_rho0 = pickle.load(f)

        with open('parameters_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'rb') as f:
            dictionary_parameter_estimates = pickle.load(f)

    except:
        run_model = True

    if run_model:
        dictionary_parameter_estimates, filtered_rho = estimate_model(marginal_models_list, n, copula_type,
                                                                      cpar_equation, list_sigma2, train_idx,
                                                                      dictionary_estimated_parameters_marginal,
                                                                      daily_realized_cov)
        dictionary_filtered_rho0 = dict(zip(list(filtered_rho.keys()), [v[:, -1] for k, v in filtered_rho.items()]))
        with open('rho0_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'wb') as f:
            pickle.dump(dictionary_filtered_rho0, f)

        with open('parameters_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'wb') as f:
            pickle.dump(dictionary_parameter_estimates, f)

    return dictionary_parameter_estimates, dictionary_filtered_rho0

def retrieve_VaR_estimation_output(portfolio_returns, distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                   train_idx, dictionary_estimated_parameters_marginal, realized_measure, weights, N,
                                   test_set_size, daily_returns, dictionary_parameter_estimates, dictionary_filtered_rho0, run_model=False):
    if distribution_marginal == gaussian:
        marginal_distribution = 'gaussian'
    else:
        marginal_distribution = 'student_t'

    try:
        with open('VaR_GARCH_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'rb') as f:
            value_at_risk_output_garch = pickle.load(f)

        with open('VaR_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'rb') as f:
            value_at_risk_output = pickle.load(f)

    except:
        run_model = True

    if run_model:
        value_at_risk_output_garch, value_at_risk_output = value_at_risk_workflow(marginal_models_list, test_set_size,
                                                                                  dictionary_estimated_parameters_marginal,
                                                                                  daily_returns,
                                                                                  weights, N, portfolio_returns,
                                                                                  train_idx, cpar_equation, copula_type,
                                                                                  dictionary_parameter_estimates,
                                                                                  dictionary_filtered_rho0,
                                                                                  realized_measure)

        with open('VaR_GARCH_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'wb') as f:
            pickle.dump(value_at_risk_output_garch, f)

        with open('VaR_' + marginal_distribution + '_' + copula_type + '_' + cpar_equation + '.pkl', 'wb') as f:
            pickle.dump(value_at_risk_output, f)

    return value_at_risk_output_garch, value_at_risk_output

def main():
    evolution_type, copula_type, cpar_equation, run_model, run_var, skip_realized, train_idx, N, weights = RunParameters.get_run_parameters()
    marginal_models_estimated = {gaussian:dic_estimated_parameters_marginal_gaussian_constant_mean,
     student_t:dic_estimated_parameters_marginal_student_t_constant_mean}
    mean_equation = constant_mean_equation

    ## estimate vine copula given vine copula type for each estimated marginal model
    for distribution_marginal, dictionary_estimated_parameters_marginal in marginal_models_estimated.items():
        marginal_models_list, n, T, daily_returns, daily_realized_cov, list_sigma2 = load_data(distribution_marginal, train_idx, mean_equation)
        test_set_size = T - train_idx

        dictionary_parameter_estimates, dictionary_filtered_rho0 = retrieve_model_estimation_output(distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                         train_idx, dictionary_estimated_parameters_marginal, daily_realized_cov, n,
                                         list_sigma2, run_model)

        ## set value-at-risk parameters
        portfolio_returns = (daily_returns.values @ weights).reshape((-1,))
        realized_measure = get_realized_measure(n, daily_realized_cov, list_sigma2, (train_idx, T))

        value_at_risk_output_garch, value_at_risk_output = retrieve_VaR_estimation_output(portfolio_returns, distribution_marginal, copula_type, cpar_equation, marginal_models_list,
                                       train_idx, dictionary_estimated_parameters_marginal, realized_measure, weights,
                                       N,
                                       test_set_size, daily_returns, dictionary_parameter_estimates,
                                       dictionary_filtered_rho0, run_var)

        value_at_risk_test_workflow(portfolio_returns, train_idx, value_at_risk_output_garch, value_at_risk_output)


if __name__ == '__main__':
    main()
