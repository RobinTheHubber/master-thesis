import pandas as pd
from vine_copula_engine.simulate_vine_algorithm import h_set_all_same, sample_from_vine3D
from vine_copula_engine.vine_copula_estimation import *
from vine_copula_engine.copula_functions import obs_pre_est
import matplotlib.pyplot as plt
from marginal_engine.marginal_model import MarginalObject
from scipy.stats import chi2
from vartests import duration_test

def update_parameters(marginal_model, estimated_parameters):
    marginal_model.mean_equation.update_parameters(estimated_parameters['mean'])
    marginal_model.volatility_equation.update_parameters(estimated_parameters['vol'])

    if marginal_model.n1 > 0:
        marginal_model.distribution_parameters = estimated_parameters['dist']
    else:
        marginal_model.distribution_parameters = None

def get_PITs_with_estimated_parameters(data, estimated_parameters, marginal_model: MarginalObject):
    update_parameters(marginal_model, estimated_parameters)

    marginal_model.set_data(data)
    marginal_model.compute_pits()
    return marginal_model.PITs


def compute_value_at_risk(data, weights, cpar_equation, copula_type, marginal_models_list, dicParameters_estimated, filtered_rho0,
                          test_set_size, N=1000, realized_measure=None):
    T = data.shape[0]
    if copula_type == 'gaussian':
        h_function = h_function_gaussian
        h_function_inv = h_function_inv_gaussian
    elif copula_type == 'student_t':
        h_function = h_function_student_t
        h_function_inv = h_function_inv_student_t

    n = data.shape[1]
    training_set_size = T - test_set_size

    # get PITs from estimated marginal models for test set
    PITs = np.zeros((test_set_size, n))
    for j in range(n):
        PITs_j = get_PITs_with_estimated_parameters(data.iloc[:, j], dicParameters_estimated[j + 1],
                                                    marginal_models_list[j])
        PITs[:, j] = PITs_j[training_set_size:]

    ## filter the rho vectors over time (non-path dependent)
    # rho0 = dict(zip(list(dic_estimated_parameters_all_gaussian.keys()), [0.5] * 15))
    dictionary_filtered_rho_test_set = get_filtered_rho_after_estimation(PITs, h_function, dicParameters_estimated, n,
                                                                         cpar_equation, filtered_rho0, realized_measure)

    dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho_test_set, h_function, h_function_inv)
    N_ = 1000
    PITs_test_set = np.zeros((N, test_set_size, n))
    for i in range(int(N / N_)):
        PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv,
                                                                      dictionary_filtered_rho_test_set, n,
                                                                      test_set_size, N_)



    Q = [0.9, 0.95, 0.99]
    value_at_risk_dictionary = {}
    for q in Q:
        value_at_risk_dictionary[q] = np.zeros(test_set_size)

    portfolio_index_simulated = np.zeros((test_set_size, N))
    for j in range(n):
        marginal_model = marginal_models_list[j]
        marginal_data_simulated = np.zeros((test_set_size, N))
        mu_, sigma2_ = marginal_model.filter()
        mu, sigma2 = mu_[training_set_size:], sigma2_[training_set_size:]
        for t in range(test_set_size):
            simulated_PITs_j = PITs_test_set[:, t, j]
            epsilon = marginal_model.distribution_module.ppf(marginal_model.distribution_parameters, simulated_PITs_j)
            marginal_data_simulated[t] = mu[t] + epsilon * np.sqrt(sigma2[t])

        portfolio_index_simulated += marginal_data_simulated * weights[j]

    for q in Q:
        value_at_risk_dictionary[q] = np.quantile(portfolio_index_simulated, q=1-q, axis=1)


    portfolio_index_return = (data.values @ weights).reshape((-1,))

    plt.plot(portfolio_index_return[training_set_size:])
    for q in Q:
        plt.plot(value_at_risk_dictionary[q])

    return value_at_risk_dictionary

def get_value_at_risk_output_garch(list_marginal_models, test_set_size, dicEstimated_parameters, data, weights, N):
    n = len(list_marginal_models)
    Q = [0.9, 0.95, 0.99]
    value_at_risk_dictionary = {}

    for q in Q:
        value_at_risk_dictionary[q] = np.zeros((test_set_size, n))

    portfolio_index_simulated = np.zeros((test_set_size, N))
    for j in range(n):
        marginal_model = list_marginal_models[j]
        marginal_model.set_data(data[:, j])
        update_parameters(marginal_model, dicEstimated_parameters[j+1])
        marginal_data_simulated = np.zeros((test_set_size, N))
        mu_, sigma2_ = marginal_model.filter()
        mu, sigma2 = mu_[-test_set_size:], sigma2_[-test_set_size:]
        for t in range(test_set_size):
            U = np.random.random(size=N)
            epsilon = marginal_model.distribution_module.ppf(marginal_model.distribution_parameters, U)
            marginal_data_simulated[t] = mu[t] + epsilon * np.sqrt(sigma2[t])

        portfolio_index_simulated += marginal_data_simulated * weights[j]

    for q in Q:
        value_at_risk_dictionary[q] = np.quantile(portfolio_index_simulated, q=1 - q, axis=1)

    portfolio_index_return = (data @ weights).reshape((-1,))
    plt.plot(portfolio_index_return[-test_set_size:])
    for q in Q:
        plt.plot(value_at_risk_dictionary[q])

    plt.show()

    return value_at_risk_dictionary

def compute_value_at_risk_test_output(data, value_at_risk_dictionary):
    Q = list(value_at_risk_dictionary.keys())
    dictionary_VaR_test_results = {}
    for q in Q:
        dictionary_VaR_test_results[q] = compute_test_value_at_risk(data[obs_pre_est:], q,
                                                                         value_at_risk_dictionary[q][obs_pre_est:])
    return dictionary_VaR_test_results


def proportion_of_failures(violations):
    pass
def haas_independence_test(violations):
    pass

def uc_test(violations, alpha):
    T = len(violations)
    T1 = sum(violations)
    T0 = T - T1

    statistic = - 2 * np.log((1-alpha)**T0 * alpha **T1 / (1 - T1/T)**T0 * T1/T**T1)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return (statistic, p_value)

def duration_testing(violations, confidence_levels):
    try:
        items_of_interest = ['log-likelihood ratio test statistic', 'null hypothesis', 'decision']
        duration_test_results = dict(zip(confidence_levels, [ \
            [v for k, v in duration_test(pd.Series(violations), alpha).items() if k in items_of_interest] \
            for alpha in confidence_levels]))
    except:
        duration_test_results = None

    return duration_test_results

def compute_test_value_at_risk(data_array, q, value_at_risk_array):
    confidence_levels = [0.01, 0.025, 0.05, 0.1]
    violations = data_array < value_at_risk_array
    violation_ratio = sum(violations) / len(data_array)
    UC_test = uc_test(violations, alpha=1-q)
    IND_test = []
    duration_test_results = duration_testing(violations, confidence_levels)
    output = {'violation_ratio': violation_ratio, 'UC':UC_test,'IND':IND_test, 'DURATION':duration_test_results}
    return output

def value_at_risk_workflow(marginal_models_list, test_set_size, dictionary_estimated_parameters_marginal, daily_returns,
                           weights, N, portfolio_returns, train_idx, cpar_equation, copula_type,
                           dictionary_parameter_estimates, dictionary_filtered_rho0, realized_measure):
    ## test value-at-risk output GARCH model
    value_at_risk_output_garch = get_value_at_risk_output_garch(marginal_models_list, test_set_size,
                                                                dictionary_estimated_parameters_marginal,
                                                                daily_returns.values, weights, N=N)

    ## test value-at-risk output dynamic copula model
    value_at_risk_output = compute_value_at_risk(daily_returns, weights, cpar_equation, copula_type,
                                                 marginal_models_list,
                                                 dictionary_parameter_estimates, dictionary_filtered_rho0,
                                                 test_set_size, N=N, realized_measure=realized_measure)

    return value_at_risk_output_garch, value_at_risk_output

def value_at_risk_test_workflow(portfolio_returns, train_idx, value_at_risk_output_garch, value_at_risk_output):
    value_at_risk_test_output_garch = compute_value_at_risk_test_output(portfolio_returns[train_idx:],
                                                                    value_at_risk_output_garch)
    for k, v in value_at_risk_test_output_garch.items():
        print((k, v))

    value_at_risk_test_output = compute_value_at_risk_test_output(portfolio_returns[train_idx:],
                                                                  value_at_risk_output)

    for k, v in value_at_risk_test_output.items():
        print((k, v))