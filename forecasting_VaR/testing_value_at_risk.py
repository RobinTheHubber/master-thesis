import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2
from vartests import duration_test
from main_engine.get_data import get_portfolio_returns
from main_engine.run_parameters import RunParameters, loop_over_run_parameters, create_filename_var_output
from vine_copula_engine.dynamic_copula_functions import obs_pre_est


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
    """
    null-hypothesis is the assumption that violations are bernoulli(alpha)
    vs alternative that they are bernoulli(p) for any p in [0,1] (i.e. p_hat which gives ML max.)
    :param violations:
    :param alpha:
    :return:
    """
    T = len(violations)
    T1 = sum(violations)
    T0 = T - T1

    LL0 = T0 * np.log(1-alpha) + T1 * np.log(alpha)
    LLA = T0 * np.log(T0/T) + T1 * np.log(T1/T)
    statistic = - 2 * (LL0 - LLA)

    p_value = 1 - chi2.cdf(statistic, df=1)
    return (statistic, p_value)
def ind_test(violations, alpha):
    """
    null-hypothesis is the assumption that violations are bernoulli(alpha)
    vs alternative that they are bernoulli(p) for any p in [0,1] (i.e. p_hat which gives ML max.)
    :param violations:
    :param alpha:
    :return:
    """
    T = len(violations)
    T1 = sum(violations)
    T0 = T - T1

    LL0 = T0 * np.log(1-alpha) + T1 * np.log(alpha)
    LLA = T0 * np.log(T0/T) + T1 * np.log(T1/T)
    statistic = - 2 * (LL0 - LLA)

    p_value = 1 - chi2.cdf(statistic, df=1)
    return (statistic, p_value)
def cc_test(violations, alpha):
    """
    null-hypothesis is the assumption that violations are bernoulli(alpha)
    vs alternative that they are bernoulli(p) for any p in [0,1] (i.e. p_hat which gives ML max.)
    :param violations:
    :param alpha:
    :return:
    """
    T = len(violations)
    T1 = sum(violations)
    T0 = T - T1

    LL0 = T0 * np.log(1-alpha) + T1 * np.log(alpha)
    LLA = T0 * np.log(T0/T) + T1 * np.log(T1/T)
    statistic = - 2 * (LL0 - LLA)

    p_value = 1 - chi2.cdf(statistic, df=1)
    return (statistic, p_value)

#
# def duration_testing(violations, confidence_levels):
#     try:
#         items_of_interest = ['log-likelihood ratio test statistic', 'null hypothesis', 'decision']
#         duration_test_results = dict(zip(confidence_levels, [ \
#             [v for k, v in duration_test(pd.Series(violations), alpha).items() if k in items_of_interest] \
#             for alpha in confidence_levels]))
#     except:
#         duration_test_results = None
#
#     return duration_test_results


def compute_test_value_at_risk(data_array, q, value_at_risk_array):
    confidence_levels = [0.01, 0.025, 0.05, 0.1]
    violations = data_array < value_at_risk_array
    violation_sizes = (1 + (value_at_risk_array-data_array)**2) * violations
    c = 0.02 # 2% opportunity cost for eg mortgage lending
    cost_of_holding_capital = c * value_at_risk_array * (1 - violations)
    violation_ratio = sum(violations) / len(data_array)
    UC_test = uc_test(violations, alpha=1-q)
    CC_test = cc_test(violations, alpha=1 - q)
    IND_test = ind_test(violations, alpha=1 - q)
    AQLF = np.mean(violation_sizes)
    AFLF = np.mean(violation_sizes - cost_of_holding_capital)

    output = {'violation_ratio': violation_ratio, 'UC':UC_test,'IND':IND_test, 'CC':CC_test, 'AQLF':AQLF, 'AFLF':AFLF}
    return output


def value_at_risk_test_workflow(portfolio_returns, train_idx, value_at_risk_output_garch, value_at_risk_output):
    value_at_risk_test_output_garch = compute_value_at_risk_test_output(portfolio_returns[train_idx:],
                                                                    value_at_risk_output_garch)
    for k, v in value_at_risk_test_output_garch.items():
        print((k, v))

    value_at_risk_test_output = compute_value_at_risk_test_output(portfolio_returns[train_idx:],
                                                                  value_at_risk_output)

    for k, v in value_at_risk_test_output.items():
        print((k, v))




def analyze_value_at_risk_output(vine_choice, copulas):
    portfolio_returns = get_portfolio_returns()
    T = len(portfolio_returns)

    ## load in different VaR results
    dictionary_all_var_output = {}
    def store_var_output_in_dictionary(dictionary_all_var_output):
        file_name = create_filename_var_output()
        name = '{} marginals -{}{} {} copula'.format(RunParameters.distribution,
                                                     (1-RunParameters.estimate_static_vine) * ' dynamic',
                                                     (1 - RunParameters.skip_realized) * ' realized',
                                                     RunParameters.copula_type)
        with open(file_name, 'rb') as f:
            file = pickle.load(f)

        dictionary_all_var_output[name] = file


    loop_over_run_parameters(store_var_output_in_dictionary, vine_choice, copulas, dictionary_all_var_output)

    # value_at_risk_output_HS = get_value_at_risk_output_HS(test_set_size, portfolio_returns, N)
    # value_at_risk_output_MV = get_value_at_risk_output_MV(test_set_size, portfolio_returns)
    # dictionary_all_var_output['historical simulation'] = value_at_risk_output_HS
    # dictionary_all_var_output['Mean-Variance method'] = value_at_risk_output_MV

    Q = RunParameters.Q
    for q in Q:
        plt.figure(1000*q)
        plt.plot(portfolio_returns)
        for label, value_at_risk_output in dictionary_all_var_output.items():
            plt.plot(value_at_risk_output[q], label=label)
        plt.legend()
        plt.show()

    dictionary_all_var_test_output = {}
    m_output = np.zeros((len(dictionary_all_var_output), len(Q)))
    for j, item in enumerate(dictionary_all_var_output.items()):
        method, var_outpout = item
        var_test_output = compute_value_at_risk_test_output(portfolio_returns, var_outpout)
        dictionary_all_var_test_output[method] = var_test_output
        print(method)
        # print(var_test_output)
        for i, q in enumerate(Q):
            m_output[j, i] = var_test_output[q]['violation_ratio']

    print(m_output)






