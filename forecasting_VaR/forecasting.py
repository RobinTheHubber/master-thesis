import pickle

import numpy as np

from main_engine.run_parameters import RunParameters, create_filename_var_output
from marginal_engine.distributions import student_t, skewed_t
from marginal_engine.garch_model import garch_11_equation, eq_cons_garch2, constant_mean_equation
from marginal_engine.realGARCH import filter_realEGARCH
from vine_copula_engine.simulate_vine_algorithm import h_set_all_same, sample_from_vine3D, sample_from_vine
from vine_copula_engine.vine_copula_estimation import *
import matplotlib.pyplot as plt
from marginal_engine.marginal_model import MarginalObject
import pandas as pd


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


def get_all_PITs_test_realEGARCH(test_set_size, n, data, dicParameters_estimated, ldist, lrv):
    sigma2_test, pits_test = np.zeros((2, test_set_size, n))
    for i in range(1, n + 1):
        rt, rv = data.iloc[:, i-1] - np.mean(data.iloc[:, i-1]), lrv[i]
        parameters_estimated_update_equation = dicParameters_estimated[i]
        loght, logx, zt = filter_realEGARCH(parameters_estimated_update_equation, rt, rv)
        x0 = [4]
        if ldist[i] == skewed_t:
            x0 += [1]
        res = minimize(fun=lambda *args: -ldist[i].loglik(*args), x0=x0, args=(zt[:-test_set_size], None, None),
                       bounds=[[2.1, 60], [0.01, 100]], method='BFGS')
        parameters_estimated_distribution = res.x
        pits = ldist[i].cdf(parameters_estimated_distribution, zt)
        pits_test[:, i-1] = pits[-test_set_size:]
        sigma2_test[:, i-1] = np.exp(loght[-test_set_size:])

    return pits_test, sigma2_test

def get_all_PITs_realEGARCH(training_idx, n, data, dicParameters_estimated, ldist, lrv):
    sigma2_test, pits_test = np.zeros((2, training_idx, n))
    for i in range(1, n + 1):
        parameters_estimated_update_equation = dicParameters_estimated[i]
        rt, rv = data.iloc[:training_idx, i - 1] - np.mean(data.iloc[:training_idx, i - 1]), lrv[i][:training_idx]
        loght, logx, zt = filter_realEGARCH(parameters_estimated_update_equation, rt, rv)
        x0 = [4]
        if ldist[i]==skewed_t:
            x0 += [1]
        res = minimize(fun=lambda *args: -ldist[i].loglik(*args), x0=x0, args=(zt, None, None), bounds=[[2.1, 60], [0.01,100]], method='BFGS')
        parameters_estimated_distribution = res.x
        pits = ldist[i].cdf(parameters_estimated_distribution, zt)
        pits_test[:, i - 1] = pits
        sigma2_test[:, i - 1] = np.exp(loght)

    return pits_test, sigma2_test








    filter_realEGARCH

    ...


def compute_value_at_risk(data, lrv, weights, cpar_equation, copula_type, marginal_models_list, dicParameters_estimated, filtered_rho0,
                          test_set_size, N=1000, realized_measure=None):
    T = data.shape[0]
    if copula_type == 'gaussian':
        h_function = h_function_gaussian
        h_function_inv = h_function_inv_gaussian
    elif copula_type == 'student_t':
        h_function = h_function_student_t
        h_function_inv = h_function_inv_student_t
    elif copula_type == 'clayton':
        h_function = h_function_clayton
        h_function_inv = h_function_inv_clayton

    n = data.shape[1]
    training_set_size = T - test_set_size

    ## filter the rho vectors over time (non-path dependent)
    ldist = RunParameters.ldist
    PITs_test, sigma2_test = get_all_PITs_test_realEGARCH(test_set_size, n, data, dicParameters_estimated, ldist, lrv)

    ## obtain PITs_test for the test set in order to forecast value-at-risk figures
    if RunParameters.estimate_static_vine:
        dictionary_h, dictionary_h_inv = h_set_all_same(dicParameters_estimated, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))
        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine(dictionary_h, dictionary_h_inv, dicParameters_estimated, n, test_set_size*N_).reshape((N_, test_set_size, n))

    else:
        dictionary_filtered_rho_test_set = get_filtered_rho_after_estimation(PITs_test, h_function, dicParameters_estimated, n,
                                                                             cpar_equation, filtered_rho0, realized_measure)

        dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho_test_set, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))

        for key in get_keys():
            plt.plot(dictionary_filtered_rho_test_set[key][0])

        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv,
                                                                          dictionary_filtered_rho_test_set, n,
                                                                          test_set_size, N_)


    Q = RunParameters.Q
    value_at_risk_dictionary = {}
    for q in Q:
        value_at_risk_dictionary[q] = np.zeros(test_set_size)

    portfolio_index_simulated = np.zeros((test_set_size, N))
    for j in range(n):
        marginal_data_simulated = np.zeros((test_set_size, N))
        mu = np.mean(data.iloc[:-test_set_size, j])
        rt, rv = data.iloc[:-test_set_size, j] - mu, lrv[j+1][:-test_set_size]
        loght, logx,  zt= filter_realEGARCH(dicParameters_estimated[j+1], rt, rv)
        x0 = [4]
        if ldist[j+1] == skewed_t:
            x0 += [1]

        res = minimize(fun=lambda *args: -ldist[j+1].loglik(*args), x0=x0, args=(zt[:-test_set_size], None, None),
                       bounds=[[2.1, 60], [0.01, 100]], method='BFGS')
        par = res.x
        for t in range(test_set_size):
            simulated_PITs_j = PITs_test_set[:, t, j]
            epsilon = ldist[j + 1].ppf(par, simulated_PITs_j)
            marginal_data_simulated[t] = mu + epsilon * np.sqrt(sigma2_test[t, j])


        portfolio_index_simulated += marginal_data_simulated * weights[j]
    #     plt.figure(j + 10)
    #     plt.plot(data.values[-test_set_size:, j])
    #     plt.plot(np.quantile(marginal_data_simulated, q=0.1, axis=1))
    #     plt.plot(np.quantile(marginal_data_simulated, q=0.05, axis=1))
    #     plt.plot(np.quantile(marginal_data_simulated, q=0.01, axis=1))
    #     print(np.sum(np.quantile(marginal_data_simulated, q=0.1, axis=1) > data.values[-test_set_size:, j]))
    # print('')

    for q in Q:
        value_at_risk_dictionary[q] = np.quantile(portfolio_index_simulated, q=1-q, axis=1)


    portfolio_index_return = (data.values @ weights).reshape((-1,))

    for q in Q:
        plt.figure(100 * q)
        plt.plot(portfolio_index_return[-test_set_size:])
        plt.plot(value_at_risk_dictionary[q])
        plt.legend()
        plt.show()

    return value_at_risk_dictionary



def compute_value_at_risk_(data, weights, cpar_equation, copula_type, marginal_models_list, dicParameters_estimated, filtered_rho0,
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

    ## filter the rho vectors over time (non-path dependent)
    PITs = get_all_PITs(test_set_size, n, data, dicParameters_estimated, marginal_models_list, training_set_size)

    ## obtain PITs for the test set in order to forecast value-at-risk figures
    if RunParameters.estimate_static_vine:
        dictionary_h, dictionary_h_inv = h_set_all_same(dicParameters_estimated, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))
        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine(dictionary_h, dictionary_h_inv, dicParameters_estimated, n, test_set_size*N_).reshape((N_, test_set_size, n))

    else:
        dictionary_filtered_rho_test_set = get_filtered_rho_after_estimation(PITs, h_function, dicParameters_estimated, n,
                                                                             cpar_equation, filtered_rho0, realized_measure)

        dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho_test_set, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))

        for key in get_keys():
            plt.plot(dictionary_filtered_rho_test_set[key][0])

        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv,
                                                                          dictionary_filtered_rho_test_set, n,
                                                                          test_set_size, N_)




    Q = RunParameters.Q
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
        plt.plot(np.quantile(portfolio_index_simulated, q=1-q, axis=1))

    for q in Q:
        value_at_risk_dictionary[q] = np.quantile(portfolio_index_simulated, q=1-q, axis=1)


    portfolio_index_return = (data.values @ weights).reshape((-1,))

    for q in Q:
        plt.figure(100 * q)
        plt.plot(portfolio_index_return[-test_set_size:])
        plt.plot(value_at_risk_dictionary[q])
        plt.legend()
        plt.show()

    return value_at_risk_dictionary


def get_all_PITs(T, n, data, dicParameters_estimated, marginal_models_list, skip_idx):
    # get PITs from estimated marginal models for test set
    PITs = np.zeros((T, n))
    for j in range(n):
        PITs_j = get_PITs_with_estimated_parameters(data.iloc[:, j], dicParameters_estimated[j + 1],
                                                    marginal_models_list[j])
        PITs[:, j] = PITs_j[skip_idx:]

    return PITs

def get_value_at_risk_output_MV(test_set_size, portfolio_returns):
    Q = RunParameters.Q
    value_at_risk_dictionary = {}
    np.random.seed(1991)

    for q in Q:
        value_at_risk_dictionary[q] = np.zeros(test_set_size)

    for i in range(test_set_size):
        t = test_set_size - i
        sigma = np.std(portfolio_returns[:-t])
        for q in Q:
            value_at_risk_dictionary[q][i] = norm.ppf(q=1-q) * sigma

    return value_at_risk_dictionary


def get_value_at_risk_output_single_garch(test_set_size, portfolio_returns, N, distribution):
    Q = RunParameters.Q
    value_at_risk_dictionary = {}
    np.random.seed(1991)

    mean_equation = constant_mean_equation
    garch_model = MarginalObject(distribution_module_epsilon=distribution, volatility_equation=garch_11_equation,
                                 mean_equation=mean_equation)

    bounds = [[-np.inf, np.inf], [-np.inf, np.inf], [0, 1], [0, 1]]
    if distribution == student_t:
        bounds = [[-np.inf, np.inf]] + bounds

    garch_model.set_bounds(bounds)
    garch_model.set_constraints([eq_cons_garch2])
    garch_model.set_data(portfolio_returns[:-test_set_size])
    garch_model.fit()

    garch_model.set_data(portfolio_returns)
    mu_, sigma2_ = garch_model.filter()
    mu, sigma2 = mu_[-test_set_size:], sigma2_[-test_set_size:]

    for q in Q:
        quantile = garch_model.distribution_module.ppf(garch_model.distribution_parameters, 1 - q)
        value_at_risk_dictionary[q] = quantile * np.sqrt(sigma2) + mu

    return value_at_risk_dictionary


def get_value_at_risk_output_HS(test_set_size, portfolio_returns, N):
    Q = RunParameters.Q
    value_at_risk_dictionary = {}
    np.random.seed(1991)

    for q in Q:
        value_at_risk_dictionary[q] = np.zeros(test_set_size)

    for i in range(test_set_size):
        t = test_set_size - i
        resampled_portfolio_returns = np.random.choice(portfolio_returns[:-t], N)
        for q in Q:
            value_at_risk_dictionary[q][i] = np.quantile(resampled_portfolio_returns, q=1-q)

    return value_at_risk_dictionary


def get_value_at_risk_output_garch(list_marginal_models, test_set_size, dicEstimated_parameters, data, weights, N):
    n = len(list_marginal_models)
    Q = RunParameters.Q
    value_at_risk_dictionary = {}

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

    return value_at_risk_dictionary

def value_at_risk_workflow(marginal_models_list, test_set_size, daily_returns,
                           weights, N, portfolio_returns, rv, train_idx, cpar_equation, copula_type,
                           dictionary_parameter_estimates, dictionary_filtered_rho0, realized_measure):

    # ## test value-at-risk output GARCH model
    # value_at_risk_output_garch = get_value_at_risk_output_garch(marginal_models_list, test_set_size,
    #                                                             dictionary_estimated_parameters_marginal,
    #                                                             daily_returns.values, weights, N=N)

    ## test value-at-risk output dynamic copula model
    if RunParameters.equal_weighting:
        value_at_risk_output = compute_value_at_risk(daily_returns, rv, weights, cpar_equation, copula_type,
                                                     marginal_models_list,
                                                     dictionary_parameter_estimates, dictionary_filtered_rho0,
                                                     test_set_size, N=N, realized_measure=realized_measure)
    else:
        value_at_risk_output = get_mv_weights_value_at_risk_output(daily_returns, rv, weights, cpar_equation, copula_type, marginal_models_list,
                                            dictionary_parameter_estimates, dictionary_filtered_rho0,
                                            test_set_size, N=N, realized_measure=realized_measure)

    return value_at_risk_output




def get_mv_weights_value_at_risk_output(data, lrv, weights, cpar_equation, copula_type, marginal_models_list, dicParameters_estimated, filtered_rho0,
                          test_set_size, N=1000, realized_measure=None):
    #
    # df = pd.read_csv('../datasets/excess_returns.csv')
    # ldf = df.groupby('TICKER')

    T = data.shape[0]
    if copula_type == 'gaussian':
        h_function = h_function_gaussian
        h_function_inv = h_function_inv_gaussian
    elif copula_type == 'student_t':
        h_function = h_function_student_t
        h_function_inv = h_function_inv_student_t
    elif copula_type == 'clayton':
        h_function = h_function_clayton
        h_function_inv = h_function_inv_clayton

    n = data.shape[1]
    training_set_size = T - test_set_size

    ## filter the rho vectors over time (non-path dependent)
    ldist = RunParameters.ldist
    PITs_test, sigma2_test = get_all_PITs_test_realEGARCH(test_set_size, n, data, dicParameters_estimated, ldist, lrv)

    ## obtain PITs_test for the test set in order to forecast value-at-risk figures
    if RunParameters.estimate_static_vine:
        dictionary_h, dictionary_h_inv = h_set_all_same(dicParameters_estimated, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))
        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine(dictionary_h, dictionary_h_inv,
                                                                        dicParameters_estimated, n,
                                                                        test_set_size * N_).reshape(
                (N_, test_set_size, n))

    else:
        dictionary_filtered_rho_test_set = get_filtered_rho_after_estimation(PITs_test, h_function,
                                                                             dicParameters_estimated, n,
                                                                             cpar_equation, filtered_rho0,
                                                                             realized_measure)

        dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho_test_set, h_function, h_function_inv)
        N_ = 1000
        PITs_test_set = np.zeros((N, test_set_size, n))

        for key in get_keys():
            plt.plot(dictionary_filtered_rho_test_set[key][0])

        for i in range(int(N / N_)):
            PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv,
                                                                          dictionary_filtered_rho_test_set, n,
                                                                          test_set_size, N_)


    ldist = RunParameters.ldist
    constraint = {'type':'eq', 'fun':lambda par: sum(par)-1}
    R = np.mean(data.values, axis=0).reshape((-1,1))
    W = np.zeros((test_set_size, RunParameters.nvar))

    Q = RunParameters.Q
    value_at_risk_dictionary = {}
    for q in Q:
        value_at_risk_dictionary[q] = np.zeros(test_set_size)

    par = []
    for j in range(n):
        rt, rv = data.iloc[:-test_set_size, j] - R[j], lrv[j + 1][:-test_set_size]
        loght, logx, zt = filter_realEGARCH(dicParameters_estimated[j + 1], rt, rv)
        x0 = [4]
        if ldist[j + 1] == skewed_t:
            x0 += [1]

        res = minimize(fun=lambda *args: -ldist[j + 1].loglik(*args), x0=x0, args=(zt[:-test_set_size], None, None),
                       bounds=[[2.1, 60], [0.01, 100]], method='BFGS')
        par.append(list(res.x))

    for t in range(test_set_size):
        ## simulate all stock returns
        simulated_PITs = PITs_test_set[:, t, :]
        lepsilon = [ldist[j + 1].ppf(par[j], simulated_PITs[:, j]) for j in range(n)]
        lreturns = [R[j] + lepsilon[j] * np.sqrt(sigma2_test[t, j]) for j in range(RunParameters.nvar)]
        returns = np.vstack(lreturns)

        ## optimize weights
        S = np.cov(returns)
        res = minimize(fun=mv_criterion_function, bounds=[[0,1]]*RunParameters.nvar, x0=[.2, .2, .2, .2, .2], constraints=constraint, method='SLSQP',
                       args=(R, S))
        W[t, :] = res.x

        portfolio_index_simulated = np.array(W[t, :]).reshape((1,-1)) @ returns
        ## compute VaR
        for q in Q:
            value_at_risk_dictionary[q][t] = np.quantile(portfolio_index_simulated, q=1-q)
    print()
    sstatic = RunParameters.estimate_static_vine * 'static_'
    srealized = (1 - RunParameters.skip_realized) * 'realized_'
    with open('MV_' + RunParameters.copula_type + sstatic + srealized + 'weights.pkl', 'wb') as f:
        pickle.dump({'W':W}, f)

    return value_at_risk_dictionary

def mv_criterion_function(W, R, S):
    W = np.array(W).reshape((-1,1))
    criterion_value = W.T @ S @ W
    return criterion_value

def mv_criterion_function_sharpe(W, R, S):
    W = np.array(W).reshape((-1,1))
    criterion_value = R.T @ W / W.T @ S @ W
    return - criterion_value