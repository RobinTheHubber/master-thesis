from marginal_engine.distributions import student_t
from marginal_engine.garch_model import garch_11_equation, eq_cons_garch2, constant_mean_equation
from vine_copula_engine.simulate_vine_algorithm import h_set_all_same, sample_from_vine3D, sample_from_vine
from vine_copula_engine.vine_copula_estimation import *
import matplotlib.pyplot as plt
from marginal_engine.marginal_model import MarginalObject


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
    Q = [0.9, 0.95, 0.99]
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
    Q = [0.9, 0.95, 0.99]
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
    Q = [0.9, 0.95, 0.99]
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
    Q = [0.9, 0.95, 0.99]
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

def value_at_risk_workflow(marginal_models_list, test_set_size, dictionary_estimated_parameters_marginal, daily_returns,
                           weights, N, portfolio_returns, train_idx, cpar_equation, copula_type,
                           dictionary_parameter_estimates, dictionary_filtered_rho0, realized_measure):

    # ## test value-at-risk output GARCH model
    # value_at_risk_output_garch = get_value_at_risk_output_garch(marginal_models_list, test_set_size,
    #                                                             dictionary_estimated_parameters_marginal,
    #                                                             daily_returns.values, weights, N=N)

    ## test value-at-risk output dynamic copula model
    value_at_risk_output = compute_value_at_risk(daily_returns, weights, cpar_equation, copula_type,
                                                 marginal_models_list,
                                                 dictionary_parameter_estimates, dictionary_filtered_rho0,
                                                 test_set_size, N=N, realized_measure=realized_measure)

    return value_at_risk_output


