
from vine_copula_engine.simulate_vine_algorithm import h_set_all_same, sample_from_vine3D
from vine_copula_engine.vine_copula_estimation import *
import matplotlib.pyplot as plt
from marginal_engine.marginal_model import MarginalObject


def get_PITs_with_estimated_parameters(data, estimated_parameters, marginal_model: MarginalObject):
    marginal_model.mean_equation.update_parameters(estimated_parameters['mean'])
    marginal_model.volatility_equation.update_parameters(estimated_parameters['vol'])

    if marginal_model.n1 > 0:
        marginal_model.distribution_parameters = estimated_parameters['dist']
    else:
        marginal_model.distribution_parameters = None

    marginal_model.set_data(data)
    marginal_model.compute_pits()
    return marginal_model.PITs


def compute_value_at_risk(data, cpar_equation, copula_type, marginal_models_list, dicParameters_estimated, filtered_rho0,
                          test_set_size, N=1000):
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
                                                                         cpar_equation, filtered_rho0)

    dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho_test_set, h_function, h_function_inv)
    N_ = 1000
    PITs_test_set = np.zeros((N, test_set_size, n))
    for i in range(int(N / N_)):
        PITs_test_set[i * N_:(i + 1) * N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv,
                                                                      dictionary_filtered_rho_test_set, n,
                                                                      test_set_size, N_)

    Q = [0.9, 0.95, 0.975, 0.99, 0.995]
    value_at_risk_dictionary = {}
    for q in Q:
        value_at_risk_dictionary[q] = np.zeros((test_set_size, n))

    for j in range(n):
        marginal_model = marginal_models_list[j]
        simulated_PITs_j = PITs_test_set[:, :, j].T
        epsilon = norm.ppf(simulated_PITs_j)
        marginal_data_simulated = np.zeros((test_set_size, N))
        mu_, sigma2_ = marginal_model.filter()
        mu, sigma2 = mu_[training_set_size:], sigma2_[training_set_size:]
        for t in range(test_set_size):
            marginal_data_simulated[t] = mu[t] + epsilon[t, :] * np.sqrt(sigma2[t])

        for q in Q:
            value_at_risk_dictionary[q][:, j] = np.quantile(marginal_data_simulated, q=1 - q, axis=1)

    plt.plot(data.values[training_set_size:, 0])
    for q in Q:
        plt.plot(value_at_risk_dictionary[q][:, 0])
    plt.show()

    return value_at_risk_dictionary


def compute_value_at_risk_test_output(data, value_at_risk):
    n = data.shape[1]
    Q = list(value_at_risk.keys())
    dictionary_VaR_test_results = {}
    for q in Q:
        for j in range(n):
            dictionary_VaR_test_results[(q, j)] = compute_test_value_at_risk(data.values[:, j], q,
                                                                             value_at_risk[q][:, j])

    return dictionary_VaR_test_results


def compute_test_value_at_risk(data_array, q, value_at_risk_array):
    pi_observed = sum(data_array < value_at_risk_array) / len(data_array)
    pi_theoretical = 1 - q
    nobs = len(data_array)
    return pi_observed


dic_estimated_parameters_all_gaussian = {1: {'dist': None, 'mean': np.array([0.08986738]),
                                             'vol': [8.875398354470813e-05, 0.016544337693967393, 0.9834384776660482]},
                                         2: {'dist': None, 'mean': np.array([0.05430363]),
                                             'vol': [0.010568217871437843, 0.0722154897664453, 0.9266189841757995]},
                                         3: {'dist': None, 'mean': np.array([0.03851587]),
                                             'vol': [0.02736560242260675, 0.08846034728261587, 0.9066551819993721]},
                                         4: {'dist': None, 'mean': np.array([0.02947448]),
                                             'vol': [0.07417158609443929, 0.06544482353201547, 0.9131331762002537]},
                                         5: {'dist': None, 'mean': np.array([0.07538257]),
                                             'vol': [0.043003770394831886, 0.07394814093356221, 0.9036025493117934]},
                                         (1, 1): [0.07703572, 0.85928694, 0.48866848],
                                         (1, 2): [0.05463046, 0.87338248, 0.2397601],
                                         (1, 3): [0.01218772, 1.00777431, -0.07531984],
                                         (1, 4): [-0.00817983, 0.99163046, 0.04963163],
                                         (2, 1): [-0.04284278, 0.97505618, 0.1919873],
                                         (2, 2): [8.77989755e-04, 1.00353930e+00, -1.07978061e-02],
                                         (2, 3): [0.00208407, 0.99146326, 0.00374785],
                                         (3, 1): [0.14318752, 0.09427923, 0.09523164],
                                         (3, 2): [-0.02074908, 0.96878669, 0.13360867],
                                         (4, 1): [-0.00490522, 0.98029199, 0.04044638]}


def test_get_v():
    v = get_filtered_rho_after_estimation(np.random.random(size=2000).reshape(400, 5), h_function_gaussian,
                                          dic_estimated_parameters_all_gaussian, 5, 'product')
    print(v)
