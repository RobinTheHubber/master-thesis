from scipy.stats import kstest

from forecasting_VaR.forecasting import  *
from forecasting_VaR.testing_value_at_risk import analyze_value_at_risk_output
from main_engine.estimation_and_var_workflows import estimation_and_var_workflow
from main_engine.get_data import load_return_models_and_data, load_return_data, load_realized_cov
from main_engine.run_parameters import loop_over_run_parameters
from marginal_engine.realGARCH import *
from utility.util import get_keys


# kstest(PITs[:, 0], 'uniform')


def plot_filtered_parameters_vine_copula(filtered_rho):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax1, ax2 = axs
    for k, v in filtered_rho.items():
        ax1.plot(v[0,:], label=k)
        ax2.plot(v[1,:], label=k)
    plt.legend()
    plt.show()



def methods_of_moments_function(par, hij_hat, marginal_models_list, PITs):
    h_function = RunParameters.get_copula_h_function()
    h_function_inv = RunParameters.get_copula_h_inv_function()
    n = RunParameters.nvar
    N = RunParameters.N_mm
    T = RunParameters.estimation_window
    evolution_npar = RunParameters.get_evolution_npar()
    cpar_equation = RunParameters.evolution_type
    keys = get_keys()

    realized_measure = dict(zip(keys, [[np.array([0]*T), np.array([0]), np.array([0])]]*len(keys)))
    dictionary_theta = dict(zip(keys, [par[i*evolution_npar:(i+1)*npar_evolution] for i in range(len(keys))]))

    dictionary_filtered_rho = get_filtered_rho_before_estimation(PITs, h_function, dictionary_theta, n, cpar_equation, realized_measure)
    dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_filtered_rho, h_function, h_function_inv)
    N_ = 1000
    vine_data = np.zeros((N, T, n))
    for i in range(int(N/N_)):
        vine_data[i*N_:(i+1)*N_, :, :] = sample_from_vine3D(dictionary_h, dictionary_h_inv, dictionary_filtered_rho, n, T, N_)

    sigma_ij_hat = {}
    for key in keys:
        sigma_ij_hat[key] = np.zeros(T)

    mu_and_sigma2 = [marginal_model.filter() for marginal_model in marginal_models_list]
    mu = np.vstack([filtered_output[0] for filtered_output in mu_and_sigma2])
    sigma2 = np.vstack([filtered_output[1] for filtered_output in mu_and_sigma2])

    for t in range(T):
        marginal_data_simulated = np.zeros((N, n))
        for j in range(n):
            marginal_model = marginal_models_list[j]
            simulated_PITs_j = vine_data[:, t, j]
            epsilon = marginal_model.distribution_module.ppf(marginal_model.distribution_parameters, simulated_PITs_j)
            marginal_data_simulated[:, j] = mu[j, t] + epsilon * np.sqrt(sigma2[j, t])

        for key in keys:
            i, j = key
            k = i + j
            sigma_ij_hat[key][t] = np.cov(marginal_data_simulated[:, i-1], marginal_data_simulated[:, k-1])[0, 1]


    res = np.sum([sum((sigma_ij_hat[key][obs_pre_est:] - hij_hat[key].reshape((-1,)))**2) for key in keys])
    return res

def method_of_moments_workflow(hij_hat):
    # 0. let copula parameter be fixed
    # 1. simulate from first tree component with same random numbers M times
    # 2. transform ui, ui+j to Xi and Xi+j
    # 3. estimate covariance matrix given copula parameter
    # repeat above steps iteratively to minimize the criterion function (gij - hij)**2 to solve for copula parameter

    nvar = RunParameters.nvar
    mean_equation = RunParameters.mean_equation
    train_idx = RunParameters.estimation_window
    distribution = RunParameters.get_marginal_distribution()
    distribution_str = RunParameters.distribution
    T = RunParameters.estimation_window

    with open('parameters_' + distribution_str + '_gaussian_difference_inv.pkl', 'rb') as f:
        dictionary_parameters = pickle.load(f)

    marginal_models_list, _, daily_returns = load_return_models_and_data(distribution, train_idx, mean_equation, nvar)
    PITs = get_all_PITs(T, nvar, daily_returns.iloc[:T, :], dictionary_parameters, marginal_models_list, skip_idx=0)
    x0 = get_vine_x0_mm()
    res = minimize(methods_of_moments_function, method='BFGS', x0=x0, args=(hij_hat, marginal_models_list, PITs))


def main():
    # data = load_return_data(RunParameters.nvar)
    #
    # cov , sig2 = load_realized_cov(5)
    # check_skewnewss_in_realEGARCH(data.iloc[:, 0], sig2)
    # for i in range(2, RunParameters.nvar+1):
    #     dictionary_realized_measure, list_sigma2 = load_realized_cov(RunParameters.nvar)
    #     par = estimate_realEGARCH(data.iloc[:, i-1], list_sigma2[i])
    #     # pits = get_PITs_realGARCH(data.iloc[:, i-1], list_sigma2[i], par)
    #     # print(kstest(pits, 'uniform'))
    #
#
# # u ** (-theta) + 1 = u1 ** (-theta) + u2 ** (-theta)
# theta = 1.5
# N = 10000
# u, u1 = np.random.random((2,N))
# u2 = h_function_inv_clayton(theta, u, u1)
# plt.scatter(u1, u2)
# plt.show()



    def perform_complete_workflow():
        estimation_and_var_workflow()

    RunParameters.run_model = False
    RunParameters.run_var = True
    vine_choice = {'static': 1, 'dynamic':1, 'realized': 1}
    copulas = ['clayton', 'gaussian']


    ## get output
    RunParameters.equal_weighting = False
    loop_over_run_parameters(perform_complete_workflow, vine_choice, copulas)
    RunParameters.equal_weighting = True
    loop_over_run_parameters(perform_complete_workflow, vine_choice, copulas)

    ## test outoput
    RunParameters.equal_weighting = True
    analyze_value_at_risk_output(vine_choice, copulas)
    RunParameters.equal_weighting = False
    analyze_value_at_risk_output(vine_choice, copulas)



if __name__ == '__main__':
    main()
