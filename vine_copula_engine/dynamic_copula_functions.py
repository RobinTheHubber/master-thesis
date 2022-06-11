import numpy as np
import matplotlib.pyplot as plt
from utility.mapping import map_logistic
from main_engine.run_parameters import RunParameters
evolution_type = RunParameters.evolution_type
npar_evolution = RunParameters.get_evolution_npar()
obs_pre_est = RunParameters.get_pre_obs_est()
# def par_filter(par, rho0, vX, vV, operation='difference'):
#     xi, phi_2 = par
#     phi_1 = 0
#     T = len(vX)
#     rho = np.zeros(T)
#     rho[0] = rho0
#     if operation == 'product':
#         rho_next = lambda rho, x, v: xi + phi_1 * rho + phi_2 * x * v
#     elif operation == 'difference':
#         rho_next = lambda rho, x, v: xi + phi_1 * rho + phi_2 * abs(x - v)
#
#     for t in range(1, T):
#         rho[t] = rho_next(rho[t - 1], vX[t - 1], vV[t - 1])
#
#     return map_logistic(rho, -1, 1, backwards=False)


#
# vX=dictionary_v[(0, i)]; vV=dictionary_v[(0, i + 1)]
# rho0 = np.corrcoef(vV, vX)[0,1]
# par1 = par_node
# par2 = [0, 0.9, 0.4]
# rho1=par_filter(par1, rho0, vX, vV, operation='product')
# rho2=par_filter(par2, rho0, vX, vV, operation='product')
# print(par1)
# plt.plot(rho1)
# plt.plot(rho2)
# plt.show()

#######################
## filter parameter
# evolution equations
# #######################
def par_filter(par, rho0, vX, vV, mapping_par, operation='difference'):
    xi, phi_1, phi_2 = par #, phi_2, phi_3 = par
    T = len(vX)
    rho = np.zeros(T+1)
    a,b = mapping_par
    rho[:obs_pre_est] = map_logistic(rho0, a, b, backwards=True)

    # rho_next = lambda d_day, d_week, d_month: xi + phi_1 * d_day + phi_2 * d_week + phi_3 * d_month
    rho_next = lambda v, x, rho: xi + phi_1 * rho + phi_2 * abs(x - v)

    for t in range(obs_pre_est, T+1):
        rho[t] = rho_next(vV[t-1], vX[t-1], rho[t-1])

    return map_logistic(rho, a, b, backwards=False)



def par_filter_realized(par, rho0, realized_measure, mapping_par, vX, vV):
    a,b = mapping_par
    # realized_measure_mapped = map_logistic(realized_measure, a, b, backwards=True)
    h_ij_hat = realized_measure
    T = len(realized_measure)
    rho = np.zeros(T)
    rho[:obs_pre_est] = map_logistic(rho0, a, b, backwards=True)

    if evolution_type == 'HAR':
        xi, phi_1, phi_2, phi_3, phi_4 = par
        def rho_next(t):
            d_day = rho[t-1]
            d_week = np.mean(rho[t - 5:t-1])
            d_month = np.mean(rho[t - 22:t-5])
            return xi + phi_1 * d_day + phi_2 * d_week + phi_3 * d_month + phi_4 * h_ij_hat[t] #+ phi_5 * h_i + phi_6 * h_j

    elif evolution_type == 'simple':
        xi, phi_1, phi_2 = par
        def rho_next(t):
            return xi + phi_1 * rho[t-1] + phi_2 * h_ij_hat[t]

    elif evolution_type == 'simple_ar':
        xi, phi, phi_1, phi_2, phi_3 = par
        def rho_next(t, rho):
            d_day_rho = np.mean(realized_measure_mapped[t - 1:t])
            d_day_c1 = np.mean(realized_cov1_mapped[t - 1:t])
            d_day_c2 = np.mean(realized_cov2_mapped[t - 1:t])
            return xi + phi * rho + phi_1 * d_day_rho + phi_2 * d_day_c1 * d_day_c2


    for t in range(obs_pre_est, T):
        rho[t] = rho_next(t)

    return map_logistic(rho, a, b, backwards=False)

# def par_filter(par, rho0, vX, vV, operation='difference'):
#     xi, phi = par
#     T = len(vX)
#     rho = np.zeros(T)
#     rho[0] = map_logistic(rho0, -1, 1, backwards=True)
#     if operation=='product':
#         rho_next = lambda x, v: xi + phi * x * v
#     elif operation=='difference':
#         rho_next = lambda x, v: xi + phi * abs(x - v)
#
#     for t in range(1, T):
#         rho[t] = rho_next(vX[t-1], vV[t-1])
#
#     return map_logistic(rho[1:], -1, 1, backwards=False)

######################
## wrappers for dynamic
# copula model
######################

def filter_copula_parameters(par, u1, u2, cpar_equation, rho0=None, realized_measure=None):
    K = int(len(par) / npar_evolution)
    T = len(u1)
    copula_par = np.zeros((K, T+1))

    if rho0 is None:
        for k in range(K):
            if k == 0:
                copula_par[0, 0] = np.corrcoef(u1, u2)[0, 1]

            if k == 1:
                copula_par[1, 0] = 4


    for k in range(K):
        # todo: make use of static vine copula estimated parameters for x0 for dynamic vine copula
        # it will be pretty similar to sample correlation between v1 and v2 but for student-t might be a bit more difficult
        if k == 0:
            mapping_par = (-0.999999, .999999)
        if k == 1:
            mapping_par = (2, 60)

        if RunParameters.skip_realized:
            copula_par[k, :] = par_filter(par[npar_evolution * k:npar_evolution * k + npar_evolution], copula_par[k, :obs_pre_est], u1, u2, mapping_par, cpar_equation)
        else:
            copula_par[k, :] = par_filter_realized(par[npar_evolution * k:npar_evolution * k + npar_evolution], copula_par[k, :obs_pre_est], realized_measure, mapping_par, u1, u2)

    return copula_par


def copula_density_dynamic(par, u1, u2, copula_density, cpar_equation, realized_measure=None):
    if RunParameters.estimate_static_vine:
        negative_llik = copula_density(par, u1, u2)
        return negative_llik

    else:
        K = int(len(par)/npar_evolution)
        par = np.hstack([transformation_dynamic_equation(par[npar_evolution*k:(k+1)*npar_evolution]) for k in range(K)])
        copula_par = filter_copula_parameters(par, u1, u2, cpar_equation, realized_measure=realized_measure)
        T = len(u1)
        negative_llik = copula_density(copula_par[:, obs_pre_est:T], u1[obs_pre_est:], u2[obs_pre_est:])

    return negative_llik


def copula_h_function_dynamic(par, u1, u2, cpar_equation, h_function, rho0=None, output_copula_par=False, realized_measure=None):
    if RunParameters.estimate_static_vine:
        h_function_value = h_function(par, u1, u2)
        return h_function_value

    else:
        T = len(u1)
        copula_par = filter_copula_parameters(par, u1, u2, cpar_equation, rho0, realized_measure)
        h_function_value = h_function(copula_par[:, :T], u1, u2).reshape((-1,))

    if output_copula_par:
        return h_function_value, copula_par[:, :T]
    else:
        return h_function_value


def transformation_dynamic_equation(par, backwards=False):
    par_ = par.copy()
    # if evolution_type == 'simple_ar':
    #     par_[1] = map_logistic(par[1], -0.99, 0.99, backwards=backwards)

    # if evolution_type == 'HAR':
    #     par_[1] = map_logistic(par[1], 0, 1, backwards=backwards)
    #     par_[2] = map_logistic(par[2], 0, 1, backwards=backwards)
    #     par_[3] = map_logistic(par[3], 0, 1, backwards=backwards)
    #     par_[4] = map_logistic(par[4], 0, 1, backwards=backwards)

    # par_[2] = map_logistic(par[1], -1, 1, backwards=backwards)
    # par_[3] = map_logistic(par[1], -1, 1, backwards=backwards)
    return par_

