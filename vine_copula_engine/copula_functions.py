import numpy as np
from scipy.stats import norm, t as student_t_module, kendalltau
from scipy.special import loggamma
from utility.mapping import map_logistic
import matplotlib.pyplot as plt
from main_engine.run_parameters import RunParameters
evolution_type = RunParameters.evolution_type
if evolution_type == 'simple':
    npar_evolution = 2
    obs_pre_est = 1
elif evolution_type == 'HAR':
    npar_evolution = 4
    obs_pre_est = 22

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
    xi, phi_1 = par #, phi_2, phi_3 = par
    T = len(vX)
    rho = np.zeros(T+1)
    a,b = mapping_par
    rho[:obs_pre_est] = map_logistic(rho0, a, b, backwards=True)
    if operation == 'difference':
        tr = lambda x, y: abs(x - y)
    elif operation == 'difference_inv':
        tr = lambda x, y: 1 / (1 + abs(x - y))

    # rho_next = lambda d_day, d_week, d_month: xi + phi_1 * d_day + phi_2 * d_week + phi_3 * d_month
    rho_next = lambda d_day : xi + phi_1 * d_day

    for t in range(obs_pre_est, T+1):
        rho[t] = rho_next(
            np.mean(tr(vV[t-1:t], vX[t-1:t]))
            # np.mean(tr(vV[t-5:t], vX[t-5:t])),
            # np.mean(tr(vV[t-22:t], vX[t-22:t]))
        )

    return map_logistic(rho, a, b, backwards=False)



def par_filter_realized(par, rho0, realized_measure, mapping_par):
    if evolution_type == 'HAR':
        xi, phi_1, phi_2, phi_3 = par
        def rho_next(t):
            d_day = np.mean(realized_measure_mapped[t-1:t])
            d_week = np.mean(realized_measure_mapped[t - 5:t])
            d_month = np.mean(realized_measure_mapped[t - 22:t])
            return xi + phi_1 * d_day + phi_2 * d_week + phi_3 * d_month

    elif evolution_type == 'simple':
        xi, phi_1 = par
        def rho_next(t):
            d_day = np.mean(realized_measure_mapped[t-1:t])
            return xi + phi_1 * d_day

    T = len(realized_measure)
    rho = np.zeros(T+1)
    a,b = mapping_par
    rho[:obs_pre_est] = map_logistic(rho0, a, b, backwards=True)
    realized_measure_mapped = map_logistic(realized_measure, a, b, backwards=True)

    for t in range(obs_pre_est, T+1):
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

    if rho0 is not None:
        copula_par[:, 0] = rho0
    else:
        for k in range(K):
            if k == 0:
                copula_par[0, 0] = np.corrcoef(u1, u2)[0, 1]

            if k == 1:
                copula_par[1, 0] = 4


    for k in range(K):
        # todo: make use of static vine copula estimated parameters for x0 for dynamic vine copula
        # it will be pretty similar to sample correlation between v1 and v2 but for student-t might be a bit more difficult
        if k == 0:
            mapping_par = (-1, 1)
        if k == 1:
            mapping_par = (2, 60)

        if realized_measure is None:
            copula_par[k, :] = par_filter(par[npar_evolution * k:npar_evolution * k + npar_evolution], copula_par[k, 0], u1, u2, mapping_par, cpar_equation)
        else:
            copula_par[k, :] = par_filter_realized(par[npar_evolution * k:npar_evolution * k + npar_evolution], copula_par[k, 0], realized_measure, mapping_par)

    return copula_par


def copula_density_dynamic(par, u1, u2, copula_density, cpar_equation, realized_measure=None):
    if cpar_equation is None:
        negative_llik = copula_density(par, u1, u2, transformation=True)
    else:
        K = int(len(par)/npar_evolution)
        par = np.hstack([transformation_dynamic_equation(par[npar_evolution*k:(k+1)*npar_evolution]) for k in range(K)])
        copula_par = filter_copula_parameters(par, u1, u2, cpar_equation, realized_measure=realized_measure)
        T = len(u1)
        negative_llik = copula_density(copula_par[:, obs_pre_est:T], u1[obs_pre_est:], u2[obs_pre_est:])

    return negative_llik


def copula_h_function_dynamic(par, u1, u2, cpar_equation, h_function, rho0=None, output_copula_par=False, realized_measure=None):
    if cpar_equation is None:
        h_function_value = h_function(par, u1, u2, transformation=True)
    else:
        T = len(u1)
        copula_par = filter_copula_parameters(par, u1, u2, cpar_equation, rho0, realized_measure)
        h_function_value = h_function(copula_par[:, T], u1, u2).reshape((-1,))

    if output_copula_par:
        return h_function_value, copula_par[:, :T]
    else:
        return h_function_value


######################
## copula density's ##
######################
"""""
define copula density in log form for numerical convenience 
"""""


def copula_density_gaussian(rho_, u1, u2, transformation=False):
    if transformation:
        rho = transformation_gaussian_copula(rho_)
    else:
        rho = rho_

    x1, x2 = norm.ppf((u1, u2))
    # c = (1-rho**2)**(-0.5) * np.exp(- (rho**2 * (x1**2 + x2**2) - 2 * rho * x1 * x2) / (2*(1-rho**2)))
    copula_log_density = -0.5 * np.log((1 - rho ** 2)) - (rho ** 2 * (x1 ** 2 + x2 ** 2) - 2 * rho * x1 * x2) / (
            2 * (1 - rho ** 2))
    llik = np.sum(copula_log_density)
    return -llik


def copula_density_student_t(par, u1, u2, transformation=False):
    if transformation:
        rho, nu = transformation_student_t_copula(par)
    else:
        rho, nu = par

    m = 2
    # kendall_tau = kendalltau(u1, u2)
    # rho = np.sin(np.pi / 2 * kendall_tau.correlation)
    x1, x2 = student_t_module.ppf((u1, u2), df=nu)
    copula_log_density = - 1 / 2 * np.log(1 - rho ** 2) + loggamma((nu + m) / 2) + (m - 1) * loggamma(
        nu / 2) - m * loggamma((nu + 1) / 2) \
                         - (nu + m) / 2 * np.log(1 + 1 / (1 - rho ** 2) * (x1 ** 2 + x2 ** 2 - 2 * rho * x1 * x2) / nu) \
                         + (nu + 1) / 2 * (np.log(1 + x1 ** 2 / nu) + np.log(1 + x2 ** 2 / nu))

    llik = np.sum(copula_log_density)
    return -llik


######################
## h function's ######
######################
"""""
h-function is defined as h(x,v):= derivative of C(x,v) wrt v
"""""


def h_function_gaussian(rho, x, v):
    x1, x2 = norm.ppf((x, v))
    h_function_value = norm.cdf((x1 - rho * x2) / np.sqrt(1 - rho ** 2))
    return h_function_value

def h_function_student_t(par, x, v):
    # rho, nu = transformation_student_t_copula(par)
    rho, nu = par
    x1, x2 = student_t_module.ppf((x, v), df=nu)
    numerator = x1 - rho * x2
    denominator = np.sqrt((nu + x2 ** 2) * (1 - rho ** 2) / (nu + 1))
    h_function_value = student_t_module.cdf(numerator / denominator, df=nu + 1)
    return h_function_value


######################
## h-inv function's ######
######################
def h_function_inv_student_t(par, x, v):
    rho, nu = par
    x1 = student_t_module.ppf(x, df=nu + 1)
    x2 = student_t_module.ppf(v, df=nu)
    h_inv_value = student_t_module.cdf(x1 * np.sqrt((nu + x2 ** 2) * (1 - rho ** 2) / (nu + 1)) + x2 * rho, df=nu)
    return h_inv_value


def h_function_inv_gaussian(rho, x, v):
    x1, x2 = norm.ppf((x, v))
    h_inv_value = norm.cdf(x1 * np.sqrt(1 - rho ** 2) + x2 * rho)
    return h_inv_value


##############################
## transformation functions ##
##############################
def transformation_gaussian_univariate(par, backwards=False):
    par_transformed = par.copy()
    if backwards:
        par_transformed[1] = np.log(par_transformed[1])
    else:
        par_transformed[1] = np.exp(par_transformed[1])

    return par_transformed


def transformation_dynamic_equation(par, backwards=False):
    par_ = par.copy()
    # par_[1] = map_logistic(par[1], -0.99, 0.99, backwards=backwards)
    # par_[2] = map_logistic(par[1], -1, 1, backwards=backwards)
    # par_[3] = map_logistic(par[1], -1, 1, backwards=backwards)
    return par_


def transformation_gaussian_copula(par, backwards=False):
    if backwards:
        x = (par + 1) / 2
        par = np.log(x / (1 - x))
    else:
        par = 2 * np.exp(par) / (1 + np.exp(par)) - 1

    return par


def transformation_student_t_copula(par, backwards=False):
    rho_, nu_ = par
    if backwards:
        x_rho = (rho_ + 1) / 2
        rho = np.log(x_rho / (1 - x_rho))
        x_nu = (nu_ - 2) / 30
        nu = np.log(x_nu / (1 - x_nu))

    else:
        rho = 2 * np.exp(rho_) / (1 + np.exp(rho_)) - 1
        nu = 30 * np.exp(nu_) / (1 + np.exp(nu_)) + 2

    par = rho, nu
    if np.isnan(nu):
        print('')

    return par

#
#
# def tn_mv(par, backwards=False):
#     ## of the flat vector, first elements are mu (2x1), remaining are cov (2x2)
#     par_transformed = par.copy()
#     if backwards:
#         par_transformed[1] = np.linalg.cholesky(par_transformed[1])
#         par_transformed = np.append(par_transformed[0], par_transformed[1].flatten()[[0,2,3]]) # leave one parameter out due to symmetry in cov
#     else:
#         par_transformed = [par_transformed[:2], np.array([par_transformed[2], par_transformed[3], par_transformed[3], par_transformed[4]]).reshape((2, 2))]
#         par_transformed[1] = par_transformed[1] @ par_transformed[1].T
#
#     return par_transformed
