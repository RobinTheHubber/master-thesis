import numpy as np
from scipy.stats import norm, t as student_t_module, kendalltau
from scipy.special import loggamma
from utility.mapping import map_logistic

######################
## copula density's ##
######################
"""""
define copula density in log form for numerical convenience 
"""""
def copula_density_independence(rho, u1, u2):
    llik = sum(np.log(u1) + np.log(u2))
    return -llik

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
def h_function_independence(rho, x, v):
    return x

def h_function_gaussian(rho, x, v):
    x1, x2 = norm.ppf((x, v))
    h_function_value = norm.cdf((x1 - rho * x2) / np.sqrt(1 - rho ** 2))
    if np.sum(np.isnan(h_function_value))>0:
        print('')
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
def h_function_inv_independence(rho, x, v):
    return x

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
