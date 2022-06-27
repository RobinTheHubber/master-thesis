import numpy as np
from scipy.stats import norm, t, skewnorm
from utility.mapping import *
from scipy.special import loggamma


class gaussian:
    @staticmethod
    def get_initial_parameters():
        return []

    @staticmethod
    def get_npar():
        return 0

    @staticmethod
    def ppf(par, x):
        return norm.ppf(x)

    @staticmethod
    def loglik(par, x, array_sigma2, array_mu):
        log_density = - 1 / 2 * np.log(array_sigma2) - 1 / 2 * (x - array_mu) ** 2 / array_sigma2
        loglik = np.sum(log_density[1:])
        return loglik

    @staticmethod
    def cdf(par, x):
        return norm.cdf(x)

    @staticmethod
    def cdf_normal(par, data):
        mu, sigma2 = par
        return norm.cdf(x=data, loc=mu, scale=sigma2)


class student_t:
    @staticmethod
    def transform_parameters(par, backwards=False):
        return [map_logistic(par[0], 1, 100, backwards)]

    @staticmethod
    def get_initial_parameters():
        return [map_logistic(5, 1, 100, backwards=True)]

    @staticmethod
    def get_npar():
        return 1

    @staticmethod
    def ppf(par, x):
        nu = par[0]
        return t.ppf(x, df=nu, loc=0, scale=np.sqrt((nu - 2) / nu))

    @staticmethod
    def loglik(par, x, array_sigma2, array_mu):
        if array_mu is None and array_sigma2 is None:
            array_mu = np.zeros(x.shape)
            array_sigma2 = np.ones(x.shape)

        nu = par[0]
        y = (x - array_mu) / np.sqrt(array_sigma2)
        logpdf = - 1 / 2 * np.log((nu - 2) * np.pi * array_sigma2) + loggamma((nu + 1) / 2) - loggamma(nu / 2) - (
                    nu + 1) / 2 * np.log(1 + y ** 2 / (nu - 2))
        loglik = sum(logpdf)
        return loglik

    @staticmethod
    def cdf(par, x):
        nu = par[0]
        return t.cdf(x, df=nu, scale=np.sqrt((nu - 2) / nu))

class skewed_t:
    @staticmethod
    def transform_parameters(par, backwards=False):
        return [map_logistic(par[0], 1, 100, backwards), map_positive(par[1], backwards)]

    @staticmethod
    def get_initial_parameters():
        return [map_logistic(6, 1, 100, backwards=True), map_positive(1, backwards=True)]

    @staticmethod
    def get_npar():
        return 2

    @staticmethod
    def ppf(par, x):
        nu, xi = par
        threshold = 1 / (2 * xi) * 2 / (xi + 1 / xi)
        s = np.sqrt((nu - 2) / nu)
        y = np.zeros(len(x))
        l = x < threshold
        r = x >= threshold
        y[l] = t.ppf((xi**2 + 1) * x[l] / 2, df=nu, scale=s) / xi
        y[r] = t.ppf(((x[r] * (xi + 1 / xi) / 2) - 1 / (2 * xi)) / xi + 1 / 2, df=nu, scale=s) * xi
        return y

    @staticmethod
    def loglik(par, x, array_sigma2, array_mu):
        if array_mu is None and array_sigma2 is None:
            array_mu = np.zeros(x.shape)
            array_sigma2 = np.ones(x.shape)

        # check if centralized observations are negative or positive and call student t loglikelihood scaled by skewness parameter xi
        nu, xi = par
        xi_ = ((x-array_mu) < 0) * xi + ((x-array_mu) > 0) / xi
        y = (x - array_mu) / np.sqrt(array_sigma2) * xi_
        logpdf = - 1 / 2 * np.log((nu - 2) * np.pi * array_sigma2) - np.log(xi + 1/xi) + loggamma((nu + 1) / 2) - loggamma(nu / 2) - (
                nu + 1) / 2 * np.log(1 + y ** 2 / (nu - 2))
        llik = sum(logpdf)
        return llik

    @staticmethod
    def cdf(par, x):
        nu, xi = par
        s = np.sqrt((nu - 2) / nu)
        cdf = 2 / (xi + 1 / xi) * (
              (x <= 0) * 1 / xi * t.cdf(x*xi, df=nu, scale=s) + \
              (x > 0) * (1 / (2 * xi) + xi * (t.cdf(x / xi, df=nu, scale=s) - t.cdf(0, df=nu, scale=s)))
              )

        return cdf




#
# class skewed_normal:
#     @staticmethod
#     def get_initial_parameters():
#         return [5, ...]
#
#     @staticmethod
#     def get_npar():
#         return 1
#
#     @staticmethod
#     def ppf(x):
#         pass
#
#     @staticmethod
#     def loglik(x):
#         pass
#
#     @staticmethod
#     def cdf(x):
#         pass
