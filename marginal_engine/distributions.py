import numpy as np
from scipy.stats import norm, t, skewnorm
from mapping import *
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
       return t.ppf(x, df=nu, loc=0, scale= np.sqrt((nu-2) / nu))

    @staticmethod
    def loglik(par, x, array_sigma2, array_mu):
        nu = par[0]
        y = (x-array_mu) / np.sqrt(array_sigma2)
        logpdf = - 1 / 2 * np.log((nu-2) * np.pi * array_sigma2) + loggamma((nu + 1) / 2) - loggamma(nu/2) - (nu + 1) / 2 * np.log(1 + y**2 / (nu-2))
        loglik = sum(logpdf)
        return loglik

    @staticmethod
    def cdf(par, x):
        nu = par[0]
        return t.cdf(x, df=nu, scale=np.sqrt((nu-2) / nu))

class skewed_normal:

    @staticmethod
    def get_initial_parameters():
        return [...]

    @staticmethod
    def get_npar():
        return 1

    @staticmethod
    def ppf(par, x):
        pass


    @staticmethod
    def loglik(par, x):
        pass


    @staticmethod
    def cdf(par, x):
        pass

class skewed_t:
    # todo: create skewed-t distribution

    @staticmethod
    def get_initial_parameters():
        return [5, ...]

    @staticmethod
    def get_npar():
        return 1

    @staticmethod
    def ppf(x):
        pass


    @staticmethod
    def loglik(x):
        pass


    @staticmethod
    def cdf(x):
        pass