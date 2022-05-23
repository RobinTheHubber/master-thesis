import numpy as np
from scipy.optimize import minimize


class FilterEquation:
    def __init__(self, update_function, initial_values_function, transformation_function, get_initial_parameters_function, npar, constraints=None):
        self.update_function = update_function
        self.initial_values_function = initial_values_function # for filtering e.g. sigma_0
        self.get_initial_parameters_function = get_initial_parameters_function # for llik to get initial parameter estimates
        self.transformation_function = transformation_function
        self.constraints = constraints
        self.npar = npar
        self.constraints = []

    def get_npar(self):
        return self.npar


    def get_initial_values(self):
        return self.initial_values_function(self.par)

    def update(self, data, filter_variable):
        return self.update_function(self.par, data, filter_variable)

    def get_initial_parameters(self, data):
        return self.transformation_function(self.get_initial_parameters_function(data), backwards=True)

    def update_parameters(self, par):
        self.par = par

    def transform_parameters(self):
        self.par = self.transformation_function(self.par)

class MarginalObject:

    def __init__(self, distribution_module_epsilon, volatility_equation, mean_equation, parameters, optimization_method='BFGS'):
        self.distribution_module = distribution_module_epsilon
        self.mean_equation = mean_equation
        self.volatility_equation = volatility_equation
        self.optimization_method = optimization_method

        self.n1 = self.distribution_module.get_npar()
        self.n2 = self.n1 + self.mean_equation.get_npar()

        self.distribution_parameters = parameters[:self.n1]
        self.mean_equation.par = parameters[self.n1:self.n2]
        self.volatility_equation.par = parameters[self.n2:]

    def set_data(self, data):
        self.data = data

    def filter(self):
        T = len(self.data)
        mu, sigma2 = np.zeros((2, T))
        mu_0 = self.mean_equation.get_initial_values()
        sigma_0 = self.volatility_equation.get_initial_values()
        sigma2[0] = sigma_0
        mu[0] = mu_0
        for t in range(1, T):
            mu[t] = self.mean_equation.update(self.data[t - 1], mu[t - 1])
            sigma2[t] = self.volatility_equation.update(self.data[t - 1]-mu[t-1], sigma2[t - 1])

        return mu, sigma2

    def compute_loglikelihood(self, par):
        self.convert_and_transform_parameter_array(par)
        array_mu, array_sigma2 = self.filter()
        loglik = self.distribution_module.loglik(self.distribution_parameters, self.data, array_sigma2, array_mu)
        # print(self.volatility_equation.par)
        return -loglik

    def get_initial_parameters(self):
        par = self.distribution_module.get_initial_parameters() #todo use data for initial parameters...?
        par += self.mean_equation.get_initial_parameters(self.data)
        par += self.volatility_equation.get_initial_parameters(self.data)
        return par

    def set_constraints(self, constraints):
        self.constraints = constraints

    def fit(self):
        x0 = self.get_initial_parameters()
        if len(self.constraints) > 0:
            self.optimization_method = 'SLSQP'

        result = minimize(x0=x0, fun=self.compute_loglikelihood, constraints=self.constraints, method=self.optimization_method, options={'maxiter':1000})
        self.convert_and_transform_parameter_array(result.x)
        self.estimated_parameters = {'dist':self.distribution_parameters,'mean':self.mean_equation.par,
                                     'vol':self.volatility_equation.par}
        self.fit_result = result
        self.likelihood = -result.fun


    def convert_and_transform_parameter_array(self, par):
        parameters_distribution = par[:self.n1]
        parameters_mean = par[self.n1:self.n2]
        parameters_volatility = par[self.n2:]
        self.mean_equation.update_parameters(parameters_mean)
        self.volatility_equation.update_parameters(parameters_volatility)

        if self.n1 > 0:
            self.distribution_parameters = self.distribution_module.transform_parameters(parameters_distribution)

        self.mean_equation.transform_parameters()
        self.volatility_equation.transform_parameters()

    def compute_pits(self):
        array_mu, array_sigma2 = self.filter()
        epsilon = (self.data - array_mu) / np.sqrt(array_sigma2)
        self.PITs = self.distribution_module.cdf(self.distribution_parameters, epsilon)

    def simulate_from_garch(self, u):
            epsilon = self.distribution_module.ppf(self.distribution_parameters, u)
            T = len(epsilon)
            y, mu, sigma2 = np.zeros((3, T))
            mu_0 = self.mean_equation.get_initial_values()
            sigma_0 = self.volatility_equation.get_initial_values()
            sigma2[0] = sigma_0
            mu[0] = mu_0
            y[0] = mu[0] + epsilon[0] * np.sqrt(sigma2[0])
            for t in range(1, T):
                mu[t] = self.mean_equation.update(y[t - 1], mu[t - 1])
                sigma2[t] = self.volatility_equation.update(y[t - 1]-mu[t-1], sigma2[t - 1])
                y[t] = mu[t] + epsilon[t] * np.sqrt(sigma2[t])

            self.data = y





