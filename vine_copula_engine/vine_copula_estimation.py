from scipy.optimize import minimize

from utility.util import get_keys
from vine_copula_engine.copula_functions import *
from vine_copula_engine.dynamic_copula_functions import *

#####################################
##### Estimate Vine copula functions
#####################################

# def likelihood_marginals(mData, lTransformations_functions, lPDF_functions_marginals, lCDF_functions_marginals, dictionary_par_marginals):
#     ## optimize parameters for each marginal
#     T, n = mData.shape[0], mData.shape[1]
#     llik_marginals = 0
#     dictionary_v = {}
#     for k in range(1, n+1):
#         function_value = lPDF_functions_marginals[k-1](dictionary_par_marginals[k], mData[:, k-1])
#         llik_marginals += function_value
#         dictionary_par_marginals[k] = lTransformations_functions[k-1](dictionary_par_marginals[k])
#         dictionary_v[(0, k)] = lCDF_functions_marginals[k-1](dictionary_par_marginals[k], mData[:, k-1])
#
#     print('marginal:',
#     llik_marginals)
#     return dictionary_v, llik_marginals
#
# def likelihood_first_tree(dictionary_v, transformations_function, copula_h_function, copula_density, dictionary_theta, n):
#     llik_tree = 0
#     ## compute likelihood contribution of the current tree
#     for i in range(1, n):
#         function_value = copula_density(dictionary_theta[(1, i)], dictionary_v[(0, i)], dictionary_v[(0, i + 1)])
#         dictionary_theta[(1, i)] = transformations_function(dictionary_theta[(1, i)])
#         llik_tree += function_value
#
#     ## compute and store h-function for next tree
#     dictionary_v_prime = {}
#     dictionary_v_prime[(1,1)] = copula_h_function(dictionary_theta[(1,1)], dictionary_v[(0, 1)], dictionary_v[(0, 2)])
#     for k in range(1, n-2):
#         dictionary_v[(1, k + 1)] = copula_h_function(dictionary_theta[(1, k + 1)], dictionary_v[(0, k + 2)], dictionary_v[(0, k + 1)])
#         dictionary_v_prime[(1, k + 1)] = copula_h_function(dictionary_theta[(1, k + 1)], dictionary_v[(0, k + 1)], dictionary_v[(0, k + 2)])
#
#     dictionary_v[(1, n-1)] = copula_h_function(dictionary_theta[(1, n-1)], dictionary_v[(0, n)], dictionary_v[(0, n-1)])
#
#     print('tree 1:',
#           llik_tree)
#     return dictionary_v_prime, llik_tree
#
# def likelihood_tree(dictionary_v, dictionary_v_prime, dictionary_theta, transformations_function, copula_h_function, copula_density, j, n):
#     llik_tree = 0
#
#     ## compute likelihood contribution of the current tree
#     for i in range(1, n-j+1):
#         function_value = copula_density(dictionary_theta[(j, i)], dictionary_v_prime[(j-1, i)], dictionary_v[(j-1, i + 1)])
#         dictionary_theta[(j, i)] = transformations_function(dictionary_theta[(j, i)])
#         llik_tree += function_value
#
#     ## compute and store h-function for next tree - if not in last tree level
#     print('tree {}:'.format(j),
#           llik_tree)
#     if j == n-1:
#         return llik_tree
#
#     dictionary_v_prime[(j, 1)] = copula_h_function(dictionary_theta[(j, 1)], dictionary_v_prime[(j-1, 1)], dictionary_v[(j-1, 2)])
#
#     if n > 4:
#         for i in range(1, n - j -1):
#             dictionary_v[(j, i + 1)] = copula_h_function(dictionary_theta[(j, i + 1)], dictionary_v[(j-1, i + 2)],
#                                                          dictionary_v_prime[(j-1, i + 1)])
#             dictionary_v_prime[(j, i + 1)] = copula_h_function(dictionary_theta[(j, i + 1)], dictionary_v_prime[(j-1, i + 1)],
#                                                                dictionary_v[(j-1, i + 2)])
#
#     dictionary_v[(j, n-j)] = copula_h_function(dictionary_theta[(j, n-j)], dictionary_v[(j-1, n-j+1)], dictionary_v_prime[(j-1, n-j)])
#
#     return llik_tree
#
#
# def likelihood_vine(listParameter_values, list_marginal_parameters, listParameter_keys, dicParameters_number, mData, dictionary_transformation_functions, dictionary_copula_h_functions, lPDF_functions_marginals, lCDF_functions_marginals, dictionary_copula_densities):
#     # pack parameters into dictionary form
#     listParameter_values = list(list_marginal_parameters) + list(listParameter_values)
#     dictionary_parameter_values = pack_parameters(listParameter_keys, listParameter_values, dicParameters_number)
#
#     ## create placeholders
#     llik = 0
#     n = mData.shape[1]
#
#     ## first optimize marginal distribution and compute PIT's
#     dictionary_v, llik_marginals = likelihood_marginals(mData, dictionary_transformation_functions['marginals'], lPDF_functions_marginals, lCDF_functions_marginals, dictionary_parameter_values)
#     llik += llik_marginals
#
#     ## perform optimization for first tree separately
#     dictionary_v_prime, llik_first_tree = \
#         likelihood_first_tree(dictionary_v,
#                             dictionary_transformation_functions[1],
#                             dictionary_copula_h_functions[1],
#                             dictionary_copula_densities[1],
#                             dictionary_parameter_values,
#                             n)
#
#     llik += llik_first_tree
#
#     ## next, estimate parameters for the copula functions sequentially tree-by-tree
#     for j in range(2, n):
#         llik_tree_k = likelihood_tree(dictionary_v,
#                                     dictionary_v_prime,
#                                     dictionary_parameter_values,
#                                     dictionary_transformation_functions[j],
#                                     dictionary_copula_h_functions[j],
#                                     dictionary_copula_densities[j],
#                                     j,
#                                     n)
#         llik += llik_tree_k
#
#     print(-llik)
#     return llik


def get_filtered_rho_before_estimation(PITs, copula_h_function, dictionary_theta, n, cpar_equation, realized_measure=None):
    get_rho0 = lambda x1, x2: np.corrcoef(x1, x2)[0,1]

    dictionary_v = {}
    dictionary_v_prime = {}
    dictionary_filtered_rho = {}

    for i in range(1, n + 1):
        dictionary_v[(0, i)] = PITs[:, i - 1]

    dictionary_v_prime[(1, 1)], dictionary_filtered_rho[(1, 1)] = copula_h_function_dynamic(dictionary_theta[(1, 1)],
                                                                                            dictionary_v[(0, 1)],
                                                                                            dictionary_v[(0, 2)],
                                                                                            cpar_equation,
                                                                                            copula_h_function,
                                                                                            get_rho0(dictionary_v[(0, 1)],
                                                                                            dictionary_v[(0, 2)]),
                                                                                            output_copula_par=True,
                                                                                            realized_measure=realized_measure[(1,1)]
                                                                                            )
    for k in range(1, n - 2):
        dictionary_v[(1, k + 1)], dictionary_filtered_rho[(1, k + 1)] = copula_h_function_dynamic(
            dictionary_theta[(1, k + 1)], dictionary_v[(0, k + 2)],
            dictionary_v[(0, k + 1)], cpar_equation, copula_h_function, get_rho0(dictionary_v[(0, k + 1)], dictionary_v[(0, k + 2)]), output_copula_par=True, realized_measure=realized_measure[(1, k + 1)])
        dictionary_v_prime[(1, k + 1)], _ = copula_h_function_dynamic(dictionary_theta[(1, k + 1)],
                                                                      dictionary_v[(0, k + 1)],
                                                                      dictionary_v[(0, k + 2)],
                                                                      cpar_equation, copula_h_function,
                                                                      get_rho0(dictionary_v[(0, k + 1)],
                                                                               dictionary_v[(0, k + 2)]), output_copula_par=True,
                                                                      realized_measure=realized_measure[(1, k + 1)]
                                                                      )

    dictionary_v[(1, n - 1)], dictionary_filtered_rho[(1, n - 1)] = copula_h_function_dynamic(
        dictionary_theta[(1, n - 1)], dictionary_v[(0, n)],
        dictionary_v[(0, n - 1)], cpar_equation, copula_h_function, get_rho0(dictionary_v[(0, n)], dictionary_v[(0, n - 1)]), output_copula_par=True,
        realized_measure=realized_measure[(1, n - 1)])

    for j in range(2, n + 1):

        dictionary_v_prime[(j, 1)], dictionary_filtered_rho[(j, 1)] = copula_h_function_dynamic(
            dictionary_theta[(j, 1)], dictionary_v_prime[(j - 1, 1)],
            dictionary_v[(j - 1, 2)], cpar_equation, copula_h_function, get_rho0(dictionary_v_prime[(j - 1, 1)], dictionary_v[(j - 1, 2)]), output_copula_par=True,
            realized_measure=realized_measure[(j, 1)]
        )

        if j == n - 1:
            return dictionary_filtered_rho

        if n > 4:
            for i in range(1, n - j - 1):
                dictionary_v[(j, i + 1)], dictionary_filtered_rho[(j, i + 1)] = copula_h_function_dynamic(
                    dictionary_theta[(j, i + 1)],
                    dictionary_v[(j - 1, i + 2)],
                    dictionary_v_prime[(j - 1, i + 1)], cpar_equation,
                    copula_h_function, get_rho0(dictionary_v[(j - 1, i + 2)], dictionary_v_prime[(j - 1, i + 1)]), output_copula_par=True,
                    realized_measure=realized_measure[(j, i + 1)]
                )

                dictionary_v_prime[(j, i + 1)], _ = copula_h_function_dynamic(dictionary_theta[(j, i + 1)],
                                                                              dictionary_v_prime[(j - 1, i + 1)],
                                                                              dictionary_v[(j - 1, i + 2)],
                                                                              cpar_equation,
                                                                              copula_h_function, get_rho0(dictionary_v_prime[(j - 1, i + 1)], dictionary_v[(j - 1, i + 2)]), output_copula_par=True,
                                                                              realized_measure=realized_measure[(j, i + 1)]
                                                                              )

        dictionary_v[(j, n - j)], dictionary_filtered_rho[(j, n - j)] = copula_h_function_dynamic(
            dictionary_theta[(j, n - j)], dictionary_v[(j - 1, n - j + 1)], dictionary_v_prime[(j - 1, n - j)],
            cpar_equation, copula_h_function, get_rho0(dictionary_v[(j - 1, n - j + 1)], dictionary_v_prime[(j - 1, n - j)]), output_copula_par=True,
            realized_measure=realized_measure[(j, n - j)]
        )



def get_filtered_rho_after_estimation(PITs, copula_h_function, dictionary_theta, n, cpar_equation, rho0, realized_measure=None):
    dictionary_v = {}
    dictionary_v_prime = {}
    dictionary_filtered_rho = {}

    for i in range(1, n + 1):
        dictionary_v[(0, i)] = PITs[:, i - 1]

    dictionary_v_prime[(1, 1)], dictionary_filtered_rho[(1, 1)] = copula_h_function_dynamic(dictionary_theta[(1, 1)],
                                                                                            dictionary_v[(0, 1)],
                                                                                            dictionary_v[(0, 2)],
                                                                                            cpar_equation,
                                                                                            copula_h_function,
                                                                                            rho0[(1, 1)],
                                                                                            output_copula_par=True,
                                                                                            realized_measure=realized_measure[(1,1)]
                                                                                            )
    for k in range(1, n - 2):
        dictionary_v[(1, k + 1)], dictionary_filtered_rho[(1, k + 1)] = copula_h_function_dynamic(
            dictionary_theta[(1, k + 1)], dictionary_v[(0, k + 2)],
            dictionary_v[(0, k + 1)], cpar_equation, copula_h_function, rho0[(1, k + 1)], output_copula_par=True, realized_measure=realized_measure[(1, k + 1)])
        dictionary_v_prime[(1, k + 1)], _ = copula_h_function_dynamic(dictionary_theta[(1, k + 1)],
                                                                      dictionary_v[(0, k + 1)],
                                                                      dictionary_v[(0, k + 2)],
                                                                      cpar_equation, copula_h_function,
                                                                      rho0[(1, k + 1)], output_copula_par=True,
                                                                      realized_measure=realized_measure[(1, k + 1)]
                                                                      )

    dictionary_v[(1, n - 1)], dictionary_filtered_rho[(1, n - 1)] = copula_h_function_dynamic(
        dictionary_theta[(1, n - 1)], dictionary_v[(0, n)],
        dictionary_v[(0, n - 1)], cpar_equation, copula_h_function, rho0[(1, n - 1)], output_copula_par=True,
        realized_measure=realized_measure[(1, n - 1)])

    for j in range(2, n + 1):

        dictionary_v_prime[(j, 1)], dictionary_filtered_rho[(j, 1)] = copula_h_function_dynamic(
            dictionary_theta[(j, 1)], dictionary_v_prime[(j - 1, 1)],
            dictionary_v[(j - 1, 2)], cpar_equation, copula_h_function, rho0[(j, 1)], output_copula_par=True,
            realized_measure=realized_measure[(j, 1)]
        )

        if j == n - 1:
            return dictionary_filtered_rho

        if n > 4:
            for i in range(1, n - j - 1):
                dictionary_v[(j, i + 1)], dictionary_filtered_rho[(j, i + 1)] = copula_h_function_dynamic(
                    dictionary_theta[(j, i + 1)],
                    dictionary_v[(j - 1, i + 2)],
                    dictionary_v_prime[(j - 1, i + 1)], cpar_equation,
                    copula_h_function, rho0[(j, i + 1)], output_copula_par=True,
                    realized_measure=realized_measure[(j, i + 1)]
                )

                dictionary_v_prime[(j, i + 1)], _ = copula_h_function_dynamic(dictionary_theta[(j, i + 1)],
                                                                              dictionary_v_prime[(j - 1, i + 1)],
                                                                              dictionary_v[(j - 1, i + 2)],
                                                                              cpar_equation,
                                                                              copula_h_function, rho0[(j, i + 1)], output_copula_par=True,
                                                                              realized_measure=realized_measure[(j, i + 1)]
                                                                              )

        dictionary_v[(j, n - j)], dictionary_filtered_rho[(j, n - j)] = copula_h_function_dynamic(
            dictionary_theta[(j, n - j)], dictionary_v[(j - 1, n - j + 1)], dictionary_v_prime[(j - 1, n - j)],
            cpar_equation, copula_h_function, rho0[(j, n - j)], output_copula_par=True,
            realized_measure=realized_measure[(j, n - j)]
        )


def estimate_marginals(list_marginal_models):
    ## optimize parameters for each marginal
    n = len(list_marginal_models)
    llik_marginals = 0
    dictionary_v = {}
    dictionary_theta = {}
    for k in range(1, n + 1):
        marginal_model = list_marginal_models[k - 1]
        marginal_model.fit()
        dictionary_theta[k] = marginal_model.estimated_parameters
        llik_marginals += marginal_model.likelihood
        marginal_model.compute_pits()
        dictionary_v[(0, k)] = marginal_model.PITs
        if not marginal_model.fit_result.success:
            print('fml')

    return dictionary_v, dictionary_theta

def get_bounds(K):
    bounds = None
    if RunParameters.estimate_static_vine:
        if RunParameters.copula_type == 'gaussian':
            bounds = [[-.999999,.999999]]
        if RunParameters.copula_type == 'student_t':
            bounds = [[-.999999, .999999], [2, 60]]

    else:
        if RunParameters.evolution_type == 'simple':
            if RunParameters.skip_realized:
                bounds = [[0, np.inf], [0, 1], [0, np.inf]]
            else:
                bounds = [[0, np.inf], [0, 1], [0, np.inf]]

            if RunParameters.copula_type == 'student_t':
                bounds += [[0, np.inf], [0, 1], [-np.inf, 0]]

    return bounds

def estimate_first_tree(filtered_rho, dictionary_v, dictionary_theta, transformations_function, copula_h_function,
                        copula_density, x0_tree, n, cpar_equation, realized_measure=None):
    llik_tree = 0

    ## compute likelihood contribution of the current tree
    for i in range(1, n):
        K = int(len(x0_tree)/npar_evolution)
        # x0 = np.hstack([transformations_function(x0_tree[npar_evolution*k:npar_evolution*(k+1)], backwards=True) for k in range(K)])
        bounds = get_bounds(K)
        # x0_tree = [map_logistic(0.67322214, -1, 1, backwards=True), 0, 0, 0,
        #            map_logistic(8.09622576, 2, 60, backwards=True), 0, 0, 0]

        res = minimize(x0=x0_tree, method=RunParameters.optimization_method, bounds=bounds, options={'maxiter': 1000}, fun=copula_density_dynamic,
                       args=(dictionary_v[(0, i)], dictionary_v[(0, i + 1)], copula_density, cpar_equation, realized_measure[(1, i)]))

        par_node = transformations_function(res.x)
        dictionary_theta[(1, i)] = par_node
        filtered_rho[(1, i)] = filter_copula_parameters(par_node, dictionary_v[(0, i)],
                                                        dictionary_v[(0, i + 1)], cpar_equation, realized_measure=realized_measure[(1, i)])
        llik_tree += res.fun
        if not res.success:
            print('fml1' + str(i))

            print(par_node)
            plt.plot(filtered_rho[(1, i)][0])
            plt.figure(2)
            plt.plot(filtered_rho[(1, i)][1])
            plt.show()

    ## compute and store h-function for next tree
    dictionary_v_prime = {}
    dictionary_v_prime[(1, 1)] = copula_h_function_dynamic(dictionary_theta[(1, 1)], dictionary_v[(0, 1)],
                                                           dictionary_v[(0, 2)], cpar_equation, copula_h_function, rho0=filtered_rho[(1,1)][:, 0], realized_measure=realized_measure[(1, 1)])
    for k in range(1, n - 2):
        dictionary_v[(1, k + 1)] = copula_h_function_dynamic(dictionary_theta[(1, k + 1)], dictionary_v[(0, k + 2)],
                                                             dictionary_v[(0, k + 1)], cpar_equation, copula_h_function, rho0=filtered_rho[(1,k+1)][:, 0], realized_measure=realized_measure[(1, k + 1)])
        dictionary_v_prime[(1, k + 1)] = copula_h_function_dynamic(dictionary_theta[(1, k + 1)],
                                                                   dictionary_v[(0, k + 1)], dictionary_v[(0, k + 2)],
                                                                   cpar_equation, copula_h_function, rho0=filtered_rho[(1,k+1)][:, 0], realized_measure=realized_measure[(1, k + 1)])

    dictionary_v[(1, n - 1)] = copula_h_function_dynamic(dictionary_theta[(1, n - 1)], dictionary_v[(0, n)],
                                                         dictionary_v[(0, n - 1)], cpar_equation, copula_h_function, rho0=filtered_rho[(1,n-1)][:, 0], realized_measure=realized_measure[(1, n - 1)])
    return dictionary_v_prime, dictionary_theta, llik_tree


def estimate_tree(filtered_rho, dictionary_v, dictionary_v_prime, dictionary_theta, transformations_function,
                  copula_h_function, copula_density, x0_tree, j, n, cpar_equation, realized_measure=None):
    llik_tree = 0

    ## compute likelihood contribution of the current tree
    for i in range(1, n - j + 1):
        # x0 = transformations_function(x0_tree, backwards=True)
        x0 = x0_tree
        K = int(len(x0)/npar_evolution)
        bounds = get_bounds(K)
        res = minimize(x0=x0, method=RunParameters.optimization_method, bounds=bounds, options={'maxiter': 1000}, fun=copula_density_dynamic, args=(
        dictionary_v_prime[(j - 1, i)], dictionary_v[(j - 1, i + 1)], copula_density, cpar_equation, realized_measure[(j, i)]))
        par_node = transformations_function(res.x)
        filtered_rho[(j, i)] = filter_copula_parameters(par_node, dictionary_v_prime[(j - 1, i)], dictionary_v[(j - 1, i + 1)], cpar_equation, realized_measure=realized_measure[(j, i)])
        dictionary_theta[(j, i)] = par_node
        llik_tree += res.fun
        if not res.success:
            print((res.success, par_node))
            plt.plot(filtered_rho[(j, i)][0])
            print('fml' + str(j) + str(i))

        # print(res.message)
        # print(par_node, end='\n\n')

    ## compute and store h-function for next tree - if not in last tree level
    if j == n - 1:
        return llik_tree

    dictionary_v_prime[(j, 1)] = copula_h_function_dynamic(dictionary_theta[(j, 1)], dictionary_v_prime[(j - 1, 1)],
                                                           dictionary_v[(j - 1, 2)], cpar_equation, copula_h_function, rho0=filtered_rho[(j,1)][:, 0], realized_measure=realized_measure[(j, 1)])

    if n > 4:
        for i in range(1, n - j - 1):
            dictionary_v[(j, i + 1)] = copula_h_function_dynamic(dictionary_theta[(j, i + 1)],
                                                                 dictionary_v[(j - 1, i + 2)],
                                                                 dictionary_v_prime[(j - 1, i + 1)], cpar_equation,
                                                                 copula_h_function, rho0=filtered_rho[(j,i+1)][:, 0], realized_measure=realized_measure[(j, i + 1)])
            dictionary_v_prime[(j, i + 1)] = copula_h_function_dynamic(dictionary_theta[(j, i + 1)],
                                                                       dictionary_v_prime[(j - 1, i + 1)],
                                                                       dictionary_v[(j - 1, i + 2)], cpar_equation,
                                                                       copula_h_function, rho0=filtered_rho[(j,i+1)][:, 0], realized_measure=realized_measure[(j, i + 1)])

    dictionary_v[(j, n - j)] = copula_h_function_dynamic(dictionary_theta[(j, n - j)], dictionary_v[(j - 1, n - j + 1)],
                                                         dictionary_v_prime[(j - 1, n - j)], cpar_equation,
                                                         copula_h_function, rho0=filtered_rho[(j,n-j)][:, 0], realized_measure=realized_measure[(j, n- j)])
    return llik_tree


def estimate_vine_sequentially(dictionary_v, dictionary_theta, dictionary_transformation_functions,
                               dictionary_copula_h_functions, dictionary_copula_densities,
                               dictionary_parameter_initial_values, n, cpar_equation=None, realized_measure=None):
    ## create placeholders
    filtered_rho = {}

    ## perform optimization for first tree separately
    dictionary_v_prime, dictionary_theta, llik_first_tree = \
        estimate_first_tree(filtered_rho, dictionary_v,
                            dictionary_theta,
                            dictionary_transformation_functions[1],
                            dictionary_copula_h_functions[1],
                            dictionary_copula_densities[1],
                            dictionary_parameter_initial_values[1],
                            n, cpar_equation, realized_measure)

    for j in range(2, n):
        if j == 3:
            print('')
        print(j)
        estimate_tree(filtered_rho, dictionary_v,
                      dictionary_v_prime,
                      dictionary_theta,
                      dictionary_transformation_functions[j],
                      dictionary_copula_h_functions[j],
                      dictionary_copula_densities[j],
                      dictionary_parameter_initial_values[j],
                      j,
                      n, cpar_equation, realized_measure)

    for key, value in dictionary_theta.items():
        print(key, ' : ', value)

    return dictionary_theta, filtered_rho

#
# def unpack_parameters(dicParameters, dictionary_transformation_functions=None):
#     dicParameters_number = {}
#     listParameters_values = np.zeros(sum([len(value) for k, value in dicParameters.items()]))
#     counter = 0
#     i = 0
#     for key, par in dicParameters.items():
#         if dictionary_transformation_functions is None:
#             listParameters_values[counter:counter + len(par)] = par
#         else:
#             if type(key) != tuple:
#                 transform = dictionary_transformation_functions['marginals'][key-1]
#             else:
#                 transform = dictionary_transformation_functions[key[0]]
#
#             listParameters_values[counter:counter + len(par)] = transform(par, backwards=True)
#
#         dicParameters_number[key] = (counter, counter + len(par))
#         counter += len(par)
#
#     listParameter_keys = list(dicParameters.keys())
#     return listParameter_keys, listParameters_values, dicParameters_number
#
# def pack_parameters(listParameter_keys, listParameters_values, dicParameter_number, dictionary_transformation_functions=None):
#     dictionary_parameters = {}
#     for key in listParameter_keys:
#         k1, k2 = dicParameter_number[key]
#         if dictionary_transformation_functions is not None:
#             if type(key) != tuple:
#                 transform = dictionary_transformation_functions['marginals'][key - 1]
#             else:
#                 transform = dictionary_transformation_functions[key[0]]
#
#             dictionary_parameters[key] = transform(listParameters_values[k1:k2])
#         else:
#             dictionary_parameters[key] = listParameters_values[k1:k2]
#
#     return dictionary_parameters
#

def get_vine_stuff(n, copula_type, dynamic=False):
    ## create tree: specify the conditioned variables in each node for every tree
    x0_copula_gaussian = [0]
    x0_copula_student_t = [0, 6]
    x0_copula_gaussian_dynamic = np.array([0.1]*npar_evolution)
    x0_copula_student_t_dynamic = np.array([0.1]*2*npar_evolution)

    if copula_type == 'gaussian':
        copula_density = copula_density_gaussian
        h_function = h_function_gaussian
        transformation_copula = transformation_gaussian_copula
        if RunParameters.estimate_static_vine:
            x0_copula = x0_copula_gaussian
        else:
            x0_copula = x0_copula_gaussian_dynamic

    if copula_type == 'student_t':
        copula_density = copula_density_student_t
        h_function = h_function_student_t
        transformation_copula = transformation_student_t_copula
        if RunParameters.estimate_static_vine:
            x0_copula = x0_copula_student_t
        else:
            x0_copula = x0_copula_student_t_dynamic

    if dynamic:
        transformation_copula = transformation_dynamic_equation

    ## specify pdf and cdf functions for the copula's per tree level
    dictionary_copula_densities = {1: copula_density, 2: copula_density, 3: copula_density, 4: copula_density,
                                   5: copula_density}
    dictionary_copula_h_functions = {1: h_function, 2: h_function, 3: h_function, 4: h_function, 5: h_function}

    ## specify the parameter transformations
    dictionary_transformation_functions = {1: transformation_copula, 2: transformation_copula, 3: transformation_copula,
                                           4: transformation_copula, 5: transformation_copula}

    ## specify the parameter initial values for the optimization
    dictionary_parameter_initial_values = {1: x0_copula, 2: x0_copula, 3: x0_copula, 4: x0_copula, 5: x0_copula}
    return dictionary_transformation_functions, dictionary_copula_h_functions, dictionary_copula_densities, dictionary_parameter_initial_values

#
# def get_vine_x0_mm():
#     ncopulas = len(get_keys())
#
#     if evolution_type == 'simple':
#         x0_copula_gaussian_dynamic = np.array([0, 0, 0, 0]*ncopulas)
#         x0_copula_student_t_dynamic = np.array([0, 0, 0, 0, 0, 0, 0, 0]*ncopulas)
#     elif evolution_type == 'HAR':
#         x0_copula_gaussian_dynamic = np.array([0.01, 0.5, 0.25 ,0.25]*ncopulas)
#         x0_copula_student_t_dynamic = np.array([0, 0, 0, 0, 0, 0, 0, 0]*ncopulas)
#     elif evolution_type == 'simple_ar':
#         x0_copula_gaussian_dynamic = np.array([0, 0, 0, 0, 0]*ncopulas)
#         x0_copula_student_t_dynamic = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]*ncopulas)
#
#     if RunParameters.distribution == 'gaussian':
#         x0_copula = x0_copula_gaussian_dynamic
#     elif RunParameters.distribution == 'student_t':
#         x0_copula = x0_copula_student_t_dynamic
#
#     return x0_copula
