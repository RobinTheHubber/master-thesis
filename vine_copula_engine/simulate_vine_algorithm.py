import numpy as np

from vine_copula_engine.copula_functions import *
from utility.mapping import map_logistic


#
def rho_update(par, rho_, x, v, operation='difference'):
    xi, phi_1, phi_2 = par
    rho = map_logistic(rho_, -1, 1, backwards=True)
    if operation == 'product':
        rho_next = xi + phi_1 * rho + phi_2 * x * v
    elif operation == 'difference':
        rho_next = xi + phi_1 * rho + phi_2 * abs(x - v)

    return map_logistic(rho_next, -1, 1, backwards=False)


# def rho_update(par, rho_, x, v, operation='difference'):
#     xi, phi = par
#     rho = map_logistic(rho_, -1, 1, backwards=True)
#     if operation == 'product':
#         rho_next = xi + phi * x * v
#     elif operation == 'difference':
#         rho_next = xi + phi * abs(x - v)
#
#     return map_logistic(rho_next, -1, 1, backwards=False)

def sample_from_dynamic_vine(distribution, n, T, par = None, cpar_equation=None):
    dictionary_theta = get_theta(distribution)
    dictionary_theta_all = dictionary_theta.copy()
    if distribution == 'gaussian':
        h_function = h_function_gaussian
        h_function_inv = h_function_inv_gaussian

    elif distribution == 'student_t':
        h_function = h_function_student_t
        h_function_inv = h_function_inv_student_t

    dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_theta, h_function, h_function_inv)
    mU = np.zeros((T, n))
    for t in range(T):
        vU = sample_from_vine(dictionary_h, dictionary_h_inv, dictionary_theta, n=n, T=1)
        vU = vU.reshape(-1, )
        mU[t, :] = vU

        for key in dictionary_theta.keys():
            i, j = key
            u, v = vU[i - 1], vU[i + j - 1]
            rho = dictionary_theta[(i, j)]
            rho_new = rho_update(par, rho, u, v, operation=cpar_equation)  # todo extend for student t
            dictionary_theta[(i, j)] = rho_new  # todo extend for student t
            if t < (T - 1):
                dictionary_theta_all[(i, j)] = np.append(dictionary_theta_all[(i, j)], rho_new)

    # mU2 = sample_from_vine(dictionary_h, dictionary_h_inv, dictionary_theta_all, n=n, T=1500)

    ### code below is for checking if filtering with estimation module
    ### yields same filtered series as with filtering with simulation module with same parameters
    # dictionary_theta_mimic = {}
    # for key in dictionary_theta.keys():
    #     i, j = key
    #     rho0 = dictionary_theta_all[(i,j)][0]
    #     u, v = mU[:, i - 1], mU[:, i + j - 1]
    #
    #     dictionary_theta_mimic[(i,j)] = \
    #         par_filter(par, rho0, u, v, operation=cpar_equation)
    #
    #     plt.plot(dictionary_theta_all[(i, j)][1:])
    #     plt.plot(dictionary_theta_mimic[(i, j)])
    #     plt.show()

    return mU


def sample_from_vine(dictionary_h, dictionary_h_inv, dictionary_theta, n, T):
    dictionary_v = {}
    W = np.random.random(size=(n + 1) * T).reshape((n + 1, T))
    X = np.zeros((n + 1, T))
    X[1, :] = W[1, :]
    dictionary_v[(1, 1)] = X[1, :]
    X[2, :] = dictionary_h_inv[(1, 1)](dictionary_theta[(1, 1)], W[2, :], dictionary_v[(1, 1)])
    dictionary_v[(2, 1)] = X[2, :]
    dictionary_v[(2, 2)] = dictionary_h[(1, 1)](dictionary_theta[(1, 1)], dictionary_v[(1, 1)], dictionary_v[(2, 1)])

    for i in range(3, n + 1):
        dictionary_v[(i, 1)] = W[i, :]

        for k in range(i - 1, 1, -1):
            try:
                dictionary_v[(i, 1)] = dictionary_h_inv[(k, i - k)](dictionary_theta[(k, i - k)], dictionary_v[(i, 1)],
                                                                    dictionary_v[(i - 1, 2 * k - 2)])
            except:
                print('')
        dictionary_v[(i, 1)] = dictionary_h_inv[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i, 1)],
                                                            dictionary_v[(i - 1, 1)])
        X[i, :] = dictionary_v[(i, 1)]

        if i == n:
            break

        dictionary_v[(i, 2)] = dictionary_h[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i - 1, 1)],
                                                        dictionary_v[(i, 1)])
        dictionary_v[(i, 3)] = dictionary_h[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i, 1)],
                                                        dictionary_v[(i - 1, 1)])

        if i > 3:
            for j in range(2, i - 1):
                dictionary_v[(i, 2 * j)] = dictionary_h[(j, i - j)](dictionary_theta[(j, i - j)],
                                                                    dictionary_v[(i - 1, 2 * j - 2)],
                                                                    dictionary_v[(i, 2 * j - 1)])
                dictionary_v[(i, 2 * j + 1)] = dictionary_h[j, i - j](dictionary_theta[(j, i - j)],
                                                                      dictionary_v[(i, 2 * j - 1)],
                                                                      dictionary_v[(i - 1, 2 * j - 2)])

        dictionary_v[(i, 2 * i - 2)] = dictionary_h[(i - 1, 1)](dictionary_theta[(i - 1, 1)],
                                                                dictionary_v[(i - 1, 2 * i - 4)],
                                                                dictionary_v[(i, 2 * i - 3)])

    return X[1:, :].T


def sample_from_vine3D(dictionary_h, dictionary_h_inv, dictionary_theta, n, T, N):
    dictionary_v = {}
    W = np.random.random(size=N * (n + 1) * T).reshape((N, T, n + 1))
    X = np.zeros((N, T, n + 1))
    X[:, :, 1] = W[:, :, 1]
    dictionary_v[(1, 1)] = X[:, :, 1]
    X[:, :, 2] = dictionary_h_inv[(1, 1)](dictionary_theta[(1, 1)], W[:, :, 2], dictionary_v[(1, 1)])
    dictionary_v[(2, 1)] = X[:, :, 2]
    dictionary_v[(2, 2)] = dictionary_h[(1, 1)](dictionary_theta[(1, 1)], dictionary_v[(1, 1)], dictionary_v[(2, 1)])

    for i in range(3, n + 1):
        dictionary_v[(i, 1)] = W[:, :, i]

        for k in range(i - 1, 1, -1):
            try:
                dictionary_v[(i, 1)] = dictionary_h_inv[(k, i - k)](dictionary_theta[(k, i - k)], dictionary_v[(i, 1)],
                                                                    dictionary_v[(i - 1, 2 * k - 2)])
            except:
                print('')
        dictionary_v[(i, 1)] = dictionary_h_inv[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i, 1)],
                                                            dictionary_v[(i - 1, 1)])
        X[:, :, i] = dictionary_v[(i, 1)]

        if i == n:
            break

        dictionary_v[(i, 2)] = dictionary_h[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i - 1, 1)],
                                                        dictionary_v[(i, 1)])
        dictionary_v[(i, 3)] = dictionary_h[(1, i - 1)](dictionary_theta[(1, i - 1)], dictionary_v[(i, 1)],
                                                        dictionary_v[(i - 1, 1)])

        if i > 3:
            for j in range(2, i - 1):
                dictionary_v[(i, 2 * j)] = dictionary_h[(j, i - j)](dictionary_theta[(j, i - j)],
                                                                    dictionary_v[(i - 1, 2 * j - 2)],
                                                                    dictionary_v[(i, 2 * j - 1)])
                dictionary_v[(i, 2 * j + 1)] = dictionary_h[j, i - j](dictionary_theta[(j, i - j)],
                                                                      dictionary_v[(i, 2 * j - 1)],
                                                                      dictionary_v[(i - 1, 2 * j - 2)])

        dictionary_v[(i, 2 * i - 2)] = dictionary_h[(i - 1, 1)](dictionary_theta[(i - 1, 1)],
                                                                dictionary_v[(i - 1, 2 * i - 4)],
                                                                dictionary_v[(i, 2 * i - 3)])

    return X[:, :, 1:]


def h_set_all_same(dictionary_theta, h_function, h_function_inv):
    dictionary_h, dictionary_h_inv = {}, {}
    for key, value in dictionary_theta.items():
        dictionary_h[key] = h_function
        dictionary_h_inv[key] = h_function_inv

    # for key in [(3,1), (3,2), (4,1)]:
    #     dictionary_h_inv[key] = h_function_inv_independence
    #     dictionary_h[key] = h_function_inv_independence

    return dictionary_h, dictionary_h_inv


def get_theta(distribution):
    dictionary_theta_t = {(1, 1): [0.5, 4], (1, 2): [0.75, 3], (1, 3): [0.5, 4], (1, 4): [0.25, 4], \
                          (2, 1): [0.2, 3], (2, 2): [0.10, 4], (2, 3): [-0.2, 3], \
                          (3, 1): [-0.5, 4], (3, 2): [0.5, 5], \
                          (4, 1): [0.1, 4]}

    dictionary_theta_g = dict(
        zip(dictionary_theta_t.keys(), [np.array([value[0]]) for k, value in dictionary_theta_t.items()]))
    if distribution == 'gaussian':
        dictionary_theta = dictionary_theta_g
    if distribution == 'student_t':
        dictionary_theta = dictionary_theta_t

    return dictionary_theta


def get_vine_data(distribution, n, T):
    dictionary_theta = get_theta(distribution)
    if distribution == 'gaussian':
        h_function = h_function_gaussian
        h_function_inv = h_function_inv_gaussian

    if distribution == 'student_t':
        h_function = h_function_student_t
        h_function_inv = h_function_inv_student_t

    dictionary_h, dictionary_h_inv = h_set_all_same(dictionary_theta, h_function, h_function_inv)

    mU = sample_from_vine(dictionary_h, dictionary_h_inv, dictionary_theta, n=n, T=T)
    return mU


def test_stuff():
    n = 5
    T = 10000
    mUg, mUt = get_vine_data('gaussian', n, T), get_vine_data('student_t', n, T)

    cg = [np.corrcoef(mUg[:, i], mUg[:, i + 1]) for i in range(4)]
    ct = [np.corrcoef(mUt[:, i], mUt[:, i + 1]) for i in range(4)]
    print(cg)
    print(ct)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.hist2d(mUg[:, 1], mUg[:, 2], density=True)
    ax2.hist2d(mUt[:, 1], mUt[:, 2], density=True)
    plt.legend()
    plt.show()

    T = 10000
    lvls = [0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99]
    k = 3

    def filter_exceedance(x1, x2, level, tail='right'):
        if tail == 'right':
            b1 = x1 >= level
            b2 = x2 >= level
            condition = b1 * b2

        if tail == 'left':
            b1 = x2 <= level
            b2 = x2 <= level
            condition = b1 * b2

        return x1[condition], x2[condition]

    correlations_g = [np.corrcoef(filter_exceedance(mUg[:, k], mUg[:, k + 1], lvl))[0, 1] for lvl in lvls]
    correlations_t = [np.corrcoef(filter_exceedance(mUt[:, k], mUt[:, k + 1], lvl))[0, 1] for lvl in lvls]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(lvls, correlations_g)
    ax2.plot(lvls, correlations_t)
