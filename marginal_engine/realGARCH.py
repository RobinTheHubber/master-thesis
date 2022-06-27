import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import kstest
from marginal_engine.distributions import skewed_t as skew_t, student_t as t

def set_initial_parameters(rt, rv):
    omega, beta, gamma, alpha, xi, phi = 0, 0.3, 0.6, 0.2, 0, 1
    b1, b2, logh1, sigma2 = -0.05, 0.05, np.log(np.var(rt)), np.var(np.log(rv))
    par = omega, beta, gamma, alpha, xi, phi, b1, b2, logh1, sigma2
    return par

def set_initial_parameters_ll(rt, rv):
    omega, beta, gamma, alpha, xi, phi = 0, 0.3, 0.6, 0.2, 0, 1
    b1, b2, logh1, sigma2 = -0.05, 0.05, np.var(rt), np.var(rv)
    par = omega, beta, gamma, alpha, xi, phi, b1, b2, logh1, sigma2
    return par

def set_initial_parameters_realEGARCH(rt, rv):
    omega, beta, gamma, alpha, xi, phi = 0, 0.3, 0.6, 0, 0, 1
    kappa, a1, a2 = 0, -0.05, 0.05
    b1, b2, logh1, sigma2 = -0.05, 0.05, np.log(np.var(rt)), np.var(np.log(rv))
    par = omega, beta, gamma, alpha, kappa, a1, a2, xi, phi, b1, b2, logh1, sigma2
    return par

def loglik(par, data, rv, filter_function):
    rt = data - np.mean(data)  # de-mean data as model input returns
    sigma2 = par[-1]
    logh, logx_hat, z = filter_function(par, rt, rv)
    ut = np.log(rv) - logx_hat
    ht = np.exp(logh)
    pdf = - 1 / 2 * (logh + rt ** 2 / ht + np.log(sigma2) + ut ** 2 / sigma2)
    loglik = sum(pdf[1:])
    return -loglik

def loglik_ll(par, data, rv, filter_function):
    rt = data - np.mean(data)  # de-mean data as model input returns
    sigma2 = par[-1]
    ht, x_hat, z = filter_function(par, rt, rv)
    ut = rv - x_hat
    logh = np.log(ht)
    pdf = - 1 / 2 * (logh + rt ** 2 / ht + np.log(sigma2) + ut ** 2 / sigma2)
    loglik = sum(pdf[1:])
    return -loglik

def filter(par, rt, rv):
    T = len(rt)
    omega, beta, gamma, alpha, xi, phi, b1, b2, logh1, sigma2 = par
    logh, logx_hat, z = np.zeros((3, T))
    logh[0] = logh1
    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(rv[t - 1]) + alpha * np.log(rt[t-1]**2)
        z[t] = rt[t] / np.sqrt(np.exp(logh[t]))
        logx_hat[t] = xi + phi * logh[t] + b1 * z[t] + b2 * (z[t] ** 2 - 1)

    return logh, logx_hat, z


def filter_ll(par, rt, rv):
    T = len(rt)
    omega, beta, gamma, alpha, xi, phi, b1, b2, h1, sigma2 = par
    h, x_hat, z = np.zeros((3, T))
    h[0] = h1
    for t in range(1, T):
        h[t] = omega + beta * h[t - 1] + gamma * rv[t - 1] + 0 * rt[t-1]**2
        z[t] = rt[t] / np.sqrt(h[t])
        x_hat[t] = xi + phi * h[t] + b1 * z[t] + b2 * (z[t] ** 2 - 1)

    return h, x_hat, z

def filter_realEGARCH(par, rt, rv):
    T = len(rt)
    omega, beta, gamma, alpha, kappa, a1, a2, xi, phi, b1, b2, logh1, sigma2 = par
    logh, logx_hat, z, u = np.zeros((4, T))
    logh[0] = logh1
    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(rv[t - 1]) + alpha * np.log(rt[t - 1] ** 2) \
                  + a1 * z[t-1] + a2 * (z[t-1]**2 -1) + kappa * u[t-1]
        z[t] = rt[t] / np.sqrt(np.exp(logh[t]))
        logx_hat[t] = xi + phi * logh[t] + b1 * z[t] + b2 * (z[t] ** 2 - 1)
        u[t] = np.log(rv[t]) - logx_hat[t]

    return logh, logx_hat, z

def filter_no_leverage(par, rt, rv):
    T = len(rt)
    omega, beta, gamma, alpha, xi, phi, b1, b2, logh1, sigma2 = par
    logh, logx_hat, z = np.zeros((3, T))
    logh[0] = logh1
    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(rv[t - 1]) + alpha * np.log(rt[t-1]**2)
        z[t] = rt[t] / np.sqrt(np.exp(logh[t]))
        logx_hat[t] = xi + phi * logh[t]

    return logh, logx_hat, z

def estimate_realEGARCH(data_, rv_):
    data, rv = data_, rv_
    # par = omega, beta, gamma, alpha, kappa, a1, a2, xi, phi, b1, b2, logh1, sigma2
    bounds = [[-10, 10],
              [0,1],
              [0,1],
              [0,1],
              [0, 1],
              [-.2,.2],
              [-.2,.2],
              [-10,10],
              [0,2],
              [-.2,.2],
              [-.2,.2],
              [-15,0],
              [10e-6,100]]

    par0 = set_initial_parameters_realEGARCH(data-np.mean(data), rv)

    filter_fun = filter_realEGARCH
    res = minimize(fun=loglik, x0=par0, args=(data, rv, filter_fun), bounds=bounds, method='SLSQP', options={'maxiter': 1000})
    par = res.x

    import matplotlib.pyplot as plt
    ht, xhat, zt_ = filter_fun(par, data-np.mean(data), rv)
    plt.plot(rv)
    plt.plot(np.exp(xhat))
    plt.plot(np.exp(ht))


    zt = zt_[1:]
    ut = np.log(rv[1:]) - xhat[1:]
    plt.scatter(zt, ut/par[-1]**0.5)
    plt.show()


    res = minimize(fun=lambda *args: -t.loglik(*args), x0=np.array([4]), args=(zt, None, None), bounds=[[2, 60]], method='L-BFGS-B')
    par_t = res.x
    res = minimize(fun=lambda *args: -skew_t.loglik(*args), x0=np.array([par_t[0], 1]), args=(zt, None, None), bounds=[[2.1, 60], [0.01,100]], method='L-BFGS-B')
    par_skew_t = res.x

    pits_normal = norm.cdf(zt)
    pits_t = t.cdf(par_t, zt)
    pits_skew_t = skew_t.cdf(par_skew_t, zt)

    plt.hist(ut, bins=45)
    plt.hist(zt, bins=45)
    plt.hist(pits_normal, bins=25)
    plt.hist(pits_t)
    plt.hist(pits_skew_t, bins=25)
    plt.show()
    print(res.message)
    print(par)
    print((kstest(pits_normal, 'uniform'),
    kstest(pits_t, 'uniform'),
    kstest(pits_skew_t, 'uniform')))

    return par

def estimate_realGARCH(data_, rv_):
    data, rv = data_, rv_
    # contraint = [{'type':'ineq', 'fun':lambda par: -(par[1]+par[2]-1)}]
    bounds = [[-10, 10],
              [0,1],
              [0,1],
              [0,1],
              [-10,10],
              [0,2],
              [-.2,.2],
              [-.2,.2],
              [-15,0],
              [10e-6,100]]


    bounds_ = [[0, 10],
              [0, 1],
              [0, 1],
              [0, 1],
              [0, 10],
              [0, 2],
              [-1, 1],
              [-1, 1],
              [0, 10],
              [1, 10000]]


    par0 = np.array([ 5.99060972e-03,  6.62542858e-01,  5.67897658e-01,  8.53294092e-02,
        5.09168667e-01,  5.37625164e-01, -6.93422831e-01,  2.90382906e-01,
        1.65514062e+00,  1.11128493e+02])

    filter_fun = filter
    res = minimize(fun=loglik, x0=par0, args=(data, rv, filter_fun), bounds=bounds, method='SLSQP', options={'maxiter': 1000})
    par = res.x
    print(res)
    print(par)

    return

def check_skewnewss_in_realEGARCH(data, rv):
    import matplotlib.pyplot as plt
    par = [ 2.49938356e-01,  6.09683982e-01,  3.89377521e-01,  1.52420133e-02,
  0.00000000e+00, -7.38739222e-02,  6.78320418e-04, -5.47017407e-01,
  9.24498870e-01, -7.51385259e-02,  2.46144406e-02, -1.54267393e-14,
  2.32760787e-01]
    ht, xhat, zt_ = filter_realEGARCH(par, data.values-np.mean(data), rv[1])
    plt.plot(rv)
    plt.plot(np.exp(xhat))
    plt.plot(np.exp(ht))


    zt = zt_[1:]
    ut = np.log(rv[1:]) - xhat[1:]
    plt.scatter(zt, ut/par[-1]**0.5)
    plt.show()


    res = minimize(fun=lambda *args: -t.loglik(*args), x0=np.array([4]), args=(zt, None, None), bounds=[[2, 60]], method='L-BFGS-B')
    par_t = res.x
    res = minimize(fun=lambda *args: -skew_t.loglik(*args), x0=np.array([par_t[0], 1]), args=(zt, None, None), bounds=[[2.1, 60], [0.01,100]], method='L-BFGS-B')
    par_skew_t = res.x

    pits_normal = norm.cdf(zt)
    pits_t = t.cdf(par_t-1, zt)
    pits_skew_t = skew_t.cdf(par_skew_t, zt)

    plt.hist(ut, bins=45)
    plt.hist(zt, bins=45)
    plt.hist(pits_normal, bins=25)
    plt.hist(pits_t)
    plt.hist(pits_skew_t, bins=25)
    plt.show()
    kstest(pits_t, 'uniform')

    return par

def get_PITs_realGARCH(data, rv_, par):
    rt = (data - np.mean(data))/100
    rv = rv_/10000
    loght, logx, zt = filter(par, rt, rv)
    pits = norm.cdf(zt)
    return pits



#
# [ 2.49938356e-01  6.09683982e-01  3.89377521e-01  1.52420133e-02
#   0.00000000e+00 -7.38739222e-02  6.78320418e-04 -5.47017407e-01
#   9.24498870e-01 -7.51385259e-02  2.46144406e-02 -1.54267393e-14
#   2.32760787e-01]

# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# [ 1.94117409e-01  6.55884955e-01  3.34503221e-01  1.54238413e-02
#   2.09100220e-16 -8.25400541e-02  2.63059175e-02 -4.49246325e-01
#   9.29356581e-01 -6.69023739e-02  8.35668694e-02  0.00000000e+00
#   2.19279936e-01]
# (KstestResult(statistic=0.03147838503876077, pvalue=0.023062554607023042), KstestResult(statistic=0.027814927946377854, pvalue=0.061220666293440584), KstestResult(statistic=0.01589991584679762, pvalue=0.6167764440593927))
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# [ 1.95150216e-01  6.77745088e-01  2.94149057e-01  1.88175822e-02
#   8.32585566e-13 -6.87674380e-02  4.82509844e-04 -5.29399616e-01
#   9.60366913e-01 -5.30155458e-02  4.90703874e-02 -4.85834753e-14
#   2.08406888e-01]
# (KstestResult(statistic=0.027823217291083086, pvalue=0.06109388885708744), KstestResult(statistic=0.012597516667941733, pvalue=0.8645076825765834), KstestResult(statistic=0.01259605148205667, pvalue=0.8645994556479262))
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# [ 2.69235170e-01  6.21492278e-01  3.29430136e-01  7.03364113e-03
#   9.77279758e-13 -3.98516011e-02 -1.70582752e-03 -7.09936823e-01
#   1.05635771e+00 -9.86694916e-03  1.98282586e-02 -3.45522769e-13
#   2.16912253e-01]
# (KstestResult(statistic=0.039020813010400834, pvalue=0.0021145531901975586), KstestResult(statistic=0.010915517874404146, pvalue=0.9495397251775003), KstestResult(statistic=0.012302113956220107, pvalue=0.8824527811305042))
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# [ 1.65583365e-01  6.64819720e-01  2.75039922e-01  5.16703208e-03
#   7.27185516e-16 -5.42011449e-02  2.44924024e-02 -4.86314541e-01
#   1.06743641e+00 -1.05442229e-01  5.62750638e-02  0.00000000e+00
#   1.86797429e-01]
# (KstestResult(statistic=0.02818831306747166, pvalue=0.055728810158938824), KstestResult(statistic=0.027701025540234347, pvalue=0.06298563239666544), KstestResult(statistic=0.037277096561009915, pvalue=0.003844442435882736))


#
# 1. st t
# 2. skew t
# 3. skew t
# 4. t
# 5. t
