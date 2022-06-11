import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as m_inv
from main_engine.get_data import load_realized_cov
from main_engine.run_parameters import RunParameters

def HAR_estimation_workflow(from_, to_):

    n = RunParameters.nvar
    estimation_window = RunParameters.estimation_window

    realized_cov, list_sigma2 = load_realized_cov(n)
    dictionary_beta_hat, dictionary_hij_hat = {}, {}
    for key, hik in realized_cov.items():
        i,j = key
        k =  i + j

        dictionary_beta_hat[key], dictionary_hij_hat[key] = estimate_HAR_system(hik, estimation_window, from_, to_)


    return dictionary_beta_hat, dictionary_hij_hat

def estimate_HAR_system(hik_, estimation_window, from_, to_):
    pre_obs = 22
    hik = np.log(hik_-2*min(hik_))
    # hik = hik_
    hij_D = [hik[t - 1] for t in range(pre_obs, estimation_window)]
    hij_W = [np.mean(hik[t - 5:t]) for t in range(pre_obs, estimation_window)]
    hij_M = [np.mean(hik[t - 22:t]) for t in range(pre_obs, estimation_window)]
    I = np.ones(estimation_window-pre_obs)
    X = np.vstack([I, hij_D, hij_W, hij_M]).T
    Y = hik[pre_obs: estimation_window].reshape((-1,1))
    beta_hat = m_inv(X.T @ X) @ X.T @ Y

    T = len(hik_)
    I = np.ones(T-pre_obs+1)
    hij_D = [hik[t - 1] for t in range(pre_obs, T+1)]
    hij_W = [np.mean(hik[t - 5:t]) for t in range(pre_obs, T+1)]
    hij_M = [np.mean(hik[t - 22:t]) for t in range(pre_obs, T+1)]
    X_ = np.vstack([I, hij_D, hij_W, hij_M]).T
    Y_hat = X_ @ beta_hat
    Y_hat = np.append(np.zeros(pre_obs), Y_hat)
    return beta_hat, Y_hat[from_:to_]


