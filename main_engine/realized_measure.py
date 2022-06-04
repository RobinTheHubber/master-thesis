import numpy as np
from forecasting_VaR.forecasting import update_parameters


def get_realized_measure(n, daily_realized_cov, list_sigma2, idx, corr=False):
    realized_measure = {}
    for j in range(1, n):
        for i in range(1, n-j+1):
            sigma2_i = list_sigma2[i][idx[0]:idx[1]]
            sigma2_i_j = list_sigma2[i+j][idx[0]:idx[1]]
            if corr:
                realized_measure[(j, i)] = daily_realized_cov[(j, i)][idx[0]:idx[1]] / (sigma2_i * sigma2_i_j) ** 0.5
            else:
                realized_measure[(j, i)] = daily_realized_cov[(j, i)][idx[0]:idx[1]]

    return realized_measure