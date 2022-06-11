# todo 0: compare dynamic gaussian vine copula garch VaR forecasts with simple GARCH parametric forecasts
# (active) todo 1: expand with dynamic t copula
# todo 2: expand with the realized volatility measures for the marginals and realized covariance measures for the evolution equations
# todo 2 to 3: probably discuss results somewhere around this point
# todo 3: try out a couple of different garch marginal or even a totally different marginal

from forecasting_VaR.forecasting import *
from forecasting_VaR.forecasting_HAR import HAR_estimation_workflow
from main_engine.realized_measure import get_realized_measure
from vine_copula_engine.vine_copula_estimation import get_vine_stuff, estimate_vine_sequentially


def estimate_model(marginal_models_list, n, copula_type, cpar_equation, list_sigma2, training_idx, dictionary_theta=None, daily_realized_cov=None):
    dynamic = True
    dictionary_transformation_functions, dictionary_copula_h_functions, \
    dictionary_copula_densities, dictionary_parameter_initial_values = get_vine_stuff(n=n, copula_type=copula_type,
                                                                                      dynamic=dynamic)

    #### Estimate marginal models
    if dictionary_theta is None:
        dictionary_v, dictionary_theta = estimate_marginals(marginal_models_list)
    else:
        dictionary_v = {}
        for i, marginal_model in enumerate(marginal_models_list):
            dictionary_v[(0, i + 1)] = get_PITs_with_estimated_parameters(marginal_model.data, dictionary_theta[i + 1],
                                                                          marginal_model)

    ### retrieve realized measures
    if daily_realized_cov is not None:
        realized_measure = get_realized_measure(n, daily_realized_cov, list_sigma2, (0, training_idx))

    #### Estimate the vine (sequentially)
    if RunParameters.skip_realized:
        keys = get_keys()
        realized_measure = dict(zip(keys, [None] * len(keys)))
    else:
        _, h_ij_hat_train = HAR_estimation_workflow(from_=0, to_=training_idx+1)
        realized_measure = h_ij_hat_train

    dicParameters_seq, filtered_rho = estimate_vine_sequentially(dictionary_v, dictionary_theta,
                                                                 dictionary_transformation_functions,
                                                                 dictionary_copula_h_functions,
                                                                 dictionary_copula_densities,
                                                                 dictionary_parameter_initial_values, n, cpar_equation, realized_measure)
    return dicParameters_seq, filtered_rho


