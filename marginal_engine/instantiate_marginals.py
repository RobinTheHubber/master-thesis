from marginal_model import MarginalObject


def instantiate_marginal_object(matrix_data):
    num_variables = 5
    list_distribution_epsilon = ['normal'] * num_variables
    list_volatility_equation = ['constant'] * num_variables
    list_mean_equation = ['constant'] * num_variables

    list_marginal_object = []
    for k in range(num_variables):
        marginal_object = MarginalObject(list_distribution_epsilon[k], list_volatility_equation[k],
                                         list_mean_equation[k], matrix_data[:, k])
        list_marginal_object.append(marginal_object)

    return list_marginal_object
