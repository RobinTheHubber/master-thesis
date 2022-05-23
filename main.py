import numpy as np
from vine_copula_estimation import get_vine_stuff, likelihood_vine, estimate_vine_sequentially, pack_parameters, unpack_parameters
from simulate_vine import simulate_from_vine_copula
from scipy.optimize import minimize


def main():
    copula_type = 'gaussian'
    distribution_marginal = 'gaussian'
    cpar_equation = 'difference'
    dynamic = True
    k = 1
    M = 1
    n = 5

    K = 10 * k
    hist = np.zeros((M, K))
    for m in range(M):
        print('RUN ', m)
        ##### Sample data
        list_marginal_models, par = simulate_from_vine_copula(m=m, T=1500, n=n, copula_type=copula_type, distribution_marginal=distribution_marginal, cpar_equation=cpar_equation)

        ##### Get vine stuff
        dictionary_transformation_functions, dictionary_copula_h_functions, \
        dictionary_copula_densities, dictionary_parameter_initial_values= get_vine_stuff(n=n, copula_type=copula_type, dynamic=dynamic)

        ####  Estimate the vine (sequentially)
        llik_seq, dicParameters_seq = estimate_vine_sequentially(list_marginal_models, dictionary_transformation_functions, dictionary_copula_h_functions
                                            , dictionary_copula_densities, dictionary_parameter_initial_values, cpar_equation)

        # listParameter_keys, listParameters_values_seq, dicParameters_number = unpack_parameters(dicParameters_seq, dictionary_transformation_functions)
        # npar_marginal = 10
        # list_marginal_parameters = listParameters_values_seq[:npar_marginal]

        # res = minimize(x0=listParameters_values_seq[npar_marginal:], method='BFGS', fun=likelihood_vine, args=(
        #     list_marginal_parameters,
        #     listParameter_keys, dicParameters_number, data_simulated, dictionary_transformation_functions,
        #     dictionary_copula_h_functions, lPDF_functions_marginals
        #     , lCDF_functions_marginals, dictionary_copula_densities))


        # llik_nonseq = res.fun
        # dicParameters_nonseq = pack_parameters(listParameter_keys, list(list_marginal_parameters) + list(res.x), dicParameters_number, dictionary_transformation_functions)

        # store parameter output
        hist_output = []
        for j in range(1, n):
            for i in range(1, n-j+1):
                hist_output += list(dicParameters_seq[(j, i)])

        hist[m, :] = hist_output

        print('#'*50 + '\nFinal likelihood sequential:', llik_seq, end='\n')
        # print('Final likelihood full:', llik_nonseq, end='\n')
        print('#'*50 + '\nFinal parameter set sequential:', dicParameters_seq)
        # print('Final parameter set full:', dicParameters_nonseq)

    import matplotlib.pyplot as plt
    count = 0
    for j in range(n-1, 0, -1):
        tree=n-j
        plt.figure(tree)
        fig, axs = plt.subplots(nrows=j, ncols=k)
        fig.suptitle('tree ' + str(tree))
        if j==1 and copula_type=='gaussian':
            axs.hist(hist[:, count])
            ax.axvline(par[count], color='red', lw=2)

        else:
            axs = axs.flatten()
            for ax in axs:
                ax.hist(hist[:, count])
                ax.axvline(par[count], color='red', lw=2)
                count += 1

    plt.show()

if __name__ == '__main__':
    main()
