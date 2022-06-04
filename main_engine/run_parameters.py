import numpy as np

class RunParameters():
    evolution_type = 'simple'
    copula_type = 'gaussian'
    cpar_equation = 'RC_' + evolution_type
    run_model = True
    run_var = run_model
    skip_realized = False
    train_idx = 1500
    N = 10000
    weights = np.array([.2, .2, .2, .2, .2]).reshape((-1, 1))

    @staticmethod
    def get_run_parameters():
        return RunParameters.evolution_type, RunParameters.copula_type, RunParameters.cpar_equation, RunParameters.run_model, RunParameters.run_var, RunParameters.skip_realized, RunParameters.train_idx, RunParameters.N, RunParameters.weights