from main_engine.run_parameters import RunParameters

def get_keys():
    n = RunParameters.nvar
    keys = []
    for j in range(1, n):
        for i in range(1, n-j+1):
            keys.append((j, i))

    return keys
