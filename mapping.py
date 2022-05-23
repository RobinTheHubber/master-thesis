import numpy as np

def map_positive(x, backwards=False):
    if backwards:
        return np.log(x)
    else:
        return np.exp(x)

def map_logistic(x, a, b, backwards):
    if backwards:
        y = (x-a) / (b-a)
        return np.log((y/(1-y)))
    else:
        return (b-a) / (1+np.exp(-x)) + a
