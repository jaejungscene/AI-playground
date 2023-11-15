import numpy as np


def calculate_entropy(probs, base='e', eps=np.finfo(float).eps):
    r"""
    entropy == 정보량(information quentity)의 기대값
    Args:
        probs (list | tuple | ndarray): list of the proability values corresponding to each random variable
        base (str, int, float): base of log function -> log_2, log_10, log_e where 2, 10 and e is base
                         only Euler number 'e' can be string, the other must be int or float
        eps (float): used to prevent a situation in which logarithmic zero diverges infinitely.

    Example:
        X~Uniform(1,3)
        probs = [1/3, 1/3, 1/3]
        Entropy = E[log_(base)(X)]
    """
    probs = np.array(probs) if isinstance(probs, (list,tuple)) else probs
    base = np.exp(1) if isinstance(base, str) else base
    
    information = -(np.log(probs+eps)/np.log(base)) 
    entropy = (probs*information).sum()
    return entropy


ex1 = [0.5, 0.5]
print(calculate_entropy(ex1, base=2))

ex2 = (0.8, 0.2)
print(calculate_entropy(ex2, base=2))

ex3 = np.array([1.0, 0.0])
print(calculate_entropy(ex3, base=2))

ex4 = [1/2, 1/4, 1/8, 1/8]
print(calculate_entropy(ex4, base=2))