import numpy as np
import pandas as pd

def get_kernel(p, m0):
    # See https://cran.r-project.org/web/packages/ETAS/ETAS.pdf
    g = lambda t: (p["p"] - 1) / p["c"] * (1 + t / p["c"]) ** (- p["p"])
    sigma = lambda m: p["D"] * np.exp(p["gamma"] * (m - m0))
    f = lambda x, y, m: (p["q"] - 1) / (np.pi * sigma(m)) * (1 + (x ** 2 + y ** 2) / sigma(m)) ** (-p["q"])
    k = lambda m: p["A"] * np.exp(p["alpha"] * (m - m0))

    def kernel(t, x, y, m):
        return k(m) * f(x, y, m) * g(t)

    return kernel



def get_kernel_integral(p, m0 , t_start, t_end ):
    g_integral = lambda t: - (t + p["c"]) / ( p["c"] * ( (t + p["c"]) / p["c"] )** (p["p"]) )
    sigma = lambda m: p["D"] * np.exp(p["gamma"] * (m - m0))
    f_integral = lambda x, y, m: (p["q"] - 1) / (np.pi * sigma(m)) * (1 + (x ** 2 + y ** 2) / sigma(m)) ** (-p["q"])
    k = lambda m: p["A"] * np.exp(p["alpha"] * (m - m0))
