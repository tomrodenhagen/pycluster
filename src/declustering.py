# Methods
import numpy as np

from src.simulation import get_kernel
from src.utils import *
from src.fitting import *
from datetime import datetime
from rpy2 import robjects
from rpy2.robjects.packages import importr
from sklearn.mixture import GaussianMixture as gm
from scipy.sparse.csgraph import connected_components


def T_Window_1(m):
    if m < 6.5:
        return 10 ** (m * 0.54 - 0.547)
    else:
        return 10 ** (m * 0.032 + 2.73)


S_Window_1 = lambda m: 10 ** (m * 0.1238 + 0.983) / 111


def window_method(events, T_Window=T_Window_1, S_Window=S_Window_1, **kwargs):
    cluster_mat = np.zeros((len(events), len(events)))
    for l, e in enumerate(events):
        for i, k in enumerate(events):
            t_dist = np.abs(k["event"]["t"] - e["event"]["t"])
            s_dist = ((e["event"]["lat"] - k["event"]["lat"]) ** 2 + (e["event"]["long"] - k["event"]["long"]) ** 2 ) ** 0.5
            if t_dist < T_Window(k["event"]["m"]) and s_dist < S_Window(k["event"]["m"]):
                cluster_mat[l, i] = 1
    return connected_components(cluster_mat)[1]


def NearestNeigbour(events, **kwargs):
    d = 1.6
    thresh = 0.0001
    q = 0.5
    cluster_mat = np.zeros((len(events), len(events)))

    distances = []
    for l, e in enumerate(events):
        for i, k in enumerate(events):
            if e["event"]["t"] < k["event"]["t"] or l == i:
                cluster_mat[l, i] = 10000000
            else:
                t_delta = (e["event"]["t"] - k["event"]["t"])
                r_delta = ((e["event"]["lat"] - k["event"]["lat"]) ** 2 + (
                        e["event"]["long"] - k["event"]["long"]) ** 2) ** 0.5
                distances.append(
                    [t_delta * 10 ** (- k["event"]["m"] * q) * r_delta ** d * 10 ** (- k["event"]["m"] * (1 - q))])

                cluster_mat[l, i] = t_delta * r_delta ** d * 10 ** (- k["event"]["m"])
    nnd = np.min(cluster_mat, axis=1)
    gm_ = gm(2)
    gm_.fit(np.log10(nnd.reshape(-1, 1)))
    cluster = gm_.predict(np.log10(nnd.reshape(-1, 1)))
    parents = np.argmin(cluster_mat, axis=1)
    cluster_mat = np.zeros((len(events), len(events)))
    for i, (cluster, p) in enumerate(zip(cluster, parents)):
        if cluster == 0:
            cluster_mat[i, p] = 1
            cluster_mat[p, i] = 1


    return connected_components(cluster_mat)[1]

def from_ETAS(events, days, m0, lat, long, fitted=None, **kwargs):
    importr("ETAS")
    events = sorted(events, key=lambda x: x["event"]["t"])

    if fitted is None:
        dt = to_catalogue(events)
        start = datetime.fromtimestamp(0).date().strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(days * 3600 * 24).date().strftime("%Y-%m-%d")
        density, params, background = fit_ETAS(dt, begin=start, start=start, end=end, lat=lat, long=long, m0=m0)
    else:
        density, params, background = fitted
    kernel = get_kernel(params, m0)
    mu, _ = get_background_intensity_func(density,  mu_0=params["mu"])
    events = sorted(events, key=lambda x: x["event"]["t"])
    trigger_prob = np.zeros((len(events), len(events)))
    for l, e in enumerate(events):
        for i, k in enumerate(events):
            if e["event"]["t"] > k["event"]["t"]:
                t_delta = e["event"]["t"] - k["event"]["t"]
                lat_delta = e["event"]["lat"] - k["event"]["lat"]
                long_delta = e["event"]["long"] - k["event"]["long"]
                trigger_prob[i, l] = kernel(t_delta, lat_delta, long_delta, k["event"]["m"])

    back_prob = np.array([mu(e["event"]["lat"], e["event"]["long"]) for e in events]).reshape(-1, 1)
    #x = np.array(robjects.r("probs(fit)"))[2]

    conc_probs = np.concatenate([trigger_prob, back_prob.reshape(1, -1)]) # For numerical stability

    conc_probs_normed = conc_probs / np.sum(conc_probs, axis=0)
    parents = []
    for i in range(conc_probs_normed.shape[1]):
        parents.append(np.argmax(np.random.multinomial(1, conc_probs_normed[:, i], size=1)))

    parent_mat = np.zeros((len(events), len(events)))

    for i, p in enumerate(parents):
        if p != len(parents):
            parent_mat[i, p] = 1
    return connected_components(parent_mat)[1]