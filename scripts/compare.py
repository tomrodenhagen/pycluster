import pickle, os

from etas_simulation.declustering import *
from etas_simulation.metrics import *
from etas_simulation.fitting import *
from etas_simulation.utils import *
from etas_simulation.simulation import *

import numpy as np



#install_etas()
configurations = [{ "cat": "iran.quakes",
                    "lat": (26, 40),
                    "long": (44, 63),
                    "days": 365 * 30,
                    "start":"1986/01/01",
                    "begin":"1973/01/01",
                    "end":"2016/01/01",
                    "m0": 5}]

methods = [{"name": "Window_method", "function" : window_method},
           {"name":"NN", "function":NearestNeigbour},
           {"name": "ETAS", "function": from_ETAS}]
metrics = {"mae":mae,
           "homogenity_score":mean_hom_comp }
n_catalogs = 1
res = {method["name"] : {name: [] for name in metrics.keys()} for method in methods}

for conf in configurations:
  catalog = conf["cat"]
  if os.path.exists(f"./{catalog}_estimate.pickle") :
    density, params, background = pickle.load(open(f"./{catalog}_estimate.pickle", "rb") )
  else:
    density, params, background = fit_ETAS(**conf)
    pickle.dump( (density, params, background),  open(f"./{catalog}_estimate.pickle", "wb"))
  mu, mu_max = get_background_intensity_func(density, mu_0 = params["mu"], **conf)
  #conf.update({"fitted" :  (density, params, background) })
  for _ in range(n_catalogs):
    simulated_events = simulate_etas(conf["days"], conf["lat"], conf["long"], params, mu, mu_max, conf["m0"], )
    simulated_events = sorted(simulated_events, key= lambda x: x["event"]["t"])
    ids = [e["mainshock_id"] for e in simulated_events]

    for m in methods:
      cluster = m["function"](simulated_events, **conf )
      for name, metric in metrics.items():
        res[m["name"]][name].append( metric(ids, cluster) )

res_agg = {method["name"] : {name: None for name in metrics.keys()} for method in methods}

for m in methods:
  for name, metric in metrics.items():
    res_agg[m["name"]][name] = np.mean(res[m["name"]][name])

print(res_agg)