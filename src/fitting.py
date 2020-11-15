# R Functions
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from datetime import datetime
import numpy as np
import pandas as pd
def install_etas():
    utils = importr("utils")
    utils.install_packages("ETAS")

def to_catalogue(events):
    events = sorted(events, key=lambda x: x["event"]["t"])
    date_time = [datetime.fromtimestamp(e["event"]["t"] * 3600 * 24) for e in events]
    date = [d.date().strftime("%Y-%m-%d") for d in date_time]
    time = [d.time().strftime("%H:%M:%S") for d in date_time]
    long = [e["event"]["long"] for e in events]
    lat = [e["event"]["lat"] for e in events]
    mag = [e["event"]["m"] for e in events]
    return pd.DataFrame(list(zip(date, time, lat, long, mag)), columns=["date", "time", "lat", "long", "mag"])


def fit_ETAS(cat, begin, start, end, lat, long, m0, n_iter=3, **kwargs):
    importr("ETAS")
    if not type(cat) is str:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(cat)
            robjects.globalenv["data"] = r_from_pd_df
            cat = "data"
    robjects.r(f'''
    cat <- catalog({cat}, time.begin="{begin}",
    study.start="{start}", study.end="{end}",
    lat.range=c{lat}, long.range=c{long}, mag.threshold={m0})
    param0 <- c(0.46, 0.23, 0.022, 2.8, 1.12, 0.012, 2.4, 0.35)
    # fitting the model
    ## Not run:
    fit <- etas(cat, param0=param0,no.itr={n_iter}, verbose=FALSE, nthreads=1)
        ''')

    density = robjects.r("rates(fit)")
    params = dict(zip(robjects.r('''names(fit$param)'''), np.array(robjects.r('''fit$param'''))))
    mags = np.array(robjects.r("cat$revents[,4]"))
    beta = np.mean(mags)
    return density, params, beta



