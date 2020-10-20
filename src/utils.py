import numpy as np

def get_background_intensity_func(rates, mu_0, **kwargs):
    bins_lat = np.array(rates[1])

    bins_long = np.array(rates[0])
    values = np.array(rates[2]) * mu_0

    def mu(lat, long):
        index_lat = np.digitize(lat, bins_lat)
        index_long = np.digitize(long, bins_long)
        return values[index_long, index_lat]

    return mu, np.max(values)
