import numpy as np
import numpy.random as rand
from etas import get_kernel
def simulate_hom_poisson(lambda_, t, lat, long, mainshock_id=None, as_dict=False):
    n_events = rand.poisson((lat[1] - lat[0]) * (long[1] - long[0]) * (t[1] - t[0]) * lambda_)
    events_t = rand.uniform(t[0], t[1], size=(n_events,))
    events_lat = rand.uniform(lat[0], lat[1], size=(n_events,))
    events_long = rand.uniform(long[0], long[1], size=(n_events,))
    if mainshock_id is None:
        mainshock_ids = [i for i in range(events_t.shape[0])]
    else:
        mainshock_ids = [mainshock_id for i in range(events_t.shape[0])]
    events = list(zip(list(events_t), list(events_lat), list(events_long), mainshock_ids))
    if as_dict:
        events = [{"event": {"t": e[0], "lat": e[1], "long": e[2]},
                   "mainshock_id": e[3]} for e in events]
    return events


def simulate_background(lambda_, lambda_max, t, lat, long, mainshock_id=None):
    events = simulate_hom_poisson(lambda_max, t, lat, long, mainshock_id=mainshock_id)
    e_arr = np.array(events)
    density = lambda_(e_arr[:, 1], e_arr[:, 2])
    keep = rand.binomial(n=1, p=density / lambda_max)
    events = [e for i, e in enumerate(events) if keep[i]]
    events = [{"event": {"t": e[0], "lat": e[1], "long": e[2]},
               "mainshock_id": e[3]} for e in events]
    return events


def simulate_inhom_poisson(lambda_, t, lat, long, x, y, mainshock_id=None):
    # Much faster
    # Build bins
    events = []
    thresh = 1000
    bin_start = t[0]
    while 1:
        lambda_max = lambda_(bin_start, x, y)
        a = thresh / ((lat[1] - lat[0]) * (long[1] - long[0]) * lambda_max) + bin_start
        bin_end = min([a, t[1]])
        bin_events = simulate_hom_poisson(lambda_max, (bin_start, bin_end), lat, long, mainshock_id=mainshock_id)
        if len(bin_events) == 0:
            continue
        if len(bin_events) == 1:
            e_arr = np.array([bin_events[0]])
        else:
            e_arr = np.array(bin_events)
        density = lambda_(e_arr[:, 0], e_arr[:, 1], e_arr[:, 2])
        keep = rand.binomial(n=1, p=density / lambda_max)
        events += [e for i, e in enumerate(bin_events) if keep[i]]
        if a >= t[1]:
            break
        else:
            bin_start = bin_end
    events = [{"event": {"t": e[0], "lat": e[1], "long": e[2]},
               "mainshock_id": e[3]} for e in events]
    return events





def simulate_etas(t_end, lat, long, kernel_parameter, mu, mu_max, m0, beta ,mainshocks=None):
    kernel = get_kernel(kernel_parameter, m0)
    # Simulate main shocks
    if mainshocks is None:
        events = simulate_background(mu, mu_max, (0, t_end), lat, long)
    else:
        events = mainshocks
    events_simulated = events
    id_counter = 0
    while events != []:
        new_events = []
        for e in events:
            e["id"] = id_counter
            id_counter += 1
            t_event = e["event"]["t"]
            lat_event = e["event"]["lat"]
            long_event = e["event"]["long"]
            m_event = np.clip(rand.exponential(beta), 0, 2.5) + m0
            e["event"].update({"m": m_event})
            intensity = lambda t, lat_, long_: kernel(t - t_event, lat_ - lat_event, long_ - long_event, m_event)
            children = simulate_inhom_poisson(intensity, (t_event, t_end), lat, long, lat_event, long_event,
                                              mainshock_id=e["mainshock_id"])
            for c in children:
                c["father"] = e["id"]
            new_events += children
        events_simulated += new_events
        events = new_events
    return events_simulated