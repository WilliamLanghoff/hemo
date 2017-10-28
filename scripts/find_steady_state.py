import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import hemo.sims as sims
import scipy.integrate
import system
import importlib
import time
import os


def surface_area(G):
    """Total surface area of vessels in the network

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------
    sa : float
        Surface area
    """
    sa = 0.0
    for src, sink in G.edges():
        sa += G[src][sink]['radius'] * G[src][sink]['length'] * 2*np.pi
    return sa


def get_wt(G, y):
    wt = 0
    n_edges = len(G.edges())
    for src, sink in G.edges():
        wt += 30 * G[src][sink]['volume'] * y[n_edges + G[src][sink]['idx']]
    return wt

def load_network(n, k, folder=None):
    path = 'C:/Users/Bill/Documents/Python/hemo/hemo/data/'
    if folder is not None:
        path += folder
    else:
        path += 'networks100'
    path += '/G_%i_%i.gpickle'
    if not os.path.isfile(path % (n, k)):
        raise IOError
    G = nx.read_gpickle(path % (n, k))
    return G


# def compute_more_steady_states(n):
#     #radius = []
#     #steady_state = []
#     radius = np.load('radii50.npy').tolist()
#     steady_state = np.load('steady_states50.npy').tolist()
#     for k in [1,2,3,4,5,6,7,8,9,10]:
#         if not os.path.isfile('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks50/G_%i_%i.gpickle' % (n, k)):
#             continue
#         G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks50/G_%i_%i.gpickle' % (n, k))
#         sims.create_source(G)
#         importlib.reload(system)
#
#         def obj(y):
#             return system.dydt(y, 1)
#
#         y0 = np.zeros(2 * len(G.edges()))
#         if k == 1:
#             start = time.time()
#             print('Solver output for n=%i, k=%i:' % (n, k))
#             y = scipy.optimize.newton_krylov(obj, y0, verbose=True)
#             end = time.time()
#             print('Solved in %.01f seconds' % (end - start))
#         else:
#             y = scipy.optimize.newton_krylov(obj, y0, verbose=False)
#         radius.append(np.mean([10 ** 4 * G[src][sink]['radius'] for src, sink in G.edges()]))
#         steady_state.append(get_wt(G, y))
#
#     np.save('steady_states50', steady_state)
#     np.save('radii50', radius)


def compute_and_save_steady_states(n, v0=5):
    for k in [1,2,3,4,5,6,7,8,9,10]:
        if not os.path.isfile('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/G_%i_%i.gpickle' % (n, k)):
            continue
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/G_%i_%i.gpickle' % (n, k))
        sims.create_source(G)
        importlib.reload(system)

        def obj(y):
            return system.dydt(y, 1)

        y0 = np.zeros(2 * len(G.edges()))
        if k == 1:
            start = time.time()
            print('Solver output for n=%i, k=%i:' % (n, k))
            y = scipy.optimize.newton_krylov(obj, y0, verbose=True)
            end = time.time()
            print('Solved in %.01f seconds' % (end - start))
        else:
            y = scipy.optimize.newton_krylov(obj, y0, verbose=False)
        np.save('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/steady_states_v0_%.0f/%i_%i' % (n, k, v0), y)


def compute_all_steady_states(v0=5):
    for n in range(4, 8):
        compute_and_save_steady_states(n, v0=v0)


def get_log_linear_fit(radii, wt):
    log_radii = np.log(radii)
    log_wt = np.log(wt)
    m, c = np.polyfit(log_radii, log_wt, 1, w=np.sqrt(wt))
    return m, c


def plot_data():
    radii = np.load('radii50.npy').tolist()
    wt = np.load('steady_states50.npy').tolist()
    wt = [1000*w for w in wt]
    m, c = np.polyfit(radii, wt, 1)

    fig, ax = plt.subplots()
    ax.scatter(radii, wt, alpha=0.5)
    x = np.linspace(min(radii), max(radii))
    y_fit = m*x + c
    ax.plot(x, y_fit, '-b')

    x_low = np.linspace(4, min(radii))
    y_low_fit = m*x_low + c
    ax.plot(x_low, y_low_fit, '--b')

    ax.set_title('Steady States (calculated)')
    ax.set_xlabel('Radius ($\mu$m)')
    ax.set_ylabel('Drug ($\mu$mol/L)')
    ax.set_ylim([0, 1.25 * max(wt)])
    ax.set_xlim([0, 1.1*max(radii)])
    plt.show()


def load_network_and_steady_state(n, k):
    path = 'C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/G_%i_%i.gpickle' % (n, k)
    if not os.path.isfile(path):
        raise IOError
    G = nx.read_gpickle(path)

    path = 'C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/steady_states/%i_%i.npy' % (n, k)
    if not os.path.isfile(path):
        raise IOError
    steady_state = np.load(path)
    return G, steady_state

def steady_states_vs_surface_area():

    for n in [4, 5, 6]:
        for k in [1,2,3,4,5]:
            pass
    radii = np.load('radii50.npy').tolist()
    wt = np.load('steady_states50.npy').tolist()
    wt = [1000 * w for w in wt]
    m, c = np.polyfit(radii, wt, 1)

    fig, ax = plt.subplots()
    ax.scatter(radii, wt, alpha=0.5)
    x = np.linspace(min(radii), max(radii))
    y_fit = m * x + c
    ax.plot(x, y_fit, '-b')

    x_low = np.linspace(4, min(radii))
    y_low_fit = m * x_low + c
    ax.plot(x_low, y_low_fit, '--b')

    ax.set_title('Steady States (calculated)')
    ax.set_xlabel('Radius ($\mu$m)')
    ax.set_ylabel('Drug ($\mu$mol/L)')
    ax.set_ylim([0, 1.25 * max(wt)])
    ax.set_xlim([0, 1.1 * max(radii)])
    plt.show()


def surface_area_steady_state_data():
    sa, wt_list = [], []
    for n in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                G, ss = load_network_and_steady_state(n, k)
            except IOError:
                continue
            sa.append(surface_area(G))
            wt = get_wt(G, ss)
            wt_list.append(wt)
    wt_list = [1000 * w for w in wt_list]
    return sa, wt_list


def surface_area_steady_state_plot():
    sa, wt_list = surface_area_steady_state_data()

    fig, ax = plt.subplots()
    ax.scatter(sa, wt_list, alpha=0.5, edgecolor='none')
    #ax.set_ylim([0, 3])
    ax.set_xlabel('Exchange Area (cm$^2$)')
    ax.set_ylabel('Drug ($\mu$mol/L)')

    return fig, ax

def surface_area_vs_expected_surface_area_plot():
    def expected_surface_area(n):
        n_vessels = 3*(n-1)*(n)**2
        r = (n * np.sqrt(90*np.pi))**-1
        l = 1/(n+1)

        area = (2*r*np.pi) * l * n_vessels
        return area

    n_range = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    nn, sa = [], []
    for n in n_range:
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                G, ss = load_network_and_steady_state(n, k)
            except IOError:
                continue
            sa.append(surface_area(G))
            nn.append(n)
    plt.scatter(nn, sa)

    exp_sa = [expected_surface_area(n) for n in range(4, 150)]
    plt.plot(range(4, 150), exp_sa)
    plt.show()


def fit_area_steady_state_data_to_curve():
    sa, wt = surface_area_steady_state_data()

    def err(b):
        ret = 0
        for a, w in zip(sa, wt):
            ret += (w - (b[0]*a)/(b[1] + a))**2
        return ret

    b0 = np.array([3, 0])

    res = scipy.optimize.minimize(err, b0)
    return res

if __name__=='__main__':
    compute_all_steady_states(v0=4)