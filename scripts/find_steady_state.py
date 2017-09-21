import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import hemo.sims as sims
import scipy.integrate
import system
import importlib



def get_wt(G, y):
    wt = 0
    n_edges = len(G.edges())
    for src, sink in G.edges():
        wt += 30 * G[src][sink]['volume'] * y[n_edges + G[src][sink]['idx']]
    return wt


# cons =({'type': 'ineq', 'fun': lambda })

radius, steady_state = [], []
for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    for n in [4,5,6,7,8,9,10]:
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' % (n, k))
        sims.create_source(G)
        importlib.reload(system)
        def obj(y):
            return system.dydt(y, 1)

        y0 = np.zeros(2 * len(G.edges()))
        y = scipy.optimize.root(obj, y0, method='krylov')
        radius.append(np.mean([10**4 * G[src][sink]['radius'] for src, sink in G.edges()]))
        steady_state.append(get_wt(G, y['x']))

radius, steady_state = [], []
for k in [0, 1,2,3]:
    n = 11
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' % (n, k))
    sims.create_source(G)
    importlib.reload(system)
    def obj(y):
        return system.dydt(y, 1)

    y0 = np.zeros(2 * len(G.edges()))
    y = scipy.optimize.root(obj, y0, method='krylov')
    radius.append(np.mean([10**4 * G[src][sink]['radius'] for src, sink in G.edges()]))
    steady_state.append(get_wt(G, y['x']))

plt.scatter(radius, steady_state)
plt.ylim([0, 0.0030])
plt.xlim([0, 130])
plt.title('Steady States (calculated)')
plt.xlabel('Radius ($\mu$m)')
plt.ylabel('Drug ($\mu$mol')
plt.show()