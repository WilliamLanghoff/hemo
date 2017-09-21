



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import hemo.sims as sims
import scipy.integrate
import system
import importlib

def get_Wt(G, times, soln, liposomes=False):
    """Compute W(t) for an entire network

    Parameters
    ----------
    G
        Graph Structure
    times : array_like
        Array of time values for simulation
    soln : array_like
        Solution - the output from odeint

    Returns
    -------
    Wt : array_like
        Array of W(t) values corresponding to the times passed to the function
    """
    n_edges = len(G.edges())
    Wt = np.zeros_like(times)
    for src, sink in G.edges():
        if liposomes:
            Wt += 30 * G[src][sink]['volume'] * soln[:, 2*n_edges + G[src][sink]['idx']]
        else:
            Wt += 30 * G[src][sink]['volume'] * soln[:, n_edges + G[src][sink]['idx']]
    return Wt


def run_sim(n, k=0, symmetric=False):
    if symmetric:
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_symm.gpickle' % n)
    else:
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' % (n,k))
    sims.create_source(G)
    importlib.reload(system)
    times = np.linspace(0, 450, 450 + 1)
    y0 = np.zeros(2 * len(G.edges()))
    soln = scipy.integrate.odeint(system.dydt, y0, times)
    #wt = get_Wt(G, times, soln)
    return times, soln


if __name__ == '__main__':
    # import time
    # #integration_time = np.load('integration_time.npy')
    # time_lengths = []
    # for n in [4,5,6,7,8,9,10,11]:
    #     k = 0
    #     start = time.time()
    #     times, soln = run_sim(n, k)
    #     #np.save('times.np', times)
    #     np.save('soln_%i_%i.np' % (n, k), soln)
    #     end = time.time()
    #     dt = end-start
    #     time_lengths.append(dt)
    #     print('%i - %i: %.01f.' % (n, k, dt))
    # np.save('time_len', time_lengths)
    #
    # plt.plot([3,4,5,6,7,8,9,10], time_lengths)
    # plt.show()

    # vessels = [3*n*((n+1)**2) for n in [3,4,5,6,7,8,9,10]]
    # time = np.load('C:/Users/Bill/Documents/Python/hemo/scripts/integration_time.npy')
    # print(time)

    mean_radii = []
    mean_steady_states = []

    all_radii, all_steady_states = [], []
    for n in [4, 5, 6, 7, 8, 9, 10]:
        rr = []
        ss = []
        for k in [1,2,3,4,5,6,7,8,9,10]:
            times = np.linspace(0, 450, 450 + 1)
            G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' %(n,k))
            soln = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/sims/soln_%i_%i.np.npy' % (n, k))
            wt = get_Wt(G, times, soln)
            ss.append(wt[-1])
            radii = [G[src][sink]['radius'] for src, sink in G.edges()]
            rr.append(np.mean(radii))
            #plt.plot(times, wt)
        mean_radii.append(np.mean(rr))
        mean_steady_states.append(np.mean(ss))
        all_radii.extend(rr)
        all_steady_states.extend(ss)
    #

    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' % (11, 0))
    soln = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/sims/soln_%i_%i.np.npy' % (11, 0))
    mean_steady_states.append(wt[-1])
    radii = [G[src][sink]['radius'] for src, sink in G.edges()]
    mean_radii.append(np.mean(radii))

    fig = plt.figure()
    ax = plt.gca()

    ax.scatter([10 ** 4 * radius for radius in all_radii], all_steady_states, alpha=0.5, edgecolors='none')
    ax.plot([10 ** 4 * radius for radius in mean_radii], mean_steady_states, c='k')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_ylim([0.00, 0.0030])
    ax.set_xlim([0, 130])

    plt.title('Steady States')
    plt.xlabel('Average Radius ($\mu$m)')
    plt.ylabel('Drug ($\mu$mol)')
    plt.show()

    # subdivs, flow = [], []
    # for n in [4, 5, 6, 7, 8, 9, 10]:
    #     for k in [1,2,3,4,5,6,7,8,9,10]:
    #         inflow = 0
    #         G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_%i.gpickle' %(n, k))
    #         subdivs.append(n-1)
    #         for src, sink in G.edges():
    #             if G.node[src]['ntype'] == 'source':
    #                 inflow += G[src][sink]['inverse_transit_time'] * (np.pi * G[src][sink]['length'] * G[src][sink]['radius']**2)
    #         flow.append(inflow)
    #
    # plt.scatter(subdivs, flow, edgecolors='none', alpha=0.5)
    # plt.title('Flow rates')
    # plt.ylabel('Flow rate (ml/sec)')
    # plt.show()

    # times = np.linspace(0, 450, 450 + 1)
    # for n in [4,5,6,7,8,9,10,11]:
    #     G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i_0.gpickle' % n)
    #     soln = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/sims/soln_%i_1.np.npy' % n)
    #     wt = get_Wt(G, times, soln)
    #     plt.plot(times, wt)
    # plt.legend([4,5,6,7,8,9,10,11])
    # plt.show()