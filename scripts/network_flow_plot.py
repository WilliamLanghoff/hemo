import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import hemo.sims as sims
import scipy.integrate
import system
import importlib
import time
import os.path
from scipy import stats




def load_network(n, k):
    path = 'C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100'
    path += '/G_%i_%i.gpickle'
    if not os.path.isfile(path % (n, k)):
        raise IOError
    G = nx.read_gpickle(path % (n, k))
    return G

def solve_for_steady_state(G):
    start = time.time()
    sims.create_source(G)
    importlib.reload(system)

    def obj(y):
        return system.dydt(y, 1)

    y0 = np.zeros(2 * len(G.edges()))
    y = scipy.optimize.root(obj, y0, method='krylov')
    end = time.time()
    print('Solved in %.0f seconds.' % (end-start))
    return y['x']


def get_wt(G, y):
    wt = 0
    n_edges = len(G.edges())
    for src, sink in G.edges():
        wt += 30 * G[src][sink]['volume'] * y[n_edges + G[src][sink]['idx']]
    return wt


def total_flow(G):
    """Total volumetric flow through network

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------
    q: float
        Volumetric flow rate through the network in ml/min

    """
    q = 0

    for src, sink in G.edges():
        if G.node[src]['ntype'] == 'source':
            q += G[src][sink]['inverse_transit_time'] * G[src][sink]['volume']
    return 60*q


def radius_flow_data(n, folder):
    radii, flows = [], []
    for k in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        try:
            G = load_network(n, k, folder)
        except OSError:
            continue
        radii.append(10**4 * np.mean([G[src][sink]['radius'] for src, sink in G.edges()]))
        flows.append(total_flow(G))
    return radii, flows

def create_network_flow_plot(folder='networks'):

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0

    fig, ax = plt.subplots()
    all_radii, all_flows = [], []
    n_range = range(4, 18)
    color = iter(cm.rainbow(np.linspace(0, 1, len(n_range))))
    for n in n_range:
        c = next(color)
        radii, flows = radius_flow_data(n, folder)
        all_radii.extend(radii)
        all_flows.extend(flows)
        ax.scatter(radii, flows, color=c, alpha=0.5, edgecolors='none')

    log_radii = np.log(all_radii)
    log_flows = np.log(all_flows)
    m, c = np.polyfit(log_radii, log_flows, 1)
    print(m)
    print(c)
    x = np.linspace(min(all_radii), max(all_radii))
    y_fit = np.exp(m*np.log(x) + c)
    ax.plot(x, y_fit, 'b')

    x_low = np.linspace(4, min(all_radii))
    y_low_fit = np.exp(m*np.log(x_low) + c)
    ax.plot(x_low, y_low_fit, '--b')

    ax.set_yscale('log')
    ax.set_xscale('log')
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_yticks([min(all_flows), 1, max(all_flows)])

    ax.set_xticks([30, 50, 100])
    ax.minorticks_off()

    ax.set_xlabel('Mean Radius ($\mu$m)')
    ax.set_ylabel('Flow rate (ml/min)')
    ax.set_title('Network Flow Rates')


def get_log_linear_fit(folder):
    all_radii, all_flows = [], []
    n_range = range(4, 18)
    for n in n_range:
        radii, flows = radius_flow_data(n, folder)
        all_radii.extend(radii)
        all_flows.extend(flows)

    log_radii = np.log(all_radii)
    log_flows = np.log(all_flows)
    m, c = np.polyfit(log_radii, log_flows, 1)
    radii_bounds = [min(all_radii), max(all_radii)]
    return m, c, radii_bounds

def total_flow_rates_multiple_dp_plot():
    color = iter(cm.rainbow(np.linspace(0, 1, 5)))
    lines = []
    fig, ax = plt.subplots()
    for folder in ['networks100', 'networks50', 'networks20']:
        col = next(color)
        m, c, rb = get_log_linear_fit(folder)
        x = np.linspace(rb[0], rb[1])
        y_fit = np.exp(m * np.log(x) + c)
        line, = ax.plot(x, y_fit, color=col)
        lines.append(line)

        x_low = np.linspace(3, rb[0])
        y_fit_low = np.exp(m * np.log(x_low) + c)
        ax.plot(x_low, y_fit_low, '--', color=col)

    ax.set_yscale('log')
    ax.set_xscale('log')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.title('Total Flow Rate')
    plt.xlabel('Mean radius ($\mu$m)')
    plt.ylabel('Mean flow (mL/min)')

    plt.legend(lines, ['100 mmHg', '50  mmHg', '20  mmHg'])
    plt.show()
    # G = load_network(10, 3)
    # radii = [10**4 * G[src][sink]['radius'] for src, sink in G.edges()]
    # plt.hist(radii)
    # plt.show()


def plot_wt_orthogonal_to_flow_direction(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    z_coord = []
    wt = []
    n_edges = len(G.edges())
    for src, sink in G.edges():
        z_coord.append((G.node[src]['pos'][2] + G.node[sink]['pos'][2])/2)
        wt.append(y[n_edges + G[src][sink]['idx']])

    plt.scatter(z_coord, wt, alpha=0.25)
    plt.ylim([0, 1.1*max(wt)])
    plt.show()


def plot_total_wt_for_slices(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    z_coords = []
    for src, sink in G.edges():
        zc = G.node[sink]['pos'][2]
        if zc not in z_coords:
            z_coords.append(zc)
    z_coords.sort()

    n_edges = len(G.edges())
    wt = np.zeros_like(z_coords)
    for src, sink in G.edges():
        zc = G.node[src]['pos'][2]
        vol = G[src][sink]['radius']**2 * G[src][sink]['length'] * np.pi
        wt[z_coords.index(zc)] += 30 * vol * y[n_edges + G[src][sink]['idx']]

    plt.plot(z_coords, wt)
    plt.show()


def plot_avg_wt_for_slices(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    z_coords = []
    for src, sink in G.edges():
        zc = G.node[sink]['pos'][2]
        if zc not in z_coords:
            z_coords.append(zc)
    z_coords.sort()

    n_edges = len(G.edges())
    wt = np.zeros_like(z_coords)
    count = np.zeros_like(z_coords)
    for src, sink in G.edges():
        zc = G.node[src]['pos'][2]
        wt[z_coords.index(zc)] += y[n_edges + G[src][sink]['idx']]
        count[z_coords.index(zc)] += 1

    for idx in range(len(wt)):
        if count[idx] > 0:
            wt[idx] /= count[idx]

    plt.plot(z_coords, wt, '-k')
    plt.ylim([0, 1.1*max(wt)])
    plt.xlim([min(z_coords), max(z_coords)])

def plot_avg_vt_for_slices(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    z_coords = []
    for src, sink in G.edges():
        zc = G.node[src]['pos'][2]
        if zc not in z_coords:
            z_coords.append(zc)
    z_coords.sort()

    n_edges = len(G.edges())
    vt = np.zeros_like(z_coords)
    counts = np.zeros_like(z_coords)
    for src, sink in G.edges():
        zc = G.node[src]['pos'][2]
        vt[z_coords.index(zc)] += y[G[src][sink]['idx']]
        counts[z_coords.index(zc)] += 1

    for idx in range(len(vt)):
        if counts[idx] > 0:
            vt[idx] = vt[idx] / counts[idx]
    plt.plot(z_coords, vt, '-k')
    plt.ylim([0, 5.1])
    plt.xlim([min(z_coords), max(z_coords)])


def plot_avg_vt_slices_multiple(n):
    for k in [1,2,3,4]:
        plot_avg_vt_for_slices(n, k)

    plt.title('Average Drug Concentration in plasma (n=%i)' % n)
    plt.xlabel('Distance (cm)')
    plt.ylabel('$v(t)$ ($\mu$mol L$^{-1}$)')
    plt.show()


def plot_avg_wt_slices_multiple(n):
    for k in [1,2,3,4]:
        plot_avg_wt_for_slices(n, k)

    plt.title('Average Drug Concentration in tissue (n=%i)' % n)
    plt.xlabel('Distance (cm)')
    plt.ylabel('$w(t)$ ($\mu$mol L$^{-1}$)')
    plt.show()

def plot_vt_orthogonal_to_flow_direction(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    z_coord = []
    wt = []
    n_edges = len(G.edges())
    for src, sink in G.edges():
        z_coord.append((G.node[src]['pos'][2] + G.node[sink]['pos'][2])/2)
        wt.append(y[G[src][sink]['idx']])

    plt.scatter(z_coord, wt, alpha=0.25)
    plt.ylim([0, 1.1*max(wt)])
    plt.show()


def plot_avg_vt_orth_to_flow_multiple_n(nvals, folder='networks100'):
    for n in nvals:
        z_coords = []
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, 1))
        for src, sink in G.edges():
            zc = G.node[src]['pos'][2]
            if zc not in z_coords:
                z_coords.append(zc)
        z_coords.sort()

        vt_avg = np.zeros_like(z_coords)
        for k in [1,2,3,4]:
            G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
            y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

            n_edges = len(G.edges())
            vt = np.zeros_like(z_coords)
            counts = np.zeros_like(z_coords)
            for src, sink in G.edges():
                zc = G.node[src]['pos'][2]
                vt[z_coords.index(zc)] += y[G[src][sink]['idx']]
                counts[z_coords.index(zc)] += 1

            for idx in range(len(vt)):
                if counts[idx] > 0:
                    vt[idx] = vt[idx] / counts[idx]
            vt_avg += vt
        vt_avg /= 4
        plt.plot(z_coords, vt_avg)


def plot_avg_wt_orth_to_flow_multiple_n(nvals, folder='networks100'):
    for n in nvals:
        z_coords = []
        G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, 1))
        for src, sink in G.edges():
            zc = G.node[src]['pos'][2]
            if zc not in z_coords:
                z_coords.append(zc)
        z_coords.sort()

        wt_avg = np.zeros_like(z_coords)
        for k in [1,2,3,4]:
            G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
            y = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

            n_edges = len(G.edges())
            wt = np.zeros_like(z_coords)
            counts = np.zeros_like(z_coords)
            for src, sink in G.edges():
                zc = G.node[src]['pos'][2]
                wt[z_coords.index(zc)] += y[n_edges + G[src][sink]['idx']]
                counts[z_coords.index(zc)] += 1

            for idx in range(len(wt)):
                if counts[idx] > 0:
                    wt[idx] = wt[idx] / counts[idx]
            wt_avg += wt
        wt_avg /= 4
        wt_avg *= 1000
        plt.plot(z_coords, wt_avg)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05*max(wt_avg)])


def contour_plot_vt(n, k, folder='networks100'):
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/G_%i_%i.gpickle' % (folder, n, k))
    soln = np.load('C:/Users/Bill/Documents/Python/hemo/hemo/data/%s/steady_states/%i_%i.npy' % (folder, n, k))

    xvals, yvals, zvals = [], [], []
    for src, sink in G.edges():
        x = (G.node[src]['pos'][0] + G.node[sink]['pos'][0]) / 2
        y = (G.node[src]['pos'][1] + G.node[sink]['pos'][1]) / 2
        z = (G.node[src]['pos'][2] + G.node[sink]['pos'][2]) / 2
        if x not in xvals:
            xvals.append(x)
        if y not in yvals:
            yvals.append(y)
        if z not in zvals:
            zvals.append(z)

    xvals.sort()
    yvals.sort()
    zvals.sort()

    nX = len(xvals)
    nZ = len(zvals)

    xv, zv = np.meshgrid(xvals, zvals)
    vt = np.zeros([nX, nZ])
    for src, sink in G.edges():
        x = (G.node[src]['pos'][0] + G.node[sink]['pos'][0])/2
        y = (G.node[src]['pos'][1] + G.node[sink]['pos'][1])/2
        z = (G.node[src]['pos'][2] + G.node[sink]['pos'][2])/2

        if y != yvals[-1]:
            continue

        x_idx = yvals.index(x)
        z_idx = zvals.index(z)

        vt[x_idx, z_idx] = soln[G[src][sink]['idx']]

    for i in range(nX):
        for j in range(nZ):
            if vt[i, j] == 0:
                summ, count = 0, 0
                if i > 0:
                    summ += vt[i-1, j]
                    count += 1
                if i < nX - 1:
                    summ += vt[i+1, j]
                    count += 1
                if j > 0:
                    summ += vt[i, j-1]
                    count += 1
                if j < nZ - 1:
                    summ += vt[i, j+1]
                    count += 1
                vt[i, j] = summ/count

    cs = plt.contourf(xv, zv, vt, 100, cmap=plt.cm.magma, interp='bicubic')
    cbar = plt.colorbar(cs)
    plt.show()


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def transit_time_histogram(n, k):
    G = load_network(n, k)
    transit_times = [G[src][sink]['inverse_transit_time']**-1 for src, sink in G.edges()]
    transit_times = reject_outliers(np.asarray(transit_times))
    plt.hist(transit_times, bins=40)
    print(np.mean(transit_times))


if __name__ == '__main__':
    transit_time_histogram(8, 1)
    plt.title('n=14')
    plt.show()