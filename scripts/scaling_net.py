import hemo.net_prep as net_prep
import hemo.sims as sims
import scipy.integrate
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import system
import importlib
import time


viscosity = 3.5e-3


def total_resistance(G):
    dp = 25 * 133.322387415
    q = total_flow(G)
    return dp / q


def total_flow(G):
    """Total volumetric flow through network

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------
    q: float
        Volumetric flow rate through the network

    """
    q = 0

    for src, sink in G.edges():
        if src == G.graph['source']:
            q += G[src][sink]['inverse_transit_time'] * G[src][sink]['volume']
    return q


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
        sa += G[src][sink]['radius'] * G[src][sink]['length']
    return sa


def get_total_volume(G):
    """Calculates total vasculature volume

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------
    float
        Total vasculature volume

    """
    total_volume = 0.0
    for src, sink in G.edges():
        total_volume += (G[src][sink]['radius'] ** 2) * G[src][sink]['length'] * np.pi
    return total_volume


def create_network_multiple_sources_and_sinks(N, *, dp=50, radii_iters=2, symmetric=False):
    delta = 1 / (N + 1)

    G = nx.DiGraph()
    G.graph['N'] = N
    G.graph['delta'] = delta

    def node_setup(G):
        # First add the nodes of the graph
        vd = {}
        counter = 0
        for x_idx in range(N):
            for y_idx in range(N):
                for z_idx in range(N):
                    vd[(x_idx, y_idx, z_idx)] = counter
                    G.add_node(counter, pos=[delta * (x_idx + 1), delta * (y_idx + 1), delta * (z_idx + 1)],
                               ntype='internal')
                    counter += 1
        return vd

    vd = node_setup(G)

    # Next add the edges
    def edge_setup(G):
        for x_idx in range(N):
            for y_idx in range(N):
                for z_idx in range(N):
                    el = []
                    if x_idx < N - 1:
                        el.append((vd[(x_idx, y_idx, z_idx)], vd[(x_idx + 1, y_idx, z_idx)]))
                    if y_idx < N - 1:
                        el.append((vd[(x_idx, y_idx, z_idx)], vd[(x_idx, y_idx + 1, z_idx)]))
                    if z_idx < N - 1:
                        el.append((vd[(x_idx, y_idx, z_idx)], vd[(x_idx, y_idx, z_idx + 1)]))
                    G.add_edges_from(el)
    edge_setup(G)

    # Sources and sinks are alternating nodes on opposite faces
    def src_sink_setup(G):
        for x_idx in range(N):
            for y_idx in range(N):
                if y_idx % 2 != x_idx % 2:
                    continue
                G.node[vd[x_idx, y_idx, 0]]['ntype'] = 'source'
                G.node[vd[x_idx, y_idx, N-1]]['ntype'] = 'sink'

        # Ensure edges direct out from sources, in to sinks
        for src, sink in G.edges():
            if G.node[sink]['ntype'] == 'source' or G.node[src]['ntype'] == 'sink':
                G.remove_edge(src, sink)
                G.add_edge(sink, src)
    src_sink_setup(G)

    def node_dist_setup(G):
        # Assign distances from source/sink nodes
        for z_idx in range(N):
            for y_idx in range(N):
                for x_idx in range(N):
                    if y_idx % 2 == x_idx % 2:
                        G.node[vd[x_idx, y_idx, z_idx]]['src_dist'] = z_idx
                        G.node[vd[x_idx, y_idx, z_idx]]['sink_dist'] = (N-1) - z_idx
                    else:
                        G.node[vd[x_idx, y_idx, z_idx]]['src_dist'] = z_idx + 1
                        G.node[vd[x_idx, y_idx, z_idx]]['sink_dist'] = N - z_idx - 1
    node_dist_setup(G)
    # assign lengths
    net_prep.calculate_lengths(G)

    if not symmetric:
        assign_distances_from_source_and_sink_nodes(G)

        alpha = 5
        beta = 1/(5 * (n + 1) * np.sqrt(90 * np.pi))


        for src, sink in G.edges():
            G[src][sink]['radius'] = np.random.gamma(alpha, beta)

        for _ in range(radii_iters):
            for src1, sink1 in G.edges():
                for src2, sink2 in G.edges():
                    if src1 == src2:
                        continue
                    if G[src1][sink1]['center_dist'] > G[src2][sink2]['center_dist'] and G[src1][sink1]['radius'] < G[src2][sink2]['radius']:
                        temp_radius = G[src1][sink1]['radius']
                        G[src1][sink1]['radius'] = G[src2][sink2]['radius']
                        G[src2][sink2]['radius'] = temp_radius

    elif symmetric:
        for src, sink in G.edges():
            G[src][sink]['radius'] = 1/((N+1) * np.sqrt(90*np.pi))

    net_prep.prep_net_for_sims(G, dp)

    return G


def assign_distances_from_source_and_sink_nodes(G):
    """For each edge, assigns a graph distance from closest source and sink

    Used in the example network where alternating nodes on two opposite faces are sources/sinks.
    Needed for sorting radii such that nodes nearest the center are smaller in radius than closest to src/sink

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """

    for src, sink in G.edges():
        G[src][sink]['src_dist'] = G.node[src]['src_dist']
        G[src][sink]['sink_dist'] = G.node[sink]['sink_dist']
        G[src][sink]['center_dist'] = abs(G[src][sink]['src_dist'] - G[src][sink]['sink_dist'])
    #
    # source_nodes, sink_nodes = [], []
    # for node in G.nodes():
    #     if G.node[node]['ntype'] == 'source':
    #         source_nodes.append(node)
    #     elif G.node[node]['ntype'] == 'sink':
    #         sink_nodes.append(node)
    #
    # for src, sink in G.edges():
    #     src_dists, sink_dists = [], []
    #     for entrance_node in source_nodes:
    #         try:
    #             src_dists.append(nx.shortest_path_length(G, source=entrance_node, target=src))
    #         except nx.NetworkXNoPath:
    #             pass
    #     for exit_node in sink_nodes:
    #         try:
    #             sink_dists.append(nx.shortest_path_length(G, source=sink, target=exit_node))
    #         except nx.NetworkXNoPath:
    #             pass
    #     if len(src_dists) == 0:
    #         G[src][sink]['src_dist'] = 0
    #     else:
    #         G[src][sink]['src_dist'] = min(src_dists)
    #     if len(sink_dists) == 0:
    #         G[src][sink]['sink_dist'] = 0
    #     else:
    #         G[src][sink]['sink_dist'] = min(sink_dists)



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
            Wt += 65 * G[src][sink]['volume'] * soln[:, 2*n_edges + G[src][sink]['idx']]
        else:
            Wt += 65 * G[src][sink]['volume'] * soln[:, n_edges + G[src][sink]['idx']]
    return Wt


def run_example_sim(n):

    G = create_network_multiple_sources_and_sinks(n)
    sims.create_source(G)
    importlib.reload(system)
    times = np.linspace(0, 3600, 3600 + 1)
    y0 = np.zeros(2 * len(G.edges()))
    soln = scipy.integrate.odeint(system.dydt, y0, times)
    wt = get_Wt(G, times, soln)

    plt.figure(1)
    plt.plot(times / 60, wt)
    plt.xlabel('time (min)')
    plt.ylabel('w(t)')
    plt.show()

    plt.figure(2)
    radii = [G[src][sink]['radius'] for src, sink in G.edges()]
    plt.hist(radii)
    plt.show()

if __name__ == '__main__':

    for dp in [20, 50, 100]:
        for n in [13, 14, 15]:
            for k in [5,6,7,8]:
                start = time.time()
                G = create_network_multiple_sources_and_sinks(n, dp=dp)
                end = time.time()
                print('Created %i - %i in %.01f seconds.' % (n, k, end-start))
                nx.write_gpickle(G, 'C:/Users/Bill/Documents/Python/hemo/hemo/data/networks%i/G_%i_%i.gpickle' % (dp, n, k))

    #G = create_network_multiple_sources_and_sinks(4)