import hemo.net_prep as net_prep
import hemo.sims as sims
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import system
import importlib


viscosity = 3.5


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



def perturb_nodes(G):
    """Perturbs the nodes of the graph, each within a radius of delta/4

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    max_dist = 0.125 * G.graph['delta']
    for n in G.nodes():
        # if n in (0, 1, 21, 22):
        # continue
        u = np.random.randn(3)
        u /= np.linalg.norm(u)
        u *= max_dist
        newpos = G.node[n]['pos'] + u
        if (0 <= newpos[0] <= 1) and (0 <= newpos[1] <= 1) and (0 <= newpos[2] <= 1):
            G.node[n]['pos'] = newpos
        else:
            print('Invalid Node Perturbation')




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


def perturb_radii(G, e1):
    """Perturbs the radius of a given edge, and some of its neighbors, so that the total volume remains fixed.

    Parameters
    ----------
    G
        Graph Structure
    e1
        Edge which will have its radius perturbed

    Returns
    -------

    """
    initial_volume = get_total_volume(G)
    src1, sink1 = G.edges()[e1]

    if len(G.successors(sink1)) == 0:
        # edge is incident to exit node
        sink2 = src1
        src2 = np.random.choice(G.predecessors(src1))

    elif len(G.predecessors(src1)) == 0:
        # edge is incident to entrance node
        src2 = sink1
        sink2 = np.random.choice(G.successors(sink1))

    else:
        src2 = sink1
        sink2 = np.random.choice(G.successors(sink1))

    if G[src1][sink1]['center_dist'] > G[src2][sink2]['center_dist']:
        src1, src2 = src2, src1
        sink1, sink2 = sink2, sink1

    r1 = G[src1][sink1]['radius']
    L1 = G[src1][sink1]['length']
    r2 = G[src2][sink2]['radius']
    L2 = G[src2][sink2]['length']

    delta_r1 = np.random.rand()*0.5 * (r1 + r2) / 2
    v1 = L1 * r1 ** 2
    v2 = L2 * r2 ** 2


    while L1 * (r1 + delta_r1) ** 2 >= v1 + v2:
        delta_r1 *= 0.9
    vv = L1 * (r1 + delta_r1) ** 2
    new_rad_2 = np.sqrt((v1 + v2 - vv) / L2)
    new_rad_1 = r1 + delta_r1

    if new_rad_2 >= 10e-6 and new_rad_2 >= 10e-6:
        G[src2][sink2]['radius'] = new_rad_2
        G[src1][sink1]['radius'] = new_rad_1
        # else:
        # print('invalid perturbation, radii too small.')
    final_volume = get_total_volume(G)
    assert np.abs(initial_volume - final_volume) < 10e-14


def make_switches(G):
    """Switch radii so that internal nodes have smaller radius than external"""
    for src1, sink1 in G.edges():
        cd1 = central_difference(G, (src1, sink1))
        r1 = G[src1][sink1]['radius']
        for src2, sink2 in G.edges():
            if src1 == src2 or sink1 == sink2:
                continue
            cd2 = central_difference(G, (src2, sink2))
            r2 = G[src2][sink2]['radius']
            if cd1 < cd2 and r1 > r2:
                G[src1][sink1]['radius'] = r2
                G[src2][sink2]['radius'] = r1
                break


def central_difference(G, edge):
    """Given an edge in G, calculate how far it is from the center.

    Given an edge (e1, e2), finds distance from source to e1 and from e2 to sink.
    Returns absolute value of the difference of these distances.

    Parameters
    ----------
    G
        Graph Structure
    edge
        A pair (src, sink) representing an edge

    Returns
    -------
    int
        The 'distance' of that edge from the center

    """

    def dist_edge_node(G, edge, node):
        # Distance from an edge to a node is minimum of distance from endpoints to node
        src, sink = edge
        try:
            d1 = nx.shortest_path_length(G, source=src, target=node)
        except nx.NetworkXNoPath:
            d1 = nx.shortest_path_length(G, source=node, target=sink)
        return d1
        # try:
        #     d2 = nx.shortest_path_length(G, source=src, target=node)
        # except nx.NetworkXNoPath:
        #     d2 = nx.shortest_path_length(G, source=node, target=sink)
        # return min(d1, d2)

    d1 = dist_edge_node(G, edge, G.graph['source'])
    d2 = dist_edge_node(G, edge, G.graph['sink'])
    return abs(d1 - d2)


def create_network(N, *, space_volume=1, vascular_fraction=0.015, radii_iters=1):
    """Create an example graph structure.

    Creates an integer lattice embedded in a cubic centimeter. Perturbs radii and sorts them such that central vessels
    are smaller than external vessels. Source and sink on opposite corners of graph.

    Parameters
    ----------
    N : int
        Integer lattice # of subdivisions
    space_volume : float
        Volume of embedding space
    vascular_fraction : float
        fraction of embedding space which is vasculature
    radii_iters : int
        number of times to iterate through radii perturbation

    Returns
    -------
    G
        The graph structure

    """
    delta = 1 / (N + 1)
    viscosity = 3.5

    G = nx.DiGraph()
    G.graph['N'] = N
    G.graph['delta'] = delta

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

    # Next add the edges
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

    # Choose source and sink on opposite corners
    G.node[vd[(0, 0, 0)]]['ntype'] = 'source'
    G.node[vd[(N - 1, N - 1, N - 1)]]['ntype'] = 'sink'
    G.graph['source'] = vd[(0, 0, 0)]
    G.graph['sink'] = vd[(N - 1, N - 1, N - 1)]

    # Perturb the nodes to break symmetry
    # perturb_nodes(G)

    # assign lengths
    net_prep.calculate_lengths(G)

    # assign initial radius uniformly
    total_len = np.sum([G[src][sink]['length'] for src, sink in G.edges()])
    r = np.sqrt((vascular_fraction * space_volume) / (np.pi * total_len))
    for src, sink in G.edges():
        G[src][sink]['radius'] = r

    # perturb radii to further break symmetry
    for j in range(radii_iters):
        for i in range(len(G.edges())):
            perturb_radii(G, i)
            # net_prep.set_volumes(G)
    make_switches(G)

    net_prep.prep_net_for_sims(G)

    return G


def create_network_multiple_sources_and_sinks(N, *, space_volume=1, radii_iters=10):
    delta = 1 / (N + 1)
    viscosity = 3.5

    G = nx.DiGraph()
    G.graph['N'] = N
    G.graph['delta'] = delta

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

    # Next add the edges
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

    # Sources and sinks are alternating nodes on opposite faces
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

    # assign lengths
    net_prep.calculate_lengths(G)

    assign_distances_from_source_and_sink_nodes(G)

    # assign initial radius uniformly
    total_len = np.sum([G[src][sink]['length'] for src, sink in G.edges()])
    r = np.sqrt((vascular_fraction * space_volume) / (np.pi * total_len))
    for src, sink in G.edges():
        G[src][sink]['radius'] = r

    # perturb radii to further break symmetry
    for j in range(radii_iters):
        for i in range(len(G.edges())):
            perturb_radii(G, i)
            # net_prep.set_volumes(G)
    # make_switches(G)

    net_prep.prep_net_for_sims(G)

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

    source_nodes, sink_nodes = [], []
    for node in G.nodes():
        if G.node[node]['ntype'] == 'source':
            source_nodes.append(node)
        elif G.node[node]['ntype'] == 'sink':
            sink_nodes.append(node)

    for src, sink in G.edges():
        src_dists, sink_dists = [], []
        for entrance_node in source_nodes:
            try:
                src_dists.append(nx.shortest_path_length(G, source=entrance_node, target=src))
            except nx.NetworkXNoPath:
                pass
        for exit_node in sink_nodes:
            try:
                sink_dists.append(nx.shortest_path_length(G, source=sink, target=exit_node))
            except nx.NetworkXNoPath:
                pass
        if len(src_dists) == 0:
            G[src][sink]['src_dist'] = 0
        else:
            G[src][sink]['src_dist'] = min(src_dists)
        if len(sink_dists) == 0:
            G[src][sink]['sink_dist'] = 0
        else:
            G[src][sink]['sink_dist'] = min(sink_dists)
        G[src][sink]['center_dist'] = abs(G[src][sink]['src_dist'] - G[src][sink]['sink_dist'])


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
    G = create_network_multiple_sources_and_sinks(5)
    radii = [G[src][sink]['radius'] for src, sink in G.edges()]
    plt.hist(radii)
    plt.show()