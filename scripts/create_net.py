import hemo.net_prep as net_prep

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D

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

    if central_difference(G, (src1, sink1)) < central_difference(G, (src2, sink2)):
        src1, src2 = src2, src1
        sink1, sink2 = sink2, sink1

    r1 = G[src1][sink1]['radius']
    L1 = G[src1][sink1]['length']
    r2 = G[src2][sink2]['radius']
    L2 = G[src2][sink2]['length']

    delta_r1 = 0.1 * (r1 + r2) / 2
    v1 = L1 * r1 ** 2
    v2 = L2 * r2 ** 2
    vv = L1 * (r1 + delta_r1) ** 2
    if v1 + v2 > vv:
        new_rad_2 = np.sqrt((v1 + v2 - vv) / L2)
        new_rad_1 = r1 + delta_r1

        if new_rad_2 >= 0.0001 and new_rad_2 >= 0.0001:
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


def plot_3d_network(G, title=None, filename=None):
    """Creates 3d plot of the network

    Parameters
    ----------
    G
        Graph Structure
    title : str
        Title for the plot
    filename : str
        filename for the plot. If specified, plot is saved

    Returns
    -------

    """
    xs = [G.node[n]['pos'][0] for n in G.nodes()]
    ys = [G.node[n]['pos'][1] for n in G.nodes()]
    zs = [G.node[n]['pos'][2] for n in G.nodes()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    for n in G.nodes():
        if G.node[n]['ntype'] == 'source':
            xsrc = G.node[n]['pos'][0] - 0.15
            ysrc = G.node[n]['pos'][1]
            zsrc = G.node[n]['pos'][2] + 0.01
        elif G.node[n]['ntype'] == 'sink':
            xsink = G.node[n]['pos'][0] + 0.05
            ysink = G.node[n]['pos'][1] - 0.05
            zsink = G.node[n]['pos'][2]

    ax.text(xsrc, ysrc, zsrc, 'Entrance')
    ax.text(xsink, ysink, zsink, 'Exit')

    for src, sink in G.edges():
        xs = [G.node[src]['pos'][0], G.node[sink]['pos'][0]]
        ys = [G.node[src]['pos'][1], G.node[sink]['pos'][1]]
        zs = [G.node[src]['pos'][2], G.node[sink]['pos'][2]]
        ax.plot3D(xs, ys, zs, 'r')

    if title is None:
        ax.set_title('Network, N=%i' % G.graph['N'])
    else:
        ax.set_title(title)
    ax.set_frame_on(False)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_autoscale_on(True)
    ax.set_autoscalez_on(True)
    ax.set_zticklabels([])
    ax.set_zticks([])
    ax.w_zaxis.line.set_lw(0.)

    if filename is not None:
        path = Path('data/img')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig('data/img/%s.pdf' % filename)


def create_network(N, *, space_volume=1, vascular_fraction=0.015, radii_iters=1):
    """Create the graph structure.

    Calls several functions to create the graph.

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
            net_prep.set_volumes(G)
    make_switches(G)

    net_prep.prep_net_for_sims(G)

    return G


if __name__ == '__main__':
    G = create_network(4)