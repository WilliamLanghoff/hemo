import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D


def prep_net_for_sims(G):
    """Takes network already embedded in R3, prepares for simulation.

    Each edge should already have 'length', 'radius' assigned as desired.
    Calculates pressure at each node, inverse transit times, does some cleanup.

    Parameters
    ----------
    G
        Graph Structure, embedded in R3

    Returns
    -------
    G
        Graph Structure, ready to use for sims
    """



    set_pressures(G)
    set_inverse_transit_times(G)
    redirect(G)

    assign_edge_indices(G)
    set_volumes(G)
    set_parent_indices(G)

    return G


def calculate_lengths(G):
    """Calculates the length of each edge of the graph.

    The graph G already has a node attribute 'pos' : [x, y, z], this calculates the distances between nodes on each
    edge and assigns it to an edge attribute 'length'

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    for src, sink in G.edges():
        v = np.asarray(G.node[src]['pos']) - np.asarray(G.node[sink]['pos'])
        G[src][sink]['length'] = np.linalg.norm(v)





def set_volumes(G):
    """Calculates volume of each edge

    The graph G already has edge attributes 'length' and 'radius', this calculates the volume of each node. Must be run
    prior to creating system source for conservation of mass term in ODE.

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    for src, sink in G.edges():
        G[src][sink]['volume'] = np.pi * G[src][sink]['radius'] ** 2 * G[src][sink]['length']


def redirect(G):
    """Reverse direction of edges which have negative flow.

    After determining flows through the network, the initial directions of edges may be opposite the direction
    of blood flow. This method reverses the direction of those edges so that all inverse transit times are positive.

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    made_switch = True
    made_any_switch = False

    while made_switch:
        made_switch = False
        for src, sink in G.edges():
            if G[src][sink]['inverse_transit_time'] < 0:
                r = G[src][sink]['radius']
                itt = -G[src][sink]['inverse_transit_time']
                G.remove_edge(src, sink)
                G.add_edge(sink, src)
                G[sink][src]['radius'] = r
                G[sink][src]['length'] = np.linalg.norm(
                    np.asarray(G.node[sink]['pos']) - np.asarray(G.node[src]['pos']))
                G[sink][src]['inverse_transit_time'] = itt
                made_switch = True
                made_any_switch = True
                # print('switched!')

        if made_any_switch:
            set_pressures(G)
            set_inverse_transit_times(G)


def set_pressures(G):
    """Ensures that all nodes have their pressure computed.

    Each edge already has attributes 'length', 'radius', 'capacitance', 'mat_idx' assigned to them.
    Using Hagen-Pousielle equation and Kirchoff's current law, calculates pressure at each internal node.

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """

    def assign_matrix_indices(G):
        """Assign matrix col/row index for each internal node

        Must be run before set_pressures

        Parameters
        ----------
        G
            Graph Structure

        Returns
        -------

        """
        m_iter = 0
        for node in G.nodes():
            if G.node[node]['ntype'] == 'internal':
                G.node[node]['mat_idx'] = m_iter
                m_iter += 1
        G.graph['n_internal'] = m_iter

    assign_matrix_indices(G)
    viscosity = 3.5

    def set_capacitance(g):
        for src2, sink2 in g.edges():
            g[src2][sink2]['capacitance'] = (np.pi * g[src2][sink2]['radius'] ** 4) / (
                8 * viscosity * g[src2][sink2]['length'])

    set_capacitance(G)

    A = np.zeros((G.graph['n_internal'], G.graph['n_internal']))
    for src, sink in G.edges():
        if G.node[src]['ntype'] == 'internal':
            src_mat_idx = G.node[src]['mat_idx']
            A[src_mat_idx, src_mat_idx] += G[src][sink]['capacitance']
            if G.node[sink]['ntype'] == 'internal':
                sink_mat_idx = G.node[sink]['mat_idx']
                A[src_mat_idx, sink_mat_idx] = -G[src][sink]['capacitance']
                A[sink_mat_idx, src_mat_idx] = -G[src][sink]['capacitance']
        if G.node[sink]['ntype'] == 'internal':
            sink_mat_idx = G.node[sink]['mat_idx']
            A[sink_mat_idx, sink_mat_idx] += G[src][sink]['capacitance']
            if G.node[src]['ntype'] == 'internal':
                src_mat_idx = G.node[src]['mat_idx']
                A[src_mat_idx, sink_mat_idx] = -G[src][sink]['capacitance']
                A[sink_mat_idx, src_mat_idx] = -G[src][sink]['capacitance']

    # print(A)

    p0 = 25 * 133.322387415
    pN = 0

    b = np.zeros(G.graph['n_internal'])
    for node1 in G.nodes():
        if G.node[node1]['ntype'] == 'internal':
            for otherNode in G.nodes():
                if G.node[otherNode]['ntype'] == 'source' and G.has_edge(otherNode, node1):
                    b[G.node[node1]['mat_idx']] = p0 * G[otherNode][node1]['capacitance']
                elif G.node[otherNode]['ntype'] == 'sink' and G.has_edge(node1, otherNode):
                    b[G.node[node1]['mat_idx']] = pN * G[node1][otherNode]['capacitance']
    # print(b)

    p = np.linalg.solve(A, b)
    # print(p/133.322387415)

    for node1 in G.nodes():
        if G.node[node1]['ntype'] == 'internal':
            G.node[node1]['pressure'] = p[G.node[node1]['mat_idx']]
        elif G.node[node1]['ntype'] == 'source':
            G.node[node1]['pressure'] = p0
        elif G.node[node1]['ntype'] == 'sink':
            G.node[node1]['pressure'] = pN


def assign_edge_indices(G):
    """Assigns an index to each edge, this will be its index in the ODE system

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    counter = 0
    for src, sink in G.edges():
        G[src][sink]['idx'] = counter
        counter += 1
        G[src][sink]['parents'] = []


def set_parent_indices(G):
    """Set up parent edge indices for each edge

    Must be run after assign_edge_indices, so that each edge has 'idx' attribute assigned.
    Needed for conservation of mass term in ODE.

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    for src1, sink1 in G.edges():
        for src2, sink2 in G.edges():
            if src1 == sink2:
                G[src1][sink1]['parents'].append(G[src2][sink2]['idx'])


def set_inverse_transit_times(G):
    """Ensures inverse transit times have been calculated.

    Must be run after set_pressures, assigns 'inverse_transit_time' attribute to each edge.

    Parameters
    ----------
    G
        Graph Structure

    Returns
    -------

    """
    viscosity = 3.5
    for src, sink in G.edges():
        dp = G.node[src]['pressure'] - G.node[sink]['pressure']
        G[src][sink]['inverse_transit_time'] = dp * (G[src][sink]['radius'] ** 2) / (
            8 * viscosity * G[src][sink]['length'] ** 2)



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

    def plot_nodes(ax, nodes, color):

        xs = [G.node[n]['pos'][0] for n in nodes]
        ys = [G.node[n]['pos'][1] for n in nodes]
        zs = [G.node[n]['pos'][2] for n in nodes]
        ax.scatter(xs, ys, zs, c=color)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    sources, internal, sinks = [], [], []

    for n in G.nodes():
        if G.node[n]['ntype'] == 'source':
            sources.append(n)
        elif G.node[n]['ntype'] == 'sink':
            sinks.append(n)
        elif G.node[n]['ntype'] == 'internal':
            internal.append(n)

    plot_nodes(ax, sources, 'g')
    plot_nodes(ax, sinks, 'b')
    plot_nodes(ax, internal, 'k')

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
        plt.savefig('%s.pdf' % filename)

