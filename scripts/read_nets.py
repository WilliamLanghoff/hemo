import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl



def create_dist_plots():
    for n in [4,5,6,7,8,9,10]:
        G = nx.read_gpickle('G_%i.gpickle' % n)
        radii = [10**4 *G[src][sink]['radius'] for src, sink in G.edges()]
        plt.hist(radii, bins=25, normed=True)
        plt.title('Radius Distribution, %i vessels' % len(G.edges()))
        plt.xlabel('Radius (um)')
        plt.savefig('radii_%i.pdf' % n, dpi=300)
        plt.show()


def plot_3d_network_radii_heatmap(G, title=None, filename=None):
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

        zs = [G.node[n]['pos'][0] for n in nodes]
        ys = [G.node[n]['pos'][1] for n in nodes]
        xs = [G.node[n]['pos'][2] for n in nodes]
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

    radii = [G[src][sink]['radius'] for src, sink in G.edges()]

    viridis = plt.get_cmap('viridis')
    cNorm = colors.LogNorm(vmin=min(radii), vmax=max(radii))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=viridis)

    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=viridis,
                                    norm=cNorm,
                                    orientation='horizontal',
                                    ticks=[min(radii), max(radii)])

    cb1.ax.set_xticklabels(['%.0f $\mu$m' %(10**4 * min(radii)), '%.0f $\mu$m' % (10**4 * max(radii))])


    for src, sink in G.edges():
        zs = [G.node[src]['pos'][0], G.node[sink]['pos'][0]]
        ys = [G.node[src]['pos'][1], G.node[sink]['pos'][1]]
        xs = [G.node[src]['pos'][2], G.node[sink]['pos'][2]]
        colorVal = scalarMap.to_rgba(G[src][sink]['radius'])
        ax.plot3D(xs, ys, zs, color=colorVal)

    plot_nodes(ax, sources, 'b')
    plot_nodes(ax, sinks, 'r')
    plot_nodes(ax, internal, 'k')



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

    plt.show()

    if filename is not None:
        path = Path('data/img')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig('%s.pdf' % filename)


def read_volumetric_flow(n):
    inflow = 0
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks/G_%i.gpickle' % n)
    for src, sink in G.edges():
        if G.node[src]['ntype'] == 'source':
            inflow += G[src][sink]['volume'] * G[src][sink]['inverse_transit_time']
    return inflow

if __name__ == '__main__':
    G = nx.read_gpickle('C:/Users/Bill/Documents/Python/hemo/hemo/data/networks100/G_7_8.gpickle')
    plot_3d_network_radii_heatmap(G)