import numpy as np
from string import Template


def create_source(G, *, v0=5, B=30, k2=0.35, K=34.2, k3=5.0e-4, k4=0, filename="system.py"):
    """Creates a system source file.


    Creates a system source file to be imported and used for numerical integration.

    Parameters
    ----------
    G
        The graph structure
    v0 : float
        The upstream concentration
    B : float
        Vascular volume ratio (default 30)
    k2 : float
        Maximal transport rate (default 0.35)
    K : float
        Concentration at which transport is half maximal (default 34.2)
    k3 : float
        Clearance rate of drug default 5.0e-4)
    filename : str
        What to name the file containing dydt

    Returns
    -------

    """
    parms = {'B': B, 'k2': k2, 'K': K, 'k3': B*k3, 'v0': v0, 'k4': k4}

    parms['n'] = len(G.edges())
    parms['n2'] = 2 * parms['n']

    with open('c:/users/bill/documents/python/hemo/hemo/data/basic_template', 'r') as template_file:
        s = Template(template_file.read())

    with open(filename, 'w') as file:
        file.write(s.substitute(parms))

        for src, sink in G.edges():
            edge = G[src][sink]
            f = edge['inverse_transit_time']
            id = edge['idx']

            dv = "    dv[%d] = -%.18f*v[%d] - (%.18f * v[%d])/(%.18f + v[%d])" % (
            id, f, id, parms['k2'], id, parms['K'], id)
            dw = "    dw[%d] = + (%.18f * v[%d])/(%.18f + v[%d]) - %.18f*w[%d]" % (
            id, parms['k2'] / parms['B'], id, parms['K'], id, parms['B'] *parms['k3'], id)

            if G.node[src]['ntype'] == 'source':
                dv += " + %.18f * np.exp(-%.18f * t)" % (f * parms['v0'], parms['k4'])

            else:
                parent_edge_volumes = []
                for par_idx in G[src][sink]['parents']:
                    for src2, sink2 in G.edges():
                        if G[src2][sink2]['idx'] == par_idx:
                            parent_edge_volumes.append(G[src2][sink2]['volume'])

                pvs = np.sum(parent_edge_volumes)

                for i, par_idx in enumerate(G[src][sink]['parents']):
                    dv += " + %.18f * (%.18f * v[%d])/%.18f" % (f, parent_edge_volumes[i], par_idx, pvs)

            file.write("%s\n%s\n\n" % (dv, dw))
        file.write("    return np.concatenate((dv, dw))")