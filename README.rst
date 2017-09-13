See hemo/data/networks for example networks to be used in publication.

Any of these networks can be imported using the following code:

import networkx as nx
G = nx.read_gpickle('path/file')

In scripts/read_nets.py there is a function to graph networks, uses a heat map to indicate radius of vessels.

