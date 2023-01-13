from os import listdir
from importlib_metadata import List
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from facenet_models import FacenetModel

import functions as fn
import database as db


class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """
        Parameters
        ----------
        ID : int
            A unique identifier for this node. Should be a
            value in [0, N-1], if there are N nodes in total.

        neighbors : Sequence[int]
            The node-IDs of the neighbors of this node.

        descriptor : numpy.ndarray
            The shape-(512,) descriptor vector for the face that this node corresponds to.

        truth : Optional[str]
            If you have truth data, for checking your clustering algorithm,
            you can include the label to check your clusters at the end.
            If this node corresponds to a picture of Ryan, this truth
            value can just be "Ryan"

        file_path : Optional[str]
            The file path of the image corresponding to this node, so
            that you can sort the photos after you run your clustering
            algorithm
        """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path
    def __repr__(self):
        return str(self.id)


def plot_graph(graph, adj):
    
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.
    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.
    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot. This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax

def create_nodes(photos: List):
    '''
    Takes in a list of file paths representing photos and returns 
    a list of nodes and a corresponding adjacency matrix

    Parameters
    -----------
    photos : List, shape = (N,)
        List of photo file paths

    Returns
    ----------
    nodes: Tuple[Node], shape = (N,)
        List containing the nodes of each photo
    adj: np.ndarray(N, N)
        Corresponding adjacency matrix for the nodes
    '''
    N = len(photos)
    nodes = []
    model = FacenetModel()
    descriptors = []
    adj = np.zeros((N, N))
    cosine_dist = []
    
    #creates descriptors for each image

    # TODO: write conditional check; if length of boxes != 1, then photo has 1+ face + skip it
    #       N = # of faces detected, not photos --> for adjacency matrix
    for i in range(N):
    
        photo = fn.load_image(photos[i])
        boxes, probabilities, landmarks = model.detect(photo)

        if len(boxes) == 1:
            for descriptor in model.compute_descriptors(photo, boxes):
                descriptors.append((descriptor, photos[i]))
        
    # descriptors is a shape (N, 512) numpy array
    descriptors = np.array(descriptors)
    
    #populates adjacency matrix
    for i in range(len(descriptors)):
        for j in range(len(descriptors)):
            dist = fn.cos_distance(descriptors[i, 0], descriptors[j, 0])
            cosine_dist.append(dist)
            if ((i != j) and (dist <= db.threshhold)):
                adj[i,j] = 1/dist**2
                adj[j,i] = 1/dist**2

    for i in range (len(descriptors)):
        node = Node(ID=i, neighbors=np.nonzero(adj[i])[0], descriptor=descriptors[i, 0], file_path = descriptors[i, 1]) # keyword assignment
        nodes.append(node)
    
    return (tuple(nodes), adj, cosine_dist)


def connected_components(nodes: List):
    '''
    Finds all connected components in given graph based on label, and stores them in 2d list. 

    Parameters
    -----------
    nodes : List
        List of all nodes in graph. Should be taken from create_nodes() in test file. 

    Returns
    ----------
    labels : List
        List of list of connected components based on label. 
    '''
    
    labels = []
    labels_covered = []

    for node_idx in range(len(nodes)):
        node_label = nodes[node_idx].label
        label_matches = []
        
        if node_label not in labels_covered:
            labels_covered.append(node_label)
            print("node label" + str(node_label))
            for node2_idx in range(len(nodes)):
                print("nodes[node2_idx].label" + str(nodes[node2_idx].label))
                if nodes[node2_idx].label == nodes[node_idx].label:
                    print("hit")
                    label_matches.append(nodes[node2_idx])

            labels.append(label_matches)
    
    return labels
    
def propagate_label(nd : Node, adjmatrix : np.ndarray, nodes: List):
    """
    propagate_label(nd : Node, adjmatrix : np.ndarray):

    Updates node labels based on the weights of neighbors

    Parameters:

    nd : Node
        Singular Node to be updated

    adjmatrix : 
        N by N matrix 

    Returns:
        Nothing
        Updates nd.label to be the neighbor with the greatest weightt
 
    """
    n = nd.id

    m = -1
    resid = -1

    # TODO: neighbor labels need to be tallied up based on adjmatrix weights and then the most significant sum selected 
    #       instead of simply taking the largest label 

    sums = np.zeros(adjmatrix.shape[0])
    
    for neighbor in nd.neighbors:
        sums[nodes[neighbor].label] += adjmatrix[n, nodes[neighbor].id] # <- this is weight of edge between n and neighbor
        
    '''
    for weight in sums:
        if(adjmatrix[n][neighbor] > m):
            m = adjmatrix[n][neighbor]
            resid = neighbor
    '''
    
    nd.label = np.argmax(sums)
    
        

def whispers(prop_times : int, nodes: List, adjmatrix : np.ndarray, print_cc=False):
    """
    calls propagate_label on a random node with a given set of times

    Parameters
    -----------

    prop_times : int
        Amount of times propagate_label is called
    
    nodes: list
        list of nodes to perform the whispers algorithnm on
    
    Returns : 
        Nothing, updates the labels to the nearest neighbor / biggest weight of random nodes

    -----------
        

        """
    #what var for number of nodes, curr N
    N = len(nodes)

    for i in range(prop_times):
        select_node = nodes[np.random.randint(N)]
        propagate_label(select_node, adjmatrix, nodes)
        
        if print_cc:
            print(connected_components(nodes))
    
    
    #for i in range(len(dists)):
        #if dists[i] < db.threshold:
            
def display_sorted(labels: np.ndarray):
    labels = np.array(labels)
    longest = max([len(labels[i]) for i in range(len(labels))])
    fig, axs = plt.subplots(longest, len(labels))
    for j in range(len(labels)):
        for i in range(len(labels[j])):
        
            #axs[i, j].imshow(functions.load_image(nodes[labels[i][j]].file_path))

            axs[i, j].imshow(fn.load_image(labels[j][i].file_path))