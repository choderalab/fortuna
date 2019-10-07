import numpy as np
import networkx as nx


def realize_experiment(g,samples_per_edge=10000):
    """
    Perform a stochastic realization of an experiment.
    
    Adds data onto graph
    
    Parameters
    ----------
    Ntotal : int 
        Total number of samples collected
    restart : bool, optional, default=False
        If True, will clear all sampled data from the graph before running the simulation.
        
    """
    edges = list(g.edges)
    n_edges = len(edges)

    for edge in g.edges():
        i,j = edge
        if 'samples' not in g.edges[edge]:
            g.edges[edge]['samples'] = []
        for _ in range(samples_per_edge):
            s = g.edges[edge]['statistical_fluctuation']
            sample = s*np.random.randn() + (g.nodes[j]['true_free_energy'] - g.nodes[i]['true_free_energy'])
            g.edges[edge]['samples'].append(sample)
        g.edges[edge]['f_ij'] = np.mean(g.edges[edge]['samples'])
        g.edges[edge]['df_ij'] = np.std(g.edges[edge]['samples']) / np.sqrt(len(g.edges[edge]['samples']))

def assign_uniform_weights(g):
    """Assign uniform sampling weights to each node"""
    n_edges = len(list(g.edges))
    for edge in g.edges:
        g.edges[edge]['weight'] = 1.0 / n_edges


def mle(g,estimate=False):
    """
    Compute maximum likelihood estimate of free energies and covariance in their estimates.

    We assume the free energy of node 0 is zero.
    
    Parameters
    ----------
    g : nx.Graph
        The graph for which an estimate is to be computed
        Each edge must have attributes 'f_ij' and 'df_ij' for the free energy and uncertainty estimate
        Will have 'bayesian_f_ij' and 'bayesian_df_ij' added to each edge
        and 'bayesian_f_i' and 'bayesian_df_i' added to each node.
        
    Returns
    -------
    f_i : np.array with shape (n_ligands,)
        f_i[i] is the absolute free energy of ligand i in kT
        f_i[0] = 0
    
    C : np.array with shape (n_ligands, n_ligands)
        C[i,j] is the covariance of the free energy estimates of i and j
    
    """
    N = len(g.nodes)
    if estimate:
        f_ij = form_edge_matrix(g, 'f_ij_est', action='antisymmetrize')
        df_ij = form_edge_matrix(g, 'df_ij_est', action='symmetrize')
    else:
        f_ij = form_edge_matrix(g, 'f_ij', action='antisymmetrize')
        df_ij = form_edge_matrix(g, 'df_ij', action='symmetrize')
    
    # Form F matrix (Eq 4)
    F = np.zeros([N,N])
    for (i,j) in g.edges:
        F[i,j] = - df_ij[i,j]**(-2)
        F[j,i] = - df_ij[i,j]**(-2)
    for i in g.nodes:
        F[i,i] = - np.sum(F[i,:])

    # Form z vector (Eq 3)
    z = np.zeros([N])
    for (i,j) in g.edges:
        z[i] += f_ij[i,j] * df_ij[i,j]**(-2)
        z[j] += f_ij[j,i] * df_ij[j,i]**(-2)
        
    # Compute MLE estimate (Eq 2)
    Finv = np.linalg.pinv(F)
    f_i = - np.matmul(Finv, z) # NOTE: This differs in sign from Eq. 2!
    f_i[:] -= f_i[0] # set x[0] = 0
    
    # Compute uncertainty
    C = Finv
    
    return f_i, C  


def has_redundant_cycles(g, min_cycles_per_node=2, verbose=False):
    """Return True if every node meets the specified minimum number of cycles per node.
    
    This is a quick-and-dirty implementation and uses a cycle basis, which is not guaranteed to be fully accurate.
    
    Parameters
    ----------
    g : nx.Graph
        The graph to check
    min_cycles_per_node : int, optional, default=2
        The minimum number of cycles (from iteration of a cycle basis) that each node must appear in
    verbose : bool optional, default=False
        If True, the number of cycles per node will be printed    
    
    Returns
    -------
    status : bool
        True if every node has the specified minimum number of cycles per node
    """
    g = g.to_undirected()
    
    cycles = nx.cycle_basis(g)
    cycles_per_node = { node : 0 for node in g.nodes() }
    for cycle in cycles:
        for node in cycle:
            cycles_per_node[node] += 1

    if verbose:
        print(cycles_per_node)
            
    for (node, ncycles) in cycles_per_node.items():
        if ncycles < min_cycles_per_node:
            return False

    return True

def average_degree(g):
    """Return the average degree of each node, which reflects the average number of transformations per molecule
    
    Parameters
    ----------
    g : nx.Graph
        The graph to check
        
    Returns
    -------
    degree : float 
        The average node degree
        
    """
    return np.mean(g.degree)

def create_graph(n_ligands=10, target_graph_degree=2.0, min_cycles_per_node=2, max_trials=1000, free_energy_stddev=3):
    """
    Use a simple stochastic process to generate a graph that resembles a typical relative free energy calculation map.
    
    Parameters
    ----------
    n_ligands : int, optional, default=10
        The number of ligands (nodes) in the graph
    target_graph_degree : float, optional, default=2
        The target average node degree; roughly the number of transformations per ligand
    min_cycles_per_node : int, optional, default=2
        The minimum number of cycles each node should be involved in.
        A value of 2 will give some cycle redundancy
    max_trials : int, optional, default=1000
        The number of edges to attempt deletions for
        Note that this is not a multistart process. The stochastic process can get stuck.
    free_energy_stddev : float, optional, default=3
        The standard deviation of the true absolute free energies to be drawn.
        
    Returns
    -------
    g : nx.Graph
        The generated graph
        Nodes will have the 'true_free_energy' attribute defined with the absolute free energy (in kT)
        Edges will have the 'statistical_fluctuation' attribute defined
    """
    # Start with a complete graph
    #g = nx.complete_graph(n_ligands)
    g = nx.DiGraph()
    for i in range(n_ligands):
        g.add_node(i)
    for i in range(n_ligands):
        for j in range(i+1, n_ligands):
            g.add_edge(i, j)

    # Assign random free energies
    for node in g.nodes:
        g.nodes[node]['true_free_energy'] = free_energy_stddev * np.random.randn()
    # Shift node 0 to be zero free energy
    offset = - g.nodes[0]['true_free_energy']
    for node in g.nodes:
        g.nodes[node]['true_free_energy'] += offset
    
    # Assign random statistical fluctuations
    for edge in g.edges:
        g.edges[edge]['statistical_fluctuation'] = 0.4*np.random.gamma(2, 5)

    
    # Prune edges at random until we hit the target average graph degree while still maintaining minimum cycle count
    for trial in range(max_trials):
        # Pick an edge to delete
        edges = list(g.edges)        
        index = np.random.choice(len(edges))
        i, j = edges[index]
        
        g_proposed = g.copy()
        g_proposed.remove_edge(i,j)
        # Accept if graph still has redundant cycles
        if has_redundant_cycles(g_proposed, min_cycles_per_node=min_cycles_per_node):
            g = g_proposed
        
        # Terminate if we have reached the target graph degree
        if average_degree(g) <= target_graph_degree:
            return g
    
    return g

def form_edge_matrix(g, label, step=None, action=None):
    """
    Extract the labeled property from edges into a matrix
    
    Parameters
    ----------
    g : nx.Graph
        The graph to extract data from
    label : str
        The label to use for extracting edge properties
    action : str, optional, default=None
        If 'symmetrize', will return a symmetric matrix where A[i,j] = A[j,i]
        If 'antisymmetrize', will return an antisymmetric matrix where A[i,j] = -A[j,i]
        
    """
    N = len(g.nodes)
    matrix = np.zeros([N,N])
    for i,j in g.edges:
        matrix[i,j] = g.edges[i,j][label]
        if action == 'symmetrize':
            matrix[j,i] = matrix[i,j]
        elif action == 'antisymmetrize':
            matrix[j,i] = -matrix[i,j]
        elif action is None:
            pass
        else:
            raise Exception(f'action "{action}" unknown.')
    return matrix
    
def form_node_array(g, label):
    N = len(g.nodes)
    array = np.zeros([N])
    for i in g.nodes:
        array[i] = g.nodes[i][label]
    return array

def reset_experiment(g,N0_ij):
    """
    This function will reset the number of steps taken, 
    and the estimation of the free energy, error, and stat flux.
    but makes no changes to the history of samples for an edge
    """
    for edge in g.edges(data=True):
        i, j, edge_data = edge
        n_steps = N0_ij[i,j]
        edge_data['n_samples'] = n_steps
        samples = edge_data['samples'][0:edge_data['n_samples']]
        edge_data['f_ij_est'] = np.mean(samples)  
        edge_data['df_ij_est'] = np.std(samples) / np.sqrt(len(samples))
        edge_data['statistical_fluctuation_est'] = np.std(samples)
