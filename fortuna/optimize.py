import numpy as np
from fortuna.graph import *

def fisher_information_matrix(g, Nij,estimate=False):
    #TODO change this to also take an estimate flag
    """Compute the true Fisher information matrix from the true statistical inefficiencies
    """
    n_nodes = len(g.nodes)
    F = np.zeros([n_nodes, n_nodes])

    if estimate:
        s_ij = form_edge_matrix(g, 'statistical_fluctuation_est', action='symmetrize')
    else:
        s_ij = form_edge_matrix(g, 'statistical_fluctuation', action='symmetrize')
    
    for (i,j) in g.edges:
        F[i,j] = -Nij[i,j] / s_ij[i,j]**2
        F[j,i] = F[i,j] # make symmetric
    for i in g.nodes:
        F[i,i] = -np.sum(F[i,:])
    
    return F

def covariance_matrix(F):
    return np.linalg.pinv(F)

def estimate_correlation_matrix(s_ij, N_ij):
    """Estimate the correlation matrix for the specified number of samples
    
    Parameters
    ----------
    s_ij : np.array with shape (n_ligands, n_ligands)
        Statistical fluctuation matrix (may be sparse with zeros)
        (should be symmetric)
    N_ij : np.array with shape (n_ligands, n_ligands)
        Number of samples per edge
        (should be symmetric)
    Returns
    -------
    C_ij : np.array with shape (n_ligands, n_ligands)
        Estimated correlation matrix for the given number of samples
    
    """
    n_nodes, _ = s_ij.shape
    F_ij = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(n_nodes):
            if N_ij[i,j] > 0:
                F_ij[i,j] = - N_ij[i,j] / s_ij[i,j]**2
    for i in range(n_nodes):
        F_ij[i,i] = - np.sum(F_ij[i,:])
    
    # Compute the covariance matrix
    C_ij = np.linalg.pinv(F_ij)
    
    return C_ij

def sensitivity_tensor(g, Nij, stepsize=1):
    """Compute the sensitivity tensor: the gradient in the covariance wrt allocating effort
    
    
    the true Fisher information matrix from the true statistical inefficiencies
    """
    n_nodes = len(g.nodes)
    S = np.zeros([n_nodes, n_nodes, n_nodes, n_nodes]) # sensitivity tensor

    s_ij = form_edge_matrix(g, 'statistical_fluctuation', action='symmetrize')
    
    # looping over edges (k,l)
    for (k,l) in g.edges:
      Nij[k,l] += stepsize
      Nij[l,k] += stepsize
      dF = np.zeros([n_nodes, n_nodes])

      # looping over edges (i,j)
      for (i,j) in g.edges:
          dF[i,j] = - 1 / s_ij[i,j]**2
          dF[j,i] = dF[i,j] # make symmetric 
      for i in g.nodes:
          dF[i,i] = - np.sum(dF[i,:])
     
      F = fisher_information_matrix(g, Nij,estimate=True)    
      C = np.linalg.pinv(F)
      
      Sij = - C * dF * C
      
      Nij[k,l] -= stepsize
      Nij[l,k] -= stepsize
      S[:,:,k,l] = Sij
      S[:,:,l,k] = Sij
        
    return S
