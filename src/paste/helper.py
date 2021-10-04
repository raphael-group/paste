import numpy as np
import scipy
import ot

def filter_for_common_genes(slices):
    """
    param: slices - list of slices (AnnData objects)
    """
    assert len(slices) > 0, "Cannot have empty list."
    
    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    
    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)
    
    return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    
    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)


def intersect(lst1, lst2): 
    """
    param: lst1 - list
    param: lst2 - list
    
    return: list of common elements
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3 

def norm_and_center_coordinates(X): 
    """
    param: X - numpy array
    
    return: 
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))


def match_spots_using_spatial_heuristic(X,Y,use_ot=True):
    """
    param: X - numpy array
    param: Y - numpy array
    
    return: pi- mapping of spots using spatial heuristic
    """
    n1,n2=len(X),len(Y)
    X,Y = norm_and_center_coordinates(X),norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X,Y)
    if use_ot:
        pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        pi = np.zeros((n1,n2))
        pi[row_ind, col_ind] = 1/max(n1,n2)
        if n1<n2: pi[:, [(j not in col_ind) for j in range(n2)]] = 1/(n1*n2)
        elif n2<n1: pi[[(i not in row_ind) for i in range(n1)], :] = 1/(n1*n2)
    return pi

## Covert a sparse matrix into a dense matrix
to_dense_array = lambda X: np.array(X.todense()) if isinstance(X,scipy.sparse.csr.spmatrix) else X

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep] 