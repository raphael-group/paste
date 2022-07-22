from typing import List
from anndata import AnnData
import numpy as np
import scipy
import ot

def filter_for_common_genes(
    slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

def match_spots_using_spatial_heuristic(
    X,
    Y,
    use_ot: bool = True) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
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

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))

def apply_trsf(
    M: np.ndarray,
    translation: List[float],
    points: np.ndarray) -> np.ndarray:
    """
    Apply a rotation from a 2x2 rotation matrix `M` together with
    a translation from a translation vector of length 2 `translation` to a list of
    `points`.

    Args:
        M (nd.array): a 2x2 rotation matrix.
        translation (nd.array): a translation vector of length 2.
        points (nd.array): a nx2 array of `n` points 2D positions.

    Returns:
        (nd.array) a nx2 matrix of the `n` points transformed.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    trsf = np.identity(3)
    trsf[:-1, :-1] = M
    tr = np.identity(3)
    tr[:-1, -1] = -translation
    trsf = trsf @ tr

    flo = points.T
    flo_pad = np.pad(flo, ((0, 1), (0, 0)), constant_values=1)
    return ((trsf @ flo_pad)[:-1]).T

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]