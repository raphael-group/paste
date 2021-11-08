import numpy as np
import anndata
import ot
from sklearn.decomposition import NMF
from scipy.spatial import distance_matrix
import scipy
from numpy import linalg as LA
from .helper import kl_divergence, intersect, to_dense_array, extract_data_matrix

def pairwise_align(sliceA, sliceB, alpha = 0.1, dissimilarity='kl', use_rep = None, G_init = None, a_distribution = None, b_distribution = None, norm = False, numItermax = 200, return_obj = False, verbose = False, **kwargs):
    """
    Calculates and returns optimal alignment of two slices. 
    
    param: sliceA - AnnData object
    param: sliceB - AnnData object
    param: alpha - Alignment tuning parameter. Note: 0 ≤ alpha ≤ 1
    param: dissimilarity - Expression dissimilarity measure: 'kl' or 'euclidean'
    param: use_rep - If none, uses slice.X to calculate dissimilarity between spots, otherwise uses the representation given by slice.obsm[use_rep]
    param: G_init - initial mapping to be used in FGW-OT, otherwise default is uniform mapping
    param: a_distribution - distribution of sliceA spots (1-d numpy array), otherwise default is uniform
    param: b_distribution - distribution of sliceB spots (1-d numpy array), otherwise default is uniform
    param: numItermax - max number of iterations during FGW-OT
    param: norm - scales spatial distances such that neighboring spots are at distance 1 if True, otherwise spatial distances remain unchanged
    param: return_obj - returns objective function output of FGW-OT if True, nothing if False
    param: verbose - FGW-OT is verbose if True, nothing if False
    
    return: pi - alignment of spots
    return: log['fgw_dist'] - objective function output of FGW-OT
    """
    
    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')
    
    # Calculate spatial distances
    D_A = distance_matrix(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance_matrix(sliceB.obsm['spatial'], sliceB.obsm['spatial'])
    
    # Calculate expression dissimilarity
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA,use_rep)), to_dense_array(extract_data_matrix(sliceB,use_rep))
    if dissimilarity.lower()=='euclidean' or dissimilarity.lower()=='euc':
        M = distance_matrix(A_X, B_X)
    else:
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence(s_A, s_B)
    
    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        a = a_distribution
        
    if b_distribution is None:
        b = np.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        b = b_distribution
    
    if norm:
        D_A /= D_A[D_A>0].min().min()
        D_B /= D_B[D_B>0].min().min()

        # M = M / np.sum(a[:, None] * b[None, :] * M)
        # const = uniform_gwloss(D_A,D_B,a,b)
        # D_A = D_A / np.sqrt(const)
        # D_B = D_B / np.sqrt(const)

        # pi1 , const1 = ot.emd(a, b, M , log=True)
        # const1 = const1['cost']
        # M= M/const1
        # pi2 , const2 = ot.gromov.gromov_wasserstein(D_A, D_B, a, b, loss_fun='square_loss', log=True, numItermax=numItermax,verbose=verbose)
        # const2 = const2['gw_dist']
        # D_A = D_A / np.sqrt(const2)
        # D_B = D_B / np.sqrt(const2)
    
    # Run OT
    if G_init is None:
        pi, logw = ot.gromov.fused_gromov_wasserstein(M, D_A, D_B, a, b, loss_fun='square_loss', alpha= alpha, log=True, numItermax=numItermax,verbose=verbose)
    else:
        pi, logw = my_fused_gromov_wasserstein(M, D_A, D_B, a, b, G_init = G_init, loss_fun='square_loss', alpha= alpha, log=True, numItermax=numItermax,verbose=verbose)
    
    if return_obj:
        return pi, logw['fgw_dist']
    return pi

def center_align(A, slices, lmbda = None, alpha = 0.1, n_components = 15, threshold = 0.001, max_iter = 10, dissimilarity='kl', use_rep = None, norm = False, random_seed = None, pis_init = None, distributions=None, verbose = False):
    """
    Computes center alignment of slices.
    
    param: A - Initialization of starting AnnData Spatial Object; Make sure to include gene expression AND spatial info
    param: slices - List of slices (AnnData objects) used to calculate center alignment
    param: lmbda - List of probability weights assigned to each slice; default is uniform weights
    param: n_components - Number of components in NMF decomposition
    param: threshold - Threshold for convergence of W and H
    param: max_iter - maximum number of iterations for solving for center slice
    param: dissimilarity - Expression dissimilarity measure: 'kl' or 'euclidean'
    param: use_rep - If none, uses slice.X to calculate dissimilarity between spots, otherwise uses the representation given by slice.obsm[use_rep]
    param: norm - scales spatial distances such that neighboring spots are at distance 1 if True, otherwise spatial distances remain unchanged
    param: random_seed - set random seed for reproducibility
    param: pis_init - initial list of mappings between 'A' and 'slices' to solver, otherwise will calculate default mappings
    param: distributions - distributions of spots for each slice (list of 1-d numpy array), otherwise default is uniform
    param: verbose
    
    return: center_slice - inferred center slice (AnnData object) with full and low dimensional representations (W, H) of
                            the gene expression matrix
    return: pi - List of pairwise alignment mappings of the center slice (rows) to each input slice (columns)
    """
    
    if lmbda is None:
        lmbda = len(slices)*[1/len(slices)]
    
    if distributions is None:
        distributions = len(slices)*[None]
    
    # get common genes
    common_genes = A.var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)

    # subset common genes
    A = A[:, common_genes]
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Run initial NMF
    if dissimilarity.lower()=='euclidean' or dissimilarity.lower()=='euc':
        model = NMF(n_components=n_components, init='random', random_state = random_seed, verbose = verbose)
    else:
        model = NMF(n_components=n_components, solver = 'mu', beta_loss = 'kullback-leibler', init='random', random_state = random_seed, verbose = verbose)
    
    if pis_init is None:
        pis = [None for i in range(len(slices))]
        W = model.fit_transform(A.X)
    else:
        pis = pis_init
        W = model.fit_transform(A.shape[0]*sum([lmbda[i]*np.dot(pis[i], to_dense_array(slices[i].X)) for i in range(len(slices))]))
    H = model.components_
    center_coordinates = A.obsm['spatial']
    
    if not isinstance(center_coordinates, np.ndarray):
        print("Warning: A.obsm['spatial'] is not of type numpy array .")
    
    # Initialize center_slice
    center_slice = anndata.AnnData(np.dot(W,H))
    center_slice.var.index = common_genes
    center_slice.obs.index = A.obs.index
    center_slice.obsm['spatial'] = center_coordinates
    
    # Minimize R
    iteration_count = 0
    R = 0
    R_diff = 100
    while R_diff > threshold and iteration_count < max_iter:
        print("Iteration: " + str(iteration_count))
        pis, r = center_ot(W, H, slices, center_coordinates, common_genes, alpha, dissimilarity = dissimilarity, norm = norm, G_inits = pis, distributions=distributions, verbose = verbose)
        W, H = center_NMF(W, H, slices, pis, lmbda, n_components, random_seed, dissimilarity = dissimilarity, verbose = verbose)
        R_new = np.dot(r,lmbda)
        iteration_count += 1
        R_diff = abs(R - R_new)
        print("Objective ",R_new)
        print("Difference: " + str(R_diff) + "\n")
        R = R_new
    center_slice = A.copy()
    center_slice.X = np.dot(W, H)
    center_slice.uns['paste_W'] = W
    center_slice.uns['paste_H'] = H
    center_slice.uns['full_rank'] = center_slice.shape[0]*sum([lmbda[i]*np.dot(pis[i], to_dense_array(slices[i].X)) for i in range(len(slices))])
    center_slice.uns['obj'] = R
    return center_slice, pis

#--------------------------- HELPER METHODS -----------------------------------

def center_ot(W, H, slices, center_coordinates, common_genes, alpha, dissimilarity = 'kl', norm = False, G_inits = None, distributions=None, verbose = False):
    center_slice = anndata.AnnData(np.dot(W,H))
    center_slice.var.index = common_genes
    center_slice.obsm['spatial'] = center_coordinates

    if distributions is None:
        distributions = len(slices)*[None]

    pis = []
    r = []
    print('Solving Pairwise Slice Alignment Problem:')
    for i in range(len(slices)):
        p, r_q = pairwise_align(center_slice, slices[i], alpha = alpha, dissimilarity = dissimilarity, norm = norm, return_obj = True, G_init = G_inits[i], b_distribution=distributions[i], verbose = verbose)
        pis.append(p)
        r.append(r_q)
    return pis, np.array(r)

def center_NMF(W, H, slices, pis, lmbda, n_components, random_seed, dissimilarity = 'kl', verbose = False):
    print('Solving Center Mapping NMF Problem:')
    n = W.shape[0]
    B = n*sum([lmbda[i]*np.dot(pis[i], to_dense_array(slices[i].X)) for i in range(len(slices))])
    if dissimilarity.lower()=='euclidean' or dissimilarity.lower()=='euc':
        model = NMF(n_components=n_components, init='random', random_state = random_seed, verbose = verbose)
    else:
        model = NMF(n_components=n_components, solver = 'mu', beta_loss = 'kullback-leibler', init='random', random_state = random_seed, verbose = verbose)
    W_new = model.fit_transform(B)
    H_new = model.components_
    return W_new, H_new

def my_fused_gromov_wasserstein(M, C1, C2, p, q, G_init = None, loss_fun='square_loss', alpha=0.5, armijo=False, log=False,numItermax=200, **kwargs):
    """
    Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping)
    
    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """
    
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = np.divide(G_init, np.sum(G_init))
    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)
    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)
    if log:
        res, log = ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True,numItermax=numItermax, **kwargs)
        log['fgw_dist'] = log['loss'][::-1][0]
        return res, log
    else:
        return ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC,numItermax=numItermax, **kwargs)

def uniform_gwloss(C1,C2,p,q,loss_fun='square_loss'):
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
    return ot.gromov.gwloss(constC, hC1, hC2, p[:, None] * q[None, :])
