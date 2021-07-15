import numpy as np
import seaborn as sns

"""
    Functions to plot slices and align spatial coordinates after obtaining a mapping from PASTE.
"""

def stack_slices_pairwise(slices, pis):
    """
    Align spatial coordinates of sequential pairwise slices.
    
    In other words, align: 
    
        slices[0] --> slices[1] --> slices[2] --> ...
    
    param: slices - list of slices (AnnData Object)
    param: pis - list of pi (pairwise_align output) between consecutive slices
    
    Return: new_layers - list of slices with aligned spatial coordinates.
    """
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."
    new_coor = []
    S1, S2  = generalized_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0])
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        x, y = generalized_procrustes_analysis(new_coor[i], slices[i+1].obsm['spatial'], pis[i])
        new_coor.append(y)
    
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    return new_slices


def stack_slices_center(center_slice, slices, pis):
    """
    Align spatial coordinates of a list of slices to a center_slice.
    
    In other words, align:
    
        slices[0] --> center_slice
        slices[1] --> center_slice
        slices[2] --> center_slice
        ...
    
    param: center_slice - inferred center slice (AnnData object)
    param: slices - list of original slices to be aligned
    param: pis - list of pi (center_align output) between center_slice and slices
    
    Return: new_center - center slice with aligned spatial coordinates.
    Return: new_layers - list of other slices with aligned spatial coordinates.
    """
    assert len(slices) == len(pis), "'slices' should have the same length 'pis'. Please double check."
    new_coor = []

    for i in range(len(slices)):
        c, y = generalized_procrustes_analysis(center_slice.obsm['spatial'], slices[i].obsm['spatial'], pis[i])
        new_coor.append(y)
    
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    
    new_center = center_slice.copy()
    new_center.obsm['spatial'] = c
    return new_center, new_slices


def generalized_procrustes_analysis(X, Y, pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers.
    
    param: X - np array of spatial coordinates (ex: sliceA.obs['spatial'])
    param: Y - np array of spatial coordinates (ex: sliceB.obs['spatial'])
    param: pi - mapping between the two layers output by PASTE

    Return: aligned spatial coordinates of X, Y
    """
    X = X - pi.sum(axis=1).dot(X) #X.mean(axis=0)
    Y = Y - pi.sum(axis=0).dot(Y) #Y.mean(axis=0)
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    return X,Y


def plot_slice(sliceX, color, ax=None, s=100):
    """
    Plots slice spatial coordinates.
    
    param: sliceX - AnnData Object of slice
    param: color - scatterplot color
    param: ax - Pre-existing axes for the plot. Otherwise, call matplotlib.pyplot.gca() internally.
    param: s - size of spots
    """
    sns.scatterplot(x = sliceX.obsm['spatial'][:,0],y = sliceX.obsm['spatial'][:,1],linewidth=0,s=s, marker=".",color=color,ax=ax)
    if ax:
        ax.invert_yaxis()
        ax.axis('off')