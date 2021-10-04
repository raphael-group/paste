import numpy as np
import seaborn as sns

"""
    Functions to plot slices and align spatial coordinates after obtaining a mapping from PASTE.
"""

def stack_slices_pairwise(slices, pis, output_params=False):
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
    thetas = []
    translations = []
    if not output_params:
        S1, S2  = generalized_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0])
    else:
        S1, S2,theta,tX,tY  = generalized_procrustes_analysis_2D(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0],output_params=output_params)
        thetas.append(theta)
        translations.append(tX)
        translations.append(tY)
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        if not output_params:
            x, y = generalized_procrustes_analysis(new_coor[i], slices[i+1].obsm['spatial'], pis[i])
        else:
            x, y,theta,tX,tY = generalized_procrustes_analysis_2D(new_coor[i], slices[i+1].obsm['spatial'], pis[i],output_params=output_params)
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)
    
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    
    if not output_params:
        return new_slices
    else:
        return new_slices, thetas, translations


def stack_slices_center(center_slice, slices, pis, output_params=False):
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
    thetas = []
    translations = []

    for i in range(len(slices)):
        if not output_params:
            c, y = generalized_procrustes_analysis(center_slice.obsm['spatial'], slices[i].obsm['spatial'], pis[i])
        else:
            c, y,theta,tX,tY = generalized_procrustes_analysis_2D(center_slice.obsm['spatial'], slices[i].obsm['spatial'], pis[i],output_params=output_params)
            thetas.append(theta)
            translations.append(tY)
        new_coor.append(y)
    
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    
    new_center = center_slice.copy()
    new_center.obsm['spatial'] = c
    if not output_params:
        return new_center, new_slices
    else:
        return new_center, new_slices, thetas, translations


def generalized_procrustes_analysis(X, Y, pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).
    
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

def generalized_procrustes_analysis_2D(X,Y,pi,output_params=True):
    """
    Finds and applies optimal rotation between spatial coordinates of two slices in 2D and returns the rotation angle and translation.
    
    param: X - np array of spatial coordinates (ex: sliceA.obs['spatial'])
    param: Y - np array of spatial coordinates (ex: sliceB.obs['spatial'])
    param: pi - mapping between the two layers output by PASTE

    Return: aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2
    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX #X.mean(axis=0)
    Y = Y - tY #Y.mean(axis=0)
    H = Y.T.dot(pi.T.dot(X))
    M = np.array([[0,-1],[1,0]])
    theta = np.arctan(np.trace(M.dot(H))/np.trace(H))
    # print('theta',theta*180/np.pi)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    Y = R.dot(Y.T).T
    if output_params:
        return X,Y,theta,tX,tY
    else:
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