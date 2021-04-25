from .STLayer import STLayer
import numpy as np
import pandas as pd
import seaborn as sns

"""
    Functions to plot layers and align spatial coordinates after obtaining a mapping from PASTE.
"""

def stack_layers_pairwise(layers, pis):
    """
    Align spatial coordinates of sequential pairwise layers.
    
    In other words, align: 
    
        layers[0] --> layers[1] --> layers[2] --> ...
    
    param: layers - list of STLayers
    param: pis - list of pi (pairwise_align output) between consecutive layers
    
    Return: new_layers - list of STLayers with aligned spatial coordinates.
    """
    assert len(layers) == len(pis) + 1, "'layers' should have length one more than 'pis'. Please double check."
    assert len(layers) > 1, "You should have at least 2 layers."
    new_coor = []
    L1, L2 = generalized_procrustes_analysis(layers[0].coordinates, layers[1].coordinates, pis[0])
    new_coor.append(L1)
    new_coor.append(L2)
    for i in range(1, len(layers) - 1):
        x, y = generalized_procrustes_analysis(new_coor[i], layers[i+1].coordinates, pis[i])
        new_coor.append(y)
    
    new_layers = []
    for i in range(len(layers)):
        new_layers.append(STLayer(layers[i].gene_exp, new_coor[i]))
    return new_layers


def stack_layers_center(center_layer, layers, pis):
    """
    Align spatial coordinates of a list of layers to a center_layer.
    
    In other words, align:
    
        layers[0] --> center_layer
        layers[1] --> center_layer
        layers[2] --> center_layer
        ...
    
    param: center_layer - center STLayer
    param: layers - list of STLayers
    param: pis - list of pi (center_align output) between center_layer and layers
    
    Return: new_center - center STLayer with aligned spatial coordinates.
    Return: new_layers - list of STLayers with aligned spatial coordinates.
    """
    assert len(layers) == len(pis), "'layers' should have the same length 'pis'. Please double check."
    new_coor = []

    for i in range(len(layers)):
        c, y = generalized_procrustes_analysis(center_layer.coordinates, layers[i].coordinates, pis[i])
        new_coor.append(y)
    
    new_layers = []
    for i in range(len(layers)):
        new_layers.append(STLayer(layers[i].gene_exp, new_coor[i]))
    
    new_center = STLayer(pd.DataFrame(center_layer.gene_exp, columns = center_layer.gene_exp.columns), c)
    return new_center, new_layers


def generalized_procrustes_analysis(X,Y,pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers.
    
    param: X - np array of spatial coordinates (ex: STLayer.coordinates)
    param: Y - np array of spatial coordinates (ex: STLayer.coordinates)
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


def plot_layer(layer, color, ax=None, s=100):
    """
    Plots STLayer spatial coordinates.
    
    param: layer - STLayer
    param: color - scatterplot color
    param: ax - Pre-existing axes for the plot. Otherwise, call matplotlib.pyplot.gca() internally.
    param: s - size of spots
    """
    sns.scatterplot(x = layer.coordinates[:,0],y = layer.coordinates[:,1],linewidth=0,s=s, marker=".",color=color,ax=ax)
    if ax:
        ax.invert_yaxis()
        ax.axis('off')

    