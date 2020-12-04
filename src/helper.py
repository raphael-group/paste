import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-dark')

def generateDistanceMatrix(layer1, layer2):
    """
    Custom function to convert spatial transcriptomics coordinate data into a euclidean distance matrix
    
    parameter: layer - Layer object
    return: D - 2D euclidean distance matrix
    """
    spots1 = layer1.gene_exp.index
    spots2 = layer2.gene_exp.index
    D = distance_matrix(layer1.coordinates, layer2.coordinates)
    D = pd.DataFrame(D, index = spots1, columns = spots2)
    return D


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
    D = []
    for row in X:
        log = np.log(row*(1./Y))
        D.append(np.dot(row, log.T))
    return np.asarray(D)



def intersect(a, b): 
    """
    param: a - list
    param: b - list
    return: list of common elements
    """
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return list(a_set & b_set) 
    else: 
        print("No common elements")
        
       