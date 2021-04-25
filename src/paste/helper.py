import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-dark')

def generateDistanceMatrix(layer1, layer2):
    """
    Custom function to convert STLayer coordinate data into a euclidean distance matrix
    
    parameter: layer - STLayer object
    
    Return: D - 2D euclidean distance matrix
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
    
    Return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)


def getCoordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.
    
    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return coor


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
        
       