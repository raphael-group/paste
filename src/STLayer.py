import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from .helper import generateDistanceMatrix

class STLayer:
    """
    Class object that stores the spatial transcriptomics data of a layer.
    
    param: self.gene_exp = gene expression data
    param: self.coordinates = 2D spatial data as a numpy array
    """
    def __init__(self, gene_exp, coordinates):
        coordinates = np.array(coordinates)
        assert(gene_exp.shape[0] == coordinates.shape[0]), "Number of spots and coordinates don't match. Try again."
        self.gene_exp = pd.DataFrame(gene_exp)
        self.coordinates = coordinates
        
    def copy(self):
        """
        Returns a copy current STLayer object.
        """
        return STLayer(self.gene_exp, self.coordinates)
    
    
    def rotate_spots(self, angle):
        """
        Rotates spots counterclockwise around center-of-mass by angle given in radians.
        """
        coordinates = self.coordinates
        
        # set origin as center of mass
        center = sum(coordinates)/coordinates.shape[0]
        for i in range(coordinates.shape[0]):
            coordinates[i] = rotate(center, coordinates[i], angle)
        self.coordinates = coordinates
            
        
    def filter_genes(self, threshold = 15):
        """
        Removes columns (genes) with less than 'threshold' non-zero entries.
        """
        nonzero_col_sum = (self.gene_exp > 0).sum(axis=0)
        nonzero_col_sum.to_numpy()
        index = []
        for i in range(len(nonzero_col_sum)):
            if nonzero_col_sum[i] >= threshold:
                index.append(i)
        self.gene_exp = self.gene_exp.iloc[:, index]
    
    
    def normalize_gene_exp(self):
        """
        Normalize gene expression
        """
        df = self.gene_exp
        sums = df.sum(axis=1)
        df = df.div(sums, axis=0)
        
        # log normalize
        df = df + 1
        df = np.log(df)

        scalar = StandardScaler()
        new_df = scalar.fit_transform(df)
        new_df = pd.DataFrame(new_df)
        new_df.columns = df.columns
        new_df.index = df.index
        self.gene_exp = new_df
        
        
    def subset_genes(self, gene_list):
        self.gene_exp = self.gene_exp[gene_list]

        
    def remove_spots(self, rows):
        """
        Removes the spots from spatial transcriptomics data
        
        param: indexes - list of row indexes (NOT row names) of spots to be removed
        """
        
        # combine gene expression and spatial into one dataframe, remove spots   
        combined_df = self.gene_exp.join(pd.DataFrame(self.coordinates, index = self.gene_exp.index))
        combined_df = combined_df.drop(combined_df.index[rows])
        
        # split combined dataframe back into gene expression and spatial info
        num_gene_col = self.gene_exp.shape[1]
        self.gene_exp = combined_df.iloc[:, :num_gene_col]
        self.coordinates = combined_df.iloc[:, num_gene_col:].to_numpy()
        
        
    def plot(self, title = None, cluster_labels = []):
        """
        Visualizes distribution of spot coordinates
        """
        plt.figure(figsize= (5, 5))
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        if len(cluster_labels) == self.gene_exp.shape[0]:
            n_colors = len(np.unique(cluster_labels))
            colors = sns.color_palette("Paired", n_colors)
            ax = sns.scatterplot(x=x, y=y, hue=cluster_labels, legend="full", palette = colors, edgecolor='black')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        else:
            ax = sns.scatterplot(x=x, y=y)
        ax.set(xlabel='x', ylabel='y')
        if title == None:
            plt.title('Visualization of Tissue Layer')
        else:
            plt.title(title)
        plt.show()
        
        
    def plot_subregion(self, indexes):
        """
        Visualizes the distribution of a subset of specified spots.
        
        param: indexes- row indexes of spots of interest
        """
        plt.figure(figsize= (5, 5))
        label = np.array(['Background']* self.coordinates.shape[0])
        label[indexes] = 'Subregion'
        df = pd.DataFrame(self.coordinates, columns = ['x', 'y'])
        df['label'] = label
        ax = sns.scatterplot(x="x", y="y", hue="label", hue_order= {'Background' : 0, 'Subregion': 1}, style="label", style_order= {'Background' : 0, 'Subregion': 1}, data=df)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Visualization of Tissue Layer')
        plt.show()
        
    def to_graph(self, degree = 4):
        """
        Converts spatial coordinates into graph using networkx library.

        param: degree - number of edges per vertex

        return: 1) G - networkx graph
                2) node_dict - dictionary mapping nodes to spots
        """
        D = generateDistanceMatrix(self, self)
        # Get column indexes of the degree+1 lowest values per row
        idx = np.argsort(D.values, 1)[:, 0:degree+1]
        # Remove first column since it results in self loops
        idx = idx[:, 1:]

        G = nx.Graph()
        for r in range(len(idx)):
            for c in idx[r]:
                G.add_edge(r, c)

        node_dict = dict(zip(range(len(D.index)), D.index))
        return G, node_dict
        
# ------------------------------------ Helper Functions -------------------------------
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
