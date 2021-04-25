"""
    Spectral Clustering Implementation that uses sparse numpy matrices to get stuff done.
"""

import numpy
from sklearn.metrics import pairwise_distances # The X term should have shape (n_samples, n_features)
from scipy.sparse import diags
from scipy.sparse.lingalg import eigs
from scipy.sparse.csr_matrix import transpose

class SpectralClustering:
    """
        Spectral Clustering reduces dimensions of the data by finding its Laplacian, and then uses 
        the Eigenvector matrix to produce a clustering based on KMeans.
        
        Attributes
        ----------
        n_cluster : int
            Num of cluster applied to data
        init_pp : bool
            Initialization method whether to use K-Means++ or not
            (the default is True, which use K-Means++)
        max_iter : int
            Max iteration to update centroid (the default is 300)
        tolerance : float
            Minimum centroid update difference value to stop iteration (the default is 1e-4)
        seed : int
            Seed number to use in random generator (the default is None)
        centroid : list
            List of centroid values
        SSE : float
            Sum squared error score
    """
    
    def __init__(
            self,
            n_cluster: int,
            init_pp: bool = True,
            max_iter: int = 300,
            tolerance: float = 1e-4,
            seed: int = None):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.seed = seed
        self.centroid = None
        self.SSE = None
    
    def _init_centroid(self, data: numpy.ndarray):
    """Initialize centroid using random method or KMeans++

    Parameters
    ----------
    data : numpy.ndarray
        Data matrix to sample from
    """

        numpy.random.seed(self.seed)
        idx = numpy.random.choice(range(len(data)), size=(self.n_cluster))
        centroid = data[idx]
        return centroid
    
    def fit(self, data: numpy.ndarray):
        W = pairwise_distances(data, metric="euclidean") #Pairwise distances of the data, 
        # W isn't the adjacency matrix, so be warned. We'll have to run on actual data and find 
        D = np.diag(np.sum(np.array(dist), axis=1))
        L = W-D
        e,v = eigs(L)
        v = v[:self.n_cluster][:]
        V = transpose(v)
        #perform kmeans on this V matrix, return centroids.
        
        
    
    