import warnings
import math
import numpy as np
import progressbar
import spams
import time
import tqdm
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import normalize
from sklearn.utils import check_array

class SelfRepresentation(BaseEstimator, ClusterMixin):
    """Base class for self-representation based subspace clustering.

    Parameters
    -----------
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
        If ``symmetrize``, then affinity_matrix_ is set to be
    		|representation_matrix_| + |representation_matrix_|^T.
		If ``nearest_neighbors``, then the affinity_matrix_ is the k nearest
		    neighbor graph for the rows of representation_matrix_

    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``.
    """

    def __init__(self,  affinity='symmetrize'):
        self.affinity = affinity


    def fit(self, X, y=None):
        """Compute representation matrix, then apply spectral clustering
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        self._representation_to_affinity()
        self.timer_time_ = time.time() - time_base

        return self

    def _representation_to_affinity(self):
        """Compute affinity matrix from representation matrix.
        """
        normalized_representation_matrix_ = normalize(self.representation_matrix_, 'l2')
        if self.affinity == 'symmetrize':
            self.affinity_matrix_ = 0.5 * (np.absolute(normalized_representation_matrix_) + np.absolute(normalized_representation_matrix_.T))
        elif self.affinity == 'cosine':
            # 计算余弦相似度矩阵
            normalized_representation_matrix_=normalized_representation_matrix_.toarray()
            self.affinity_matrix_ = np.matmul(normalized_representation_matrix_, normalized_representation_matrix_.T)

            # 确保值域在[0,1]区间
            self.affinity_matrix_ = np.maximum(self.affinity_matrix_, 0)
            self.affinity_matrix_ = np.nan_to_num(self.affinity_matrix_, nan=0.0)

def sparse_subspace_clustering_orthogonal_matching_pursuit(X, n_nonzero=50, thr=1.0e-6):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
    Compute self-representation matrix C by solving the following optimization problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	

    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """	
    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype = int)
    cols = np.zeros(n_samples * n_nonzero, dtype = int)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0

    for i in range(n_samples):
    # for i in range(n_samples):
        residual = X[i, :].copy()  # initialize residual
        supp = np.empty(shape=(0), dtype = int)  # initialize support
        residual_norm_thr = np.linalg.norm(X[i, :]) * thr
        for t in range(n_nonzero):  # for each iteration of OMP  
            # compute coherence between residuals and X     
            coherence = abs( np.matmul(residual, X.T) )
            coherence[i] = 0.0
            # update support
            supp = np.append(supp, np.argmax(coherence))
            # compute coefficients
            c = np.linalg.lstsq( X[supp, :].T, X[i, :].T, rcond=None)[0]
            # compute residual
            residual = X[i, :] - np.matmul(c.T, X[supp, :])
            # check termination
            if np.sum(residual **2) < residual_norm_thr:
                break

        rows[curr_pos:curr_pos + len(supp)] = i
        cols[curr_pos:curr_pos + len(supp)] = supp
        vals[curr_pos:curr_pos + len(supp)] = c
        curr_pos += len(supp)

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
					

class SparseSubspaceClusteringOMP(SelfRepresentation):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP). 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``

    References
    -----------	
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    def __init__( self,affinity='symmetrize', n_nonzero=10, thr=1.0e-6):
        self.n_nonzero = n_nonzero
        self.thr = thr
        SelfRepresentation.__init__(self,  affinity)
    
    def _self_representation(self, X):
        self.representation_matrix_ = sparse_subspace_clustering_orthogonal_matching_pursuit(X, self.n_nonzero, self.thr)

def SSCOMPOD(data,n_nonzero = 10, thr = 1e-6):
    """
    Parameters
    -----------
    data : array-like, shape (n_samples, n_features)
    Input data to be detected
    n_nonzero : int, default 10
    thr : float, default 1.0e-6
    Returns
    -------
    affinity_matrix : array-like, shape: n_samples by n_samples
    SSCOMPAnomalyScore : array-like, shape: n_samples
    """
    n_samples,n_features = data.shape
    SSCModel = SparseSubspaceClusteringOMP(
        affinity='cosine',
        n_nonzero=n_nonzero,
        thr=thr
    )
    SSCModel.fit(data)
    affinity_matrix = SSCModel.affinity_matrix_
    representation_matrix = SSCModel.representation_matrix_
    Arrary_representation_matrix = representation_matrix.toarray()

    # 计算子空间异常分数
    subspace_score = np.zeros(n_samples)
    for i in range(n_samples):
        reconstruction = np.zeros(n_features)
        for  j in range(n_samples):
            reconstruction += Arrary_representation_matrix[i,j] * data[j]
        subspace_score[i] = np.linalg.norm(reconstruction - data[i])
    # 如果为0那么就是全0，不需要再进行归一化（否则报错）
    if np.max(subspace_score) > 0:
        subspace_score = subspace_score / np.max(subspace_score)
    return subspace_score,affinity_matrix
    
    

    
