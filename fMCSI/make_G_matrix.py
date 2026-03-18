# -*- coding: utf-8 -*-
"""
Creates the sparse Toeplitz matrix that models the AR dynamics.

Written Feb 2026, DMM
"""



import numpy as np
import scipy.sparse as sps

def make_G_matrix(T, g, segment_lengths=None):
    """
    Creates the sparse Toeplitz matrix that models the AR dynamics.
    
    Parameters
    ----------
    T:               int, size of matrix (number of total timebins)
    g:               float or array-like, discrete time constants
    segment_lengths: array-like, optional (equivalent to varargin{1})
                     Lengths of segments used to break the AR dynamics.
                     
    Returns
    -------
    G:               scipy.sparse.csr_matrix, matrix of AR dynamics
    """
    
    g = np.atleast_1d(g).flatten()
    
    if len(g) == 1 and g[0] < 0:
        g = np.array([0.0])
        
    p = len(g)

    offsets = np.arange(-p, 1)
    
    vals = np.append(-np.flip(g), 1.0)
    
    G = sps.diags(vals, offsets, shape=(T, T), format='lil')
    
    if segment_lengths is not None:
        segment_lengths = np.atleast_1d(segment_lengths).flatten()
        sl = np.concatenate(([0], np.cumsum(segment_lengths)))
        
        for i in range(len(sl) - 1):

            row_idx = int(sl[i])
            col_idx = int(sl[i+1] - 1)
            

            if row_idx < T and col_idx < T:
                G[row_idx, col_idx] = 0.0

    return G.tocsr()

