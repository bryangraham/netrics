# Load library dependencies
import numpy as np
import scipy as sp
import itertools as it
import pandas as pd

from numba import jit, vectorize
import numexpr as ne

# This file defines a number of helpfer functions which are used by the main
# package functions.

#-----------------------------------------------------------------------------#
#- Define generate_dyad_to_tetrads_dict() function                           -#
#-----------------------------------------------------------------------------#

def generate_dyad_to_tetrads_dict(tetrad_list):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, June 2016
            Revised/updated for Python 3.6 October, 2018
    WEB:    http://bryangraham.github.io/econometrics/
    
    Each of the n = 0.5N(N-1) dyad's in the network appear in (N-2) choose 2 different tetrads. This function
    creates a Python dictionary with dyad tuples (i,j) as `keys' and a set of associated tetrad indices as
    `items'. Each set includes N-2 choose 2 (i.e., 0.5(N-2)(N-3))elements. This information is required to compute the 
    asymptotic covariance matrix for tetrad logit. It is computationally intensive to produce this dictionary.
    The function returns this dictionary.
    
    NOTE: @jit decorator only generates modest speed-ups because this function returns an object with
          no native numba type
    NOTE: for a network of size N, this dictionary needs to be created just once, not each time a model is fitted      
    
    INPUTS
    ------
    tetrad_list          : N choose 4 length list with quadruple [i, j, k, l] as elements. Here i, j, k and l index
                           individual agents.
                     
    OUTPUTS
    -------
    dyad_to_tetrads_dict : Dyad-to-tetrads mapping; dictionary with 0.5N(N-1) `keys' of the form '(i, j)', 
                           one for each dyad in the network. The items correspond to tetrad index lists 
                           (saved as sets) with 0.5(N-2)(N-3) elements in each set (since each dyad belongs to 
                           0.5(N-2)(N-3) different tetrads)
                           
    CALLED BY:             ...generate_tetrad_indices()...
    ---------
    """
    
    # initialize dictionary
    dyad_to_tetrads_dict = {} 
    
    # iterate over all tetrads in the network/graph
    for tetrad_index, [i, j, k ,l] in enumerate(tetrad_list):
        
        # iterate over each of the six dyads in a given tetrad
        for dyad in [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]:
            
            # If dyad already appears as a dictionary key, append tetrad index to list,
            # otherwise first add dyad key with an empty set for its initial entry. This is what
            # the setdefault method is doing (avoiding a more verbose if, then, else statement).
            
            dyad_to_tetrads_dict.setdefault(dyad, set()).add(tetrad_index)   
        
    return dyad_to_tetrads_dict


#-----------------------------------------------------------------------------#
#- Define generate_tetrad_indices() function                                 -#
#-----------------------------------------------------------------------------#

def generate_tetrad_indices(N, full_set=False):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, June 2016
            Revised/updated for Python 3.6 October, 2018
    WEB:    http://bryangraham.github.io/econometrics/
    
    This function executes the preliminary combinatoric calculations needed to generate the various indice 
    matrices & dictionaries needed to evaluate the tetrad logit criterion function and compute its asymptotic 
    variance-covariance matrix. The latter operation is particularly intensive computationally.
    
    INPUTS
    -------
    N                           : Number of agents in the network (or used in tetrad logit calculations)
    full_set                    : (optional) If True calculate dyad_to_tetrads_dict map (see below), otherwise
                                  only calculate tetrad_to_dyads_indices map. Set to false if using re-sampling
                                  methods to assess sampling variability (e.g., bootstrap)
    
    OUTPUTS
    -------
    tetrad_to_dyads_indices     : tetrad-to-dyads mapping; N(N-1)(N-3)(N-3)/24 x 6 numpy 2d array with indices for 
                                  each of the six dyads in each tetrad
    dyad_to_tetrads_dict        : dyad-to-tetrads mapping; dictionary with n = 0.5N(N-1) `keys', one for each
                                  dyad in the network. The items correspond to tetrad index lists (saved as python 
                                  sets) with 0.5(N-2)(N-3) elements in each set (since each dyad belongs to 
                                  0.5(N-2)(N-3) different tetrads)
                                  
    CALLS:                      ...generate_dyad_to_tetrads_dict()...
    ------
    
    CALLED BY:                  ...organize_data_tetrad_logit()...
    ----------
    """
   
    # Create list of all tetrads in sample of N agents using itertools.combinations
    tetrad_list = np.asarray(list(it.combinations(range(0,N), 4)), dtype='int')
    
    # Individual components of tetrad_list
    # i = tetrad_list[:,0]
    # j = tetrad_list[:,1]
    # k = tetrad_list[:,2]
    # l = tetrad_list[:,3]
    
    #[i, j, k, l] = [tetrad_list[:,0], tetrad_list[:,1], tetrad_list[:,2], tetrad_list[:,3]]
   
    # Construct multi-indices for each of the six dyads appearing in each tetrad
    # This shows where to find these dyads in a vectorized version of an N x N matrix
    ij = np.ravel_multi_index([tetrad_list[:,0],tetrad_list[:,1]], (N,N))
    ik = np.ravel_multi_index([tetrad_list[:,0],tetrad_list[:,2]], (N,N))
    il = np.ravel_multi_index([tetrad_list[:,0],tetrad_list[:,3]], (N,N))
    jk = np.ravel_multi_index([tetrad_list[:,1],tetrad_list[:,2]], (N,N))
    jl = np.ravel_multi_index([tetrad_list[:,1],tetrad_list[:,3]], (N,N))
    kl = np.ravel_multi_index([tetrad_list[:,2],tetrad_list[:,3]], (N,N))
         
    # Concatenate indices into an (N choose 4) x 6 matrix
    # The function returns this numpy 2d array as its first output
    tetrad_to_dyads_indices = np.column_stack((ij, ik, ik, jk, jl, kl))
    
    # Form dyad-to-tetrads dictionary if directed to do so
    if full_set:
        dyad_to_tetrads_dict = generate_dyad_to_tetrads_dict(tetrad_list)
    else:
        dyad_to_tetrads_dict = None
    
    return [tetrad_to_dyads_indices, dyad_to_tetrads_dict]


#-----------------------------------------------------------------------------#
#- Define organize_data_tetrad_logit() function                              -#
#-----------------------------------------------------------------------------#

def organize_data_tetrad_logit(D, W, dtcon=None):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, June 2016
            Revised/updated for Python 3.6 October, 2018
    WEB:    http://bryangraham.github.io/econometrics/    
    
    This function takes an N x N undirected adjacency matrix and K-list of N x N dyad covariate matrices
    and organizes them into an 6 (N choose 4) x 1 vector S and 6 (N choose 4) x K matrix W_tilde. These objects
    can then be used to compute the tetrad logit estimates of beta as described in Graham (2017, Econometrica). 
    This can be done using a standard logit solver.
    
    A dictionary of tetrad index lists, with dyads as keys and the tetrad lists as items, used to speed up 
    the calculation of the asymptotic variance-covariance matrix, is also returned.
    
    INPUTS:
    -------
    D                 : N x N undirected adjacency matrix
    W                 : List with elements consisting of N x N 2d numpy arrays of dyad-specific 
                        covariates such that W[k][i,j] gives the k-th covariate for dyad ij                   
    dtcon             : Dyad and tetrad concordance (dtcon). List with elements [tetrad_to_dyads_indices, 
                        dyad_to_tetrads_dict]. If dtcon=None, then construct it using generate_tetrad_indices() 
                        function. See header to generate_tetrad_indices() for more information.
                 
    OUTPUTS:
    --------
    S                 : 3 (N choose 4) x 1 vector with elements of -1, 0 or 1 depending on the how the
                        tetrad is wired (returned as a numpy array).
    W_tilde           : 3 (N choose 4) x K regressor matrix (returned as a numpy array).
    tetrad_frac_TL    : Fraction of tetrads that contribute to Tetrad Logit criterion function
    proj_tetrads_dict : Dictionary with dyads as keys and a numpy array of tetrad indices with non-zero 
                        contributions to each dyad's "score projection" as items.
                             
    CALLS:              ...generate_tetrad_indices()...
    ------             
    
    CALLED BY:          ...tetrad_logit()...
    ----------
    """
    
    def calc_S(D_ij, D_kl, D_ik, D_jl):
        
        """
        This function computes S(ij,kl) = D(ij)D(kl)(1-D(ik))(1-D(jl)) - (1- D(ij))(1-D(kl))D(ik)D(jl).
        It is fully vectorized and uses the numexpr module in order to speed up the compution of all
        6 (N choose 4) needed values of S.
        """

        return ne.evaluate('D_ij*D_kl*(1-D_ik)*(1-D_jl) - (1-D_ij)*(1-D_kl)*D_ik*D_jl')
        
    def calc_W_tilde(W_ij, W_kl, W_ik, W_jl):
    
        """
        This function computes W_tilde(ij,kl) = W(ij) + W(kl) - (W(ik) + W(jl)).
        It is fully vectorized and uses the numexpr module in order to speed up the compution of all
        3 (N choose 4) needed values of W_tilde.
        """
        return ne.evaluate('W_ij + W_kl - (W_ik + W_jl)')    
    
    
    # compute size of the network and dimension of regressor matrix
    N        = np.shape(D)[0]            # Number of agents in the network
    Nchoose4 = N*(N-1)*(N-2)*(N-3) // 24 # Number of tetrads in network
    K        = len(W)                    # Extract length of W, number of covariates
    D        = D.reshape((-1,))          # Reshape adjacency matrix into a 1-d array (vectorize)
    
    # check to see if user has provided the needed mapping indices...
    # ...if not compute these indices with a call to generate_tetrad_indices() 
    if dtcon is None:
        dtcon = generate_tetrad_indices(N, full_set=True)
    
    # -------------------------------------------------------------- #
    # - Constuct "outcome" vector and "regressor" matrix needed to - # 
    # - compute the tetrad-logit point estimates                   - #
    # -------------------------------------------------------------- #
     
    # ------------------------------ #    
    # - Construct "outcome" vector - #
    # ------------------------------ # 
   
    # S(ij,kl) = D(ij)D(kl)(1-D(ik))(1-D(jl)) - (1- D(ij))(1-D(kl))D(ik)D(jl)
    S1 = calc_S(D[dtcon[0][:,0]], D[dtcon[0][:,5]], D[dtcon[0][:,1]], D[dtcon[0][:,4]])
    
    # S(ij,lk) = D(ij)D(kl)(1-D(il))(1-D(jk)) - (1- D(ij))(1-D(kl))D(il)D(jk)
    S2 = calc_S(D[dtcon[0][:,0]], D[dtcon[0][:,5]], D[dtcon[0][:,2]], D[dtcon[0][:,3]])
    
    # S(ik,lj) = D(ik)D(jl)(1-D(il))(1-D(jk)) - (1- D(ik))(1-D(jl))D(il)D(jk)
    S3 = calc_S(D[dtcon[0][:,1]], D[dtcon[0][:,4]], D[dtcon[0][:,2]], D[dtcon[0][:,3]])
    
    
    # concatenate S vectors into N choose 4 x 3 2d numpy array
    S = np.column_stack((S1, S2, S3))
    
    # find set of indices for tetrads which contribute to the criterion function
    # (i.e., look for tetrads where at least one S permutations calculated above is non-zero)
    tetrads_to_keep = set(S.nonzero()[0])

    # calculate fraction of tetrads which DO contribute to the criterion function
    # NOTE: This fraction will be low in sparse networks
    tetrad_frac_TL = len(tetrads_to_keep)/Nchoose4
    
    # vectorize S (order='F' option ensures normal matrix algebra vectorization)
    # this creates a 3 (N choose 4) x 1 vector
    S = S.reshape((-1,1), order='F')
 
    # Remove tetrads that don't contribute to the criterion function from the dyad_to_tetrads_dict dictionary
    # NOTE: Removal speeds up variance-covariance matrix calculation. Do this while making any calls to
    #       dyad_to_tetrads_dict read-only in nature (as in list comprehension below)
    # NOTE: also convert index set to a numpy 1d array to support fancy indexing needed 
    #       for variance-covariance matrix computation
    
    # This list comprehension is slow since upto O(N-2 choose 2) operations are needed for each iteration
    proj_tetrads_dict = {dyad: np.asarray(list(tetrads & tetrads_to_keep), dtype='int') \
                         for dyad, tetrads in dtcon[1].items()}
    
    # We add 2 additional indices for each tetrad to account for the three dyad permutations appearing
    # in each tetrad contribution. This list comprehension is quick.
    proj_tetrads_dict = {dyad: np.ravel([tetrads + k*Nchoose4 for k in [0,1,2]]) \
                         for dyad, tetrads in proj_tetrads_dict.items()}
    
    # -------------------------------------------- #
    # - Construct conformable "regressor" matrix - #
    # -------------------------------------------- #
    
    # Recall the concordance
    # ij = dtcon[0][:,0]   # jk = dtcon[0][:,3]
    # ik = dtcon[0][:,1]   # jl = dtcon[0][:,4]
    # il = dtcon[0][:,2]   # kl = dtcon[0][:,5]
    
    # Reshape list of dyad regressor matrices into a N**2 x K 2d array
    W_vec = np.zeros((N**2,K))
    
    for k in range(0,K):
        # Vectorize kth N x N dyad-specific regressor matrix; turn into 1d numpy array
        # NOTE: order argument below not strictly needed since W[k] is symmetric
        W_vec[:,k] = W[k].reshape((-1,), order="F")
    
    # W_tilde(ij,kl)
    W_tilde_1 = calc_W_tilde(W_vec[dtcon[0][:,0],:], W_vec[dtcon[0][:,5],:], W_vec[dtcon[0][:,1],:], W_vec[dtcon[0][:,4],:])
    
    # W_tilde(ij,lk)    
    W_tilde_2 = calc_W_tilde(W_vec[dtcon[0][:,0],:], W_vec[dtcon[0][:,5],:], W_vec[dtcon[0][:,2],:], W_vec[dtcon[0][:,3],:])
    
    # W_tilde(ik,lj)
    W_tilde_3 = calc_W_tilde(W_vec[dtcon[0][:,1],:], W_vec[dtcon[0][:,4],:], W_vec[dtcon[0][:,2],:], W_vec[dtcon[0][:,3],:])
    
    
    # 3 (N choose 4) x K matrix of regressors (put in a pandas dataframe)
    W_tilde = np.row_stack((W_tilde_1, W_tilde_2, W_tilde_3))
    
    return [S, W_tilde, tetrad_frac_TL, proj_tetrads_dict]
    

#-----------------------------------------------------------------------------#
#- Define tetrad_logit_score_proj() function                                 -#
#-----------------------------------------------------------------------------#
    

def tetrad_logit_score_proj(dyad_score_components): 
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, June 2016
    Revised/updated for Python 3.6 October, 2018
    WEB:    http://bryangraham.github.io/econometrics/
    
    This function accepts a list consisting of two components. The first element of the list is a numpy array
    of (up to) (N-2)(N-3)/2 generalized residuals associated with all tetrads that make non-zero contributions to
    the tetrad logit criterion function (and to which dyad ij belongs) as elements. The second component of list 
    consists of all associated regressor values. Together these "dyad score components" can be used to compute 
    ij's contribution to the score projection described in Graham (2014, NBER). 
    
    This function is designed to work with Python's built in map() function and also to be parallelizable 
    (at least in principle). In this case it treats the (up to) n X (N-2)(N-3)/2 2d "proj_tetrads_dict" dictionary 
    produced by the organize_data_tetrad_logit() function as an iterable. Each index array is then used to
    construct the required generalized residual and covariate matrix slices, which are then passed into
    tetrad_logit_score_proj().
    
    INPUTS:
    -------
    dyad_score_components   :  List with elements [dyad_gen_residuals, dyad_W_tilde]. Both these elements are
                               numpy arrays. See function header above and the tetrad_logit() function for
                               more information.
                               
    OUTPUTS:
    --------
    proj                    :  1 x K score projection for dyad ij
               
    CALLED BY :             ...tetrad_logit()...
    ---------
    """
    
    def multiply_AB(A, B):
    
        """
        This function computes the pointwise product of A and B. It is used to compute the score projection
        more efficiently.
        """

        return ne.evaluate('A * B')    

     
    # NOTE: dyad_gen_residuals = dyad_score_components[0]    up to 6 (N - 2 choose 2) non-zero generalized residuals
    #       dyad_W_tilde       = dyad_score_components[1]    associated K vectors of dyad-specific covariates
    #       This function doesn't make these assignments explicit in order to avoid making copies of large arrays
    
    # Check to see if dyad ij belongs to any contributing/identifying tetrads. If yes, compute projection, ctherwise
    # return zero vector
    if dyad_score_components[0].size:
        # Compute ij's projection contribution by summing scores over all contributing 
        # tetrads to which ij belongs.
        #proj = (dyad_score_components[0] * dyad_score_components[1]).sum(axis=0)
        proj = multiply_AB(dyad_score_components[0], dyad_score_components[1]).sum(axis=0)
    else:
        # ij's projection contribution is a zero vector
        K    = np.shape(dyad_score_components[1])[1]
        proj = np.zeros((K,))
        
    return proj
    
#-----------------------------------------------------------------------------#
#- Define dyad_jfe_select_matrix() function                                  -#
#-----------------------------------------------------------------------------#    
    
def dyad_jfe_select_matrix(N):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, August 2016
    Revised/updated for Python 3.6 October, 2018
    WEB:    http://bryangraham.github.io/econometrics/
    
    This function creates a dictionary with (i) dyad indicies as keys and agent tuples as items and (ii) a n x N
    scipy sparse matrix with dummy variables equal to one in elements (k,i) and (k,j) if the k-th dyad consists of agents
    i and j and zeros elsewhere. This is serves as a selection matrix for the individual effects.
    
    INPUTS
    ------
    N                   : Number of agents in the network
    
    OUTPUTS
    -------
    T                   : n x N sparse matrix described in function header above
    dyad_to_agents_dict : dictionary described in function header above
    
    """
    
    # Number of dyads in the network
    n = N * (N-1) // 2
    
    # Create list of all tetrads in sample of N agents using itertools.combinations
    # Saving list as a numpy array facilitates fancy indexing
    dyad_list = np.asarray(list(it.combinations(range(0,N), 2)), dtype='int')
    
    # Intitialize sparse matrix and dictionary returnables
    T = sp.sparse.lil_matrix((n, N), dtype=np.bool)
    dyad_to_agents_dict = {} 
    
    # iterate over all dyads in the network/graph and update returnables
    for dyad_index, [i, j] in enumerate(dyad_list):
        T[dyad_index,[i, j]] = 1
        dyad_to_agents_dict[dyad_index] = (i,j)
    
    # Change T to csr_matrix format and return    
    return [T.tocsr(), dyad_to_agents_dict]    
