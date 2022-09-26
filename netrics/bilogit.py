# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize
import scipy.stats

import pandas as pd
import itertools as it

# Import additional functions called by eplm()
from ipt.print_coef import print_coef
from ipt.logit import logit

# Define bilogit() function
#-----------------------------------------------------------------------------#

def bilogit(Y, R, nocons=False, silent=False, cov='DR_bc'):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu, September 2022
            (revised September 2022)
    PYTHON 3.6
    
    This function computes bipartite logit regression estimates 
        
    N = number of "customers"
    M = number of "products"
    n = NM, dumber of dyads 
    
    
    INPUTS:
    -------
    Y              :  n-length Pandas series with outcome for each
                      dyad as elements  
    R              :  n x K Pandas dataframe / regressor matrix
    nocons         :  if True do NOT append a constant to the regressor
                      matrix (default is to append a constant)
    silent         :  if True suppress optimization and estimation output
    cov            :  covariance matrix estimator
                      'jack-knife', 'dense', 'sparse' are allowable choices (see below)
    
    
    The three variance-covariance matrices are as described in Graham (2020).
    'ind' assumes independence across dyads; `DR' allows 
    for dependence across dyads sharing an indices in common. It corresponds to the 
    usual Jackknife variance estimate of the leading term in the asymptotic variance 
    expression. 'DR_bc' is a bias-corrected variance estimate.                     
    
    
    OUTPUTS:
    --------
    theta_BL        : K x 1 vector of coefficient estimates 
    vcov_theta_BL   : K x K variance-covariance matrix 
       
    FUNCTIONS CALLED : ...logit()...
    ----------------   ...print_coef()...
    """


    #-------------------------------------------------------------------#
    #- STEP 1 : ORGANIZE DATA AND PREPARE FOR ESTIMATION               -#
    #-------------------------------------------------------------------#
    
    # Check to make sure Y and R are pandas objects with multi-indices
    if not isinstance(Y.index, pd.MultiIndex):
        print("Y is not a Pandas Series with an (i,j) multi-index")
        return [None, None, None]
    
    if not isinstance(R.index, pd.MultiIndex):
        print("R is not a Pandas DataFrame with an (i,j) multi-index")
        return [None, None, None]
    
    # Get dataset dimensions and agent/dyad indices
    NM, K    = np.shape(R)          # Number of dyads (n) and regressors (K)
    i, j    = R.index.levels        # "customer" and "product" indices associated with each dyad
    N       = len(i)                # Number of customers   
    M       = len(j)                # Number of products  
    n       = N + N                 # Number of customers & products        
    idx, jdx = list(R.index.names)  # Labels for i and j indices (e.g., 'customer', 'product')
    
    # Check to see if expected number of outcomes/regressors are present. Function
    # only works with balanced datasets at the present time.
    if (NM != N*M):
        print("Number of dyadic outcomes does not equal NM")
        return [None, None, None]
    
    #-------------------------------------------------------------------#
    #- STEP 2 : COMPUTE POINT ESTIMATES BY PSEUDO COMPoSITE MLE        -#
    #-------------------------------------------------------------------#
    
    # Add constant to R if needed
    if not nocons:
        K += 1
        R.insert(0, 'constant', np.ones_like(Y))
        
    # Sort data prior to estimation (speeds up variance-covariance calculation)
    Y.sort_index(level = [idx, jdx], inplace = True) 
    R.sort_index(level = [idx, jdx], inplace = True)    
    
    # Compute point estimate of theta
    [theta_BL, _, H, S_ij, _, _] \
        = logit(Y, R, s_wgt=None, nocons=nocons, c_id=None, silent=silent, full=False)
        
    #-------------------------------------------------------------------#
    #- STEP 3 : COMPUTE VARIANCE-COVARIANCE MATRIX OF COEFFICIENTS     -#
    #-------------------------------------------------------------------#
    
    # Reshape coefficient vector into 2d array (K x 1)
    theta_BL   = theta_BL.reshape(-1,1)    
    
    # Compute inverse of the negative Hessian matrix    
    iGamma =  np.linalg.inv(-H/NM)                                 # K x K matrix
        
    # Compute estimates of....    
    s_bar2  = S_ij                   # NM x K matrix of scores
    s_bar1i = np.zeros((N,K))        # Initialize matrix for score projections (customers) 
    s_bar1j = np.zeros((M,K))        # Initialize matrix for score projections (products)                             
        
    # Form symmetrized scores for each of the k=1,...,K elements of theta using S_ij (non-symmetric "Scores")
    for k in range(0,K):
        S_k          = S_ij[:,k].reshape((N,M), order="F")          # reshape NM x 1 score into N x M matrix
        s_bar1i[:,k] = np.mean(S_k, axis=1)                         # average each row
        s_bar1j[:,k] = np.mean(S_k, axis=0)                         # average each column
        
    # Compute Sigma2 (summation over NM terms)
    Sigma23 = (s_bar2.T @ s_bar2)/NM                                # K x K
      
    # Compute Sigma1c and Sigma1p (summation over N terms)
    Sigma1c =(s_bar1i.T @ s_bar1i)/N                                # K x K
    Sigma1p =(s_bar1j.T @ s_bar1j)/M                                # K x K
      
    # Bias correct estimate of Sigma1 as in Efron/Stein 
    # Sigma1c =
    # Sigma1p =
    
    # Compute the asymptotic variance-covariance matrix 
    # according to specificed method
       
    if cov == 'jack-knife':
        # Assume independence across dyads
        vcov_theta_BL = (iGamma @ (Sigma1c/N + Sigma1p/M) @ iGamma)/NM
        
    elif cov == 'dense':
        # Dependence across dyads allowed, only leading variance term retained
        vcov_theta_BL = (iGamma @ ((M/(M-1))*(Sigma1c - (1/M)*Sigma23)/N + (N/(N-1))*(Sigma1p - (1/N)*Sigma23)/M ) @ iGamma)
        
        # Use eigendecomposition to ensure variance matrix is positive
        # definite
        [L, Q] = np.linalg.eig(vcov_theta_BL)
        if not np.all(L>0):               # check for negative eigenvalues
            L[L<0] = 0                    # remove negative eigenvalues
            L      = np.diag(L)
            iQ     = np.linalg.inv(Q)
            vcov_theta_BL = Q @ L @ iQ    # positive definite matrix
            
    else:
        # Dependence across dyads allowed, both variance terms retained
        vcov_theta_BL = (iGamma @ (Sigma1c/N + Sigma1p/M - Sigma23/NM) @ iGamma)
        
        # Use eigendecomposition to ensure variance matrix is positive
        # definite
        [L, Q] = np.linalg.eig(vcov_theta_BL)
        if not np.all(L>0):               # check for negative eigenvalues
            L[L<0] = 0                    # remove negative eigenvalues
            L      = np.diag(L)
            iQ     = np.linalg.inv(Q)
            vcov_theta_BL = Q @ L @ iQ    # positive definite matrix
    
    #-------------------------------------------------------------------#
    #- STEP 4 : DISPLAY ESTIMATION RESULTS                             -#
    #-------------------------------------------------------------------#
    
    if not silent:
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- BILOGIT REGRESSION ESTIMATION RESULTS                                                   -")
        print("-------------------------------------------------------------------------------------------")
        print("")
        print("Number of agents,           N : " + "{:>15,.0f}".format(N))
        print("Number of dyads,            n : " + "{:>15,.0f}".format(n))
        print("")
        print("")
        print("-------------------------------------------------------------------------------------------")
        print_coef(theta_BL, vcov_theta_BL, list(R.columns.values))
        if cov == 'ind':
            # Assume independence across dyads
            print("NOTE: Standard errors assume independence across dyads.")
        elif cov == 'DR':
            # Dependence across dyads allowed, only leading variance term retained
            print("NOTE: Standard errors allow for dependence across dyads with agents in common.")
            print("      (Jackknife variance estimate). ")
        else:
            # Dependence across dyads allowed, both variance terms retained
            print("NOTE: Standard errors allow for dependence across dyads with agents in common.")
            print("      (Bias-corrected variance estimate). ")
    
    # Remove constant from W if needed
    if not nocons:
        R.drop('constant', axis=1, inplace=True)
    
    return [theta_BL, vcov_theta_BL]

