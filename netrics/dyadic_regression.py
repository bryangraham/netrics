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
from ipt.ols import ols
from ipt.logit import logit
from ipt.poisson import poisson
from ipt.iv import iv


# Define dyadic_regression() function
#-----------------------------------------------------------------------------#

def dyadic_regression(Y, R, regmodel='normal', directed=True, nocons=False, silent=False, cov='DR_bc'):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu, July 2017
            (revised September 2018)
    PYTHON 3.6
    
    This function computes dyadic regression estimates under linear, logit or poisson conditional mean
    models. The outcome may be directed or undirected. The outcome model is E[Y_ij|R_ij=r] = r'theta, 
    exp(r'theta)/[1+exp(r'theta)], or exp(r'theta) depending on whether the regression model is chosen to be
    linear, logit or poisson. A variety of standard error estimates, as described further below, are
    reported along with point estimates of theta. Function only works with balanced datasets at the present 
    time. The basic procedure is as described in the _Handbook of Econometrics_ chaper by
    Graham (forthcoming),
        
    N = number of agents
    n = N(N-1), the number of *directed* dyads, or n = 0.5N(N-1), the number of *undirected* dyads
    
    
    INPUTS:
    -------
    Y              :  n-length Pandas series with outcome for each
                      (directed) dyad as elements  
    R              :  n x K Pandas dataframe / regressor matrix
    regmodel       :  Model for E[Y_ij|R_ij]: 'normal', 'logit' or 'poisson'
    directed       :  if True then assume N*(N-1) directed outcomes present, 
                      otherwise assume 0.5*N(N-1) undirected outcomes are present
    nocons         :  if True do NOT append a constant to the regressor
                      matrix (default is to append a constant)
    silent         :  if True suppress optimization and estimation output
    cov            :  covariance matrix estimator
                      'ind', 'DR', 'DR_bc' are allowable choices (see below)
    
    
    The three variance-covariance matrices are as described in Graham (forthcoming).
    'ind' assumes independence across dyads (note this corresponds to "clustering"
    of dyads in the directed case where each dyad is observed twice); `DR' allows 
    for dependence across dyads sharing an agent in common. It corresponds to the 
    usual Jackknife variance estimate of the leading term in the asymptotic variance 
    expression. 'DR_bc' is a bias-corrected variance estimate.                     
    
    
    OUTPUTS:
    --------
    theta_DR        : K x 1 vector of coefficient estimates 
    vcov_theta_DR   : K x K variance-covariance matrix 
       
    FUNCTIONS CALLED : ...ols(), logit(), poisson()... 
    ----------------   ...print_coef()...
    """
    
    def normal_dreg(Y, R, s_wgt=None, nocons=False, silent=True):
        
        # Compute dyadic regression by normal composite mle (i.e., ols)
        [theta_DR, _, H, S_ij, _] = ols(Y, R, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=silent)
    
        return [theta_DR, H, S_ij]

    def logit_dreg(Y, R, s_wgt=None, nocons=False, silent=True):
        
        # Compute dyadic regression by logit composite mle
        [theta_DR, _, H, S_ij, _, _] \
            = logit(Y, R, s_wgt=s_wgt, nocons=nocons, c_id=None, silent=silent, full=False)
        
        return [theta_DR, H, S_ij]

    def poisson_dreg(Y, R, s_wgt=None, nocons=False, silent=True):
        
        # Compute dyadic regression by poisson composite mle
        [theta_DR, _, H, S_ij, _, _] \
            = poisson(Y, R, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=silent, full=False, phi_sv=None)
        
        return [theta_DR, H, S_ij]

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
    n, K    = np.shape(R)           # Number of directed/undirected dyads (n) and regressors (K)
    i, j    = R.index.levels        # Agent-level indices associated with each dyad
    agents  = set(i | j)            # Set of all unique agent-level indices
    N       = len(agents)           # Number of agents    
    idx, jdx = list(R.index.names)  # Labels for i and j indices (e.g., 'exporter', 'importer')
    
    # Check to see if expected number of outcomes/regressors are present. Function
    # only works with balanced datasets at the present time.
    if directed and (n != N*(N-1)):
        print("Number of directed outcomes does not equal N(N-1)")
        return [None, None, None]
        
    if not directed and (n != (N*(N-1) // 2)):
        print("Number of undirected outcomes does not equal 0.5N(N-1)")    
        return [None, None, None]
    
    #-------------------------------------------------------------------#
    #- STEP 2 : COMPUTE POINT ESTIMATES BY PSEUDO MLE                  -#
    #-------------------------------------------------------------------#
    
    # Add constant to R if needed
    if not nocons:
        K += 1
        R.insert(0, 'constant', np.ones_like(Y))
        
    # Sort data prior to estimation (speeds up variance-covariance calculation)
    Y.sort_index(level = [idx, jdx], inplace = True) 
    R.sort_index(level = [idx, jdx], inplace = True)    
    
    # Compute point estimate of theta
    regmodel_dict = {"normal": normal_dreg, "logit": logit_dreg, "poisson": poisson_dreg}
    [theta_DR, H, S_ij] = regmodel_dict[regmodel](Y, R, s_wgt=None, nocons=True, silent=silent)
        
    #-------------------------------------------------------------------#
    #- STEP 3 : COMPUTE VARIANCE-COVARIANCE MATRIX OF COEFFICIENTS     -#
    #-------------------------------------------------------------------#
    
    # Reshape coefficient vector into 2d array (K x 1)
    theta_DR   = theta_DR.reshape(-1,1)    
    
    if not directed:
        #-----------------------------------#
        #- Case 1: Undirected outcome data -#
        #-----------------------------------#
        
        # Compute inverse of the negative Hessian matrix    
        iGamma = np.linalg.inv(-H/n)                                               # K x K matrix
    
        # Compute estimates of Sigma1 and Sigma2, the covariance between score
        # contributions with, respectively, one and two indices in common. In
        # the undirected case we do not need to symmetrize the score prior
        # to the required U-statistic type variance calculations.
        
        s_bar2  = S_ij                    # n x K matrix of scores
        s_bar1  = np.zeros((N,K))         # Initialize matrix for score projections                          
        lt_ij   = np.tril_indices(N,-1)   # Indices of lower triangle of an N x N matrix 
       
        for k in range(0,K):
            S_k         = np.zeros((N,N))                          # N x N matrix
            S_k[lt_ij]  = S_ij[:,k]                                # Put in scores for theta_k in lower triangle
            s_bar1[:,k] = (N/(N-1))*np.mean(S_k + S_k.T, axis=0)   # Column sums of symmetric N x N matrix of scores
                                                                   # (Hajek projection terms)
       
        # Compute projection of Score_ij onto i and form Sigma1 estimate (dyads-to-agents)
        #k          = 0
        #for i in agents:
        #    # Average over all scores with index i present
        #    s_bar1[k,:] = np.mean(S_ij[(R.index.get_level_values(idx) == i) | \
        #                               (R.index.get_level_values(jdx) == i)], axis = 0)
        #    k +=1
          
      
        # Compute Sigma2 (summation over N choose 2 terms)
        Sigma2     = (s_bar2.T @ s_bar2)/n                                 # K x K
      
        # Compute Sigma1 (summation over N terms)
        Sigma1     = (s_bar1.T @ s_bar1)/N                                 # K x K
    
    else:
        #-----------------------------------#
        #- Case 2: Directed outcome data   -#
        #-----------------------------------#
    
        # Compute inverse of the negative Hessian matrix    
        iGamma = np.linalg.inv(-2*H/n)                                     # K x K matrix
    
        
        # Compute estimates of Sigma1 and Sigma2, the covariance between score
        # contributions with, respectively, one and two indices in common. In
        # the directed case we first need to symmetrize the score prior
        # to the required U-statistic type variance calculations.
        # Combine S_ij + S_ji to form symmetrized kernel "s_bar2"    
        
        s_bar2 = np.zeros((n // 2,K))    # Initialize matrix for symmetrized scores
        s_bar1 = np.zeros((N,K))         # Initialize matrix for score projections                          
        lt_ij  = np.tril_indices(N,-1)   # Indices of lower triangle of an N x N matrix      
        
        # Form symmetrized scores for each of the k=1,...,K elements of theta using S_ij (non-symmetric "Scores")
        for k in range(0,K):
            S_k         = S_ij[:,k].reshape(((N-1),N), order="F")          # reshape N(N-1) x 1 score into N-1 x N matrix
            S_k_ij      = np.vstack([np.zeros((N,)),np.tril(S_k, k=0)])    # pad and rotate to compute S_ij + S_ji for 
            S_k_ji      = np.vstack([np.triu(S_k, k=1), np.zeros((N,))])   # all 0.5N(N-1) dyads
            S_k_sym     = (S_k_ij + S_k_ji.T)                              # Lower triangle matrix with symmetric kernels
            s_bar2[:,k] = S_k_sym[lt_ij]                                   # 0.5N(N-1) x 1 vector of symmetric scores for theta_k
            s_bar1[:,k] = (N/(N-1))*np.mean(S_k_sym + S_k_sym.T, axis=0)   # Column sums of symmetric N x N matrix with scores
                                                                           # (Hajek projection terms)
        
        # Compute Sigma2 (summation over N choose 2 terms)
        Sigma2 = 2*(s_bar2.T @ s_bar2)/n                                   # K x K
      
        # Compute Sigma1 (summation over N terms)
        Sigma1 =   (s_bar1.T @ s_bar1)/N                                   # K x K
      
    # Bias correct estimate of Sigma1 as in Efron/Stein 
    # Sigma1 = ((N-1)/(N-2))*(Sigma1 - Sigma2/(N-1))
    
    # Compute the asymptotic variance-covariance matrix 
    # according to specificed method
       
    if cov == 'ind':
        # Assume independence across dyads
        vcov_theta_DR = (2/(N-1))*(iGamma @ Sigma2 @ iGamma)/N
    elif cov == 'DR':
        # Dependence across dyads allowed, only leading variance term retained
        vcov_theta_DR = 4*(iGamma @ Sigma1 @ iGamma)/N
    else:
        # Dependence across dyads allowed, both variance terms retained
        vcov_theta_DR = 4*(iGamma @ (Sigma1 - 0.5*Sigma2/(N-1)) @ iGamma)/N
        
        # Use eigendecomposition to ensure variance matrix is positive
        # definite
        [L, Q] = np.linalg.eig(vcov_theta_DR)
        if not np.all(L>0):               # check for negative eigenvalues
            L[L<0] = 0                    # remove negative eigenvalues
            L      = np.diag(L)
            iQ     = np.linalg.inv(Q)
            vcov_theta_DR = Q @ L @ iQ    # positive definite matrix
    
    #-------------------------------------------------------------------#
    #- STEP 4 : DISPLAY ESTIMATION RESULTS                             -#
    #-------------------------------------------------------------------#
    
    if not silent:
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- DYADIC REGRESSION ESTIMATION RESULTS                                                    -")
        print("- (" + regmodel + " regression model)                                                     -") 
        print("-------------------------------------------------------------------------------------------")
        print("")
        print("Number of agents,           N : " + "{:>15,.0f}".format(N))
        print("Number of dyads,            n : " + "{:>15,.0f}".format(n))
        print("")
        print("")
        print("-------------------------------------------------------------------------------------------")
        print_coef(theta_DR, vcov_theta_DR, list(R.columns.values))
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
    
    return [theta_DR, vcov_theta_DR]

