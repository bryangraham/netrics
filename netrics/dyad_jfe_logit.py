# Ensure "normal" division
from __future__ import division

# Load library dependencies
import numpy as np
import scipy as sp
import scipy.optimize

from ipt.print_coef import print_coef
from .helpers import dyad_jfe_select_matrix

# Define dyad_jfe_logit() function
#-----------------------------------------------------------------------------#

def dyad_jfe_logit(D, W, T=None, silent=False, W_names=None, beta_sv=None):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, August 2016
    
    This function computes the dyadic joint fixed effects logit estimator introduced in Graham (2017, Econometrica) 
    -- "An Econometric Model of Link Formation with Degree Heterogeneity". The implementation is as described in 
    the paper. Notation attempts to follow that used in the paper.
    
    INPUTS:
    ------
    D                 : N x N undirected adjacency matrix
    W                 : List with elements consisting of N x N 2d numpy arrays of dyad-specific 
                        covariates such that W[k][i,j] gives the k-th covariate for dyad ij
    T                 : n x N scipy sparse matrix such that T_ij'A = A_i+A_j
    silent            : If True then suppress all optimization and estimation output, show output otherwise.  
    W_names           : List of K strings giving names of the columns of W_tilde. If silent=False then use
                        these in presentation of estimation output.
    beta_sv           : K vector of parameter starting values for numerical optimization                    
                   
    OUTPUTS:
    -------
    beta_JFE          :  K x 1 vector of joint fixed effects logit point estimates of beta
    beta_JFE_BC       :  K x 1 vector of bias corrected joint fixed effects point estimates of beta
    vcov_beta_JFE     :  K x K asymptotic-variance matrix for beta_JFE 
                         NOTE: vcov_beta_JFE is already "divided by n" (just take square root of diagonal for std. errs.)
    A_JFE             :  N x 1 vector of individual-level degree heterogeneity parameter estimates
    success           :  corresponds to success component of OptimizeResult associated with Scipy's minimize function;
                         success = True if the optimizer exited successfully]
                         
    CALLS:            ...dyad_jfe_select_matrix()... (unless T is passed in by user)
    ------
    """
    
    # A_i_p_A_j and vcov_beta_JFE get updated incidentally during fixed point calculations below. 
    # Defining them as global variables saves on repeating calculations.
    
    global A_i_p_A_j
    global vcov_beta_JFE
    
    # ------------------------------------------------------- #
    # - STEP 1: Define helper functions                     - #
    # ------------------------------------------------------- #
    
    def FixedEffectsFP(At, D_plus, W_nK, beta, ij_LowTri):
        
        """
        This function returns one iterate of the fixed point iteration procedure
        described in Chatterjee, Diaconis and Sly (2011, Annals of Applied Probability)
        and further adapted in Graham (2017, Econometrica): 
        
                    At+1 = log(D+) - log(r+) with r = r(At)
        
        INPUTS:
        ------
        At        : current value of the N vector of degree heterogeneity terms (1d numpy array)
        D_plus    : network degree sequence (1d numpy array)
        W_nK      : n x K regressor matrix (2d numpy array)
        beta      : coefficient vector (1d numpy array)
        ij_LowTri : multi indices for the lower triangle of an N x N matrix
        
        OUTPUT:
        ------
        A[t+1]    : One iterate update of A vector given At
        """
        
        N        = len(D_plus)               # Number of agents in the network
        At       = np.reshape(At,(-1,1))     # Make A a 2-dimensional numpy array
        
        # Numerator of r
        r1            = np.zeros((N,N))
        r1[ij_LowTri] = np.ravel(np.exp(W_nK @ beta))
        r1            = r1 + r1.T                            # N x N matrix with zeros on the diagonal
        
        # First term in denominator of r
        At_j = np.tile(At.T, (N, 1)) - np.diag(np.ravel(At)) # N x N matrix with A_j in each element of column j
        r2  = np.exp(-At_j)                                  # N x N matrix    
    
        # Second term in denominator of r
        At_i = np.tile(At, (1, N)) - np.diag(np.ravel(At))   # N x N matrix with A_i in each element of row i
        r3  = r1 * np.exp(At_i)                              # N x N matrix    
    
        # Calculate r for current value of beta and A
        r = r1 / (r2 + r3)                                   # N x N matrix with zeros on the diagonal
    
        r_plus = np.sum(r, axis=1)                           # 1d array of row sums of r
        
        return np.log(D_plus) - np.log(r_plus)               # 1d array of values of A[t+1]; next iterate
    
    
    def dyad_jfe_logit_logl(beta, A0, D, W_nK, D_plus, ij_LowTri):
                
        """
        This function computes the concentrated log-likelihood of beta, where the concentration
        is over the N dimensional vector of individual-level heterogeneity parameters A.
        
        INPUTS:
        -------
        beta       : K x 1 vector of current values for the parameter vector (1d numpy array)
        A0         : N x 1 vector of starting values for individual effects (1d numpy array)
        D          : N x N adjacency matrix (2d numpy array)
        W_nK       : n x K matrix of regressors (2d numpy array)
        D_plus     : N x 1 degree sequence for network (1d numpy array) 
        ij_LowTri  : multi indices for the lower triangle of an N x N matrix
        
        OUTPUTS:
        --------
        foc        : K x 1 gradient vector (returned as 1d numpy array)
        
        """
        
        # Define A_i_p_A_j (i.e., the mle of the vector with elements A_i + A_j conditional on 
        # beta) to be a global variable so that dyad_jfe_logit_foc() and dyad_jfe_logit_soc() 
        # can access these values without computing an additional N-dimensional fixed point.
        
        global A_i_p_A_j
        
        try:
            # Find vector of A heterogeneity parameters for current value of beta
            A_beta = sp.optimize.fixed_point(FixedEffectsFP, A0, args=(D_plus, W_nK, beta, ij_LowTri), \
                                             xtol=1e-08, maxiter=1000, method='iteration')
            
        except Exception as e:
            
            if not silent:
                # Print error message
                print("MLE may not exist. Fixed point iteration for A(beta) failed at current value of beta.")
                print("%s" % e)
            
            # Set A_i_p_A_j to None
            A_i_p_A_j = None
            
            # Set c_logl to None
            return None
        
        # Form concentrated log-likelihood function
        beta     = np.reshape(beta,(-1,1))                                      # Make beta a 2-dimensional numpy array
        A_beta   = np.reshape(A_beta,(-1,1))                                    # Make A_beta a 2-dimensional numpy array
        
        # form n x 1 vector with elements A_i + A_j for each dyad
        A_i_p_A_j= (np.tile(A_beta, (1, N)) + np.tile(A_beta.T, (N, 1)) - \
                    2*np.diag(np.ravel(A_beta)))[ij_LowTri].reshape((-1,1))             
        
        D        = D[ij_LowTri].reshape((-1,1))                                 # n x 1 vector of links
        index    = (W_nK @ beta) + A_i_p_A_j                                    # logit indices
        
        return -(np.sum(D * index) - np.sum(np.log(1+np.exp(index))))           # Concentrated log-likelihood
    
    
    def dyad_jfe_logit_foc(beta, A0, D, W_nK, D_plus, ij_LowTri):
                        
        """
        Returns (quasi-) first derivative vector of joint fixed effects logit concentrated
        log likelihood function with respect beta. See the header for dyad_jfe_logit_logl 
        for description of parameters.
        """
                
        if A_i_p_A_j is not None: 
            # Case 1: mle of A exists conditional on current value of beta
            beta       = np.reshape(beta,(-1,1))                        # make beta 2-dimensional object
            exp_index  = np.exp((W_nK @ beta) + A_i_p_A_j)              # exponential of logit index
            D          = D[ij_LowTri].reshape((-1,1))                   # n x 1 vector of links
            foc        = -W_nK.T @ (D - exp_index/(1+exp_index))        # gradient (K x 1 vector)
            foc        = np.ravel(foc)                                  # make foc 1-dimensional numpy array
        else:
            # Case 2: mle of A DOES NOT exists conditional on current value of beta
            foc = None
        
        return foc
    
    def dyad_jfe_logit_soc(beta, A0, D, W_nK, D_plus, ij_LowTri):
        
        """
        Returns hessian matrix of joint fixed effects logit concentrated
        log likelihood function with respect beta. See the header for dyad_jfe_logit_logl 
        for description of parameters.
        """
        
        if A_i_p_A_j is not None:
            # Case 1: mle of A exists conditional on current value of beta
            beta       = np.reshape(beta,(-1,1))                               # make beta 2-dimensional object
            exp_index  = np.exp((W_nK @ beta) + A_i_p_A_j)                     # exponential of logit index
            D          = D[ij_LowTri].reshape((-1,1))                          # n x 1 vector of links
            soc        = ((exp_index/(1+exp_index)**2) * W_nK).T @ W_nK        # hessian (note use of numpy broadcasting rules)
                                                                               # K x K "matrix" (2d numpy array) 
        else:
            soc = None
            # Case 2: mle of A DOES NOT exists conditional on current value of beta
            
        return [soc]
    
    def dyad_jfe_logit_callback(beta):
        print("Value of c_logl = "   + "%.6f" % dyad_jfe_logit_logl(beta, A0, D, W_nK, D_plus, ij_LowTri) + \
              ",  2-norm of c_score = "+ "%.6f" % np.linalg.norm(dyad_jfe_logit_foc(beta, A0, D, W_nK, \
                                                                                       D_plus, ij_LowTri)))
            
    def dyad_jfe_logit_biasFP(beta, beta_JFE, A0, D, W, W_nK, T, D_plus, ij_LowTri):
        
        """
        This function computes the asymptotic bias and variance of beta_JFE. Bias correction is
        done using the iteration algorithm described in Hahn and Newey (2004, Econometrica). An
        asymptotic variance-covariance matrix is computed as a by product.
        
        
        INPUTS:
        ------
        beta       : current value of beta in fixed point iteration scheme
                     NOTE: start with beta_JFE (joint fixed effects MLE)
        beta_JFE   : joint fixed effects MLE of beta             
        D          : N x N undirected adjacency matrix
        A0         : N x 1 vector of starting values for A (1d numpy array)
        W          : List with elements consisting of N x N 2d numpy arrays of dyad-specific 
                     covariates such that W[k][i,j] gives the k-th covariate for dyad ij
        W_nK       : n x K matrix of regressors (2d numpy array)
        T          : n x N scipy sparse matrix such that T_ij'A = A_i+A_j
        D_plus     : N x 1 degree sequence for network (1d numpy array) 
        ij_LowTri  : multi indices for the lower triangle of an N x N matrix
        
        OUTPUT:
        ------
        beta_bc    : Bias-corrected estimate of beta
        
        """
        
        # Define vcov_beta_JFE as global so that covariance matrix can be computed without
        # additional fixed point calculations beyond those used for bias correction
        
        global vcov_beta_JFE
        
        # compute size of the network and dimension of regressor matrix
        K        = len(W)                    # Number of dyad-specific covariates
        N        = np.shape(D)[0]            # Number of agents in the network
        n        = N*(N-1) // 2              # Number of dyads in network  
               
        # Find MLE of individual effects by fixed point iteration
        A_JFE    = sp.optimize.fixed_point(FixedEffectsFP, A0, args=(D_plus, W_nK, beta_JFE, ij_LowTri), \
                                           xtol=1e-08, maxiter=1000, method='iteration')
        
        # Calculate (K + N) x (K + N) hessian of the full parameter vector
        theta     = np.concatenate((beta,A_JFE)).reshape((-1,1))     # Full parameter vector (2d array, (K + N) x 1 )
        R         = sp.sparse.hstack((W_nK, T), format="csr")        # Full regressor matrix (sparse matrix, n x (K+N)) 
        exp_index = np.exp(R.dot(theta))                             # exponential of logit index (2d array, n x (K+N))
        
        # Compute full (K + N) x (K + N) hessian matrix and extract three sub-blocks
        H         = -(R.T).dot(R.multiply(exp_index/(1+exp_index)**2)).toarray()
        H_bb      = H[0:K,0:K]                                         
        H_bA      = H[0:K,K:(K+N)]
        H_AA      = H[K:(K+N),K:(K+N)]
        
        # Calculate matrices used for hessian approximation
        p            = np.zeros((N,N))                               # N x N matrix of link probabilities 
        p[ij_LowTri] = np.ravel(exp_index/(1+exp_index))
        p            = p + p.T                                       
        iV_N         = np.diag(np.sum(p * (1 - p), axis = 1)**(-1))  # inverse of V_N matrix
        Q            = iV_N - np.sum(p * (1 - p), axis = None)**(-1) # Q matrix with zeros on the diagonal
               
        # Calculate asymptotic variance-covariance matrix of beta_JFE
        # NOTE: vcov_beta_JFE is defined as global so that updates of it can be accessed by in the main
        #       body of the dyad_jfe_logit() function
        
        try:
            # First try computing "exact" hessian
            info          = -(H_bb - (H_bA.dot(np.linalg.inv(H_AA)).dot(H_bA.T)))/n
        except:
            # Second use diagonal approximation of H_aa to compute approximate hessian
            info          = -(H_bb - (H_bA.dot(Q).dot(H_bA.T)))/n
            
        vcov_beta_JFE = np.linalg.inv(info)
        
        # Calculate K vector of asymptotic bias terms
        s_b_AA = -p*(1 - p)*(1 - 2*p)                                # N x N 2d numpy array
        bias = np.zeros((K,))                                        # K x 1 1d numpy array
       
        for k in range(0,K):
            bias[k]  = 0.5*np.sum(np.sum(s_b_AA * W[k], axis = 1) * np.diag(iV_N))
        
        bias = vcov_beta_JFE.dot(bias.reshape(-1,1))/n
                
        return beta_JFE - np.ravel(bias)                             # bias correct beta_JFE
    
    # ------------------------------------------------------- #
    # - STEP 2: Prepare data for estimation                 - #
    # ------------------------------------------------------- #
    
    # compute size of the network and dimension of regressor matrix
    K        = len(W)                    # Number of dyad-specific covariates
    N        = np.shape(D)[0]            # Number of agents in the network
    n        = N*(N-1) // 2              # Number of dyads in network  
    D_plus   = np.sum(D, axis = 1)       # Network degree sequence    

    # Get multi-indices for lower triangle of N x N matrix
    ij_LowTri = np.tril_indices(N, -1)
    
    # Form n x K regressor matrix
    W_nK = [W[k][ij_LowTri] for k in range(0,K)]
    W_nK = np.column_stack(W_nK)
   
    # Form n x N individual effects dummy variable matrix if needed
    if T is None:
        T = dyad_jfe_select_matrix(N)[0]
    
    # ------------------------------------------------------- #
    # - STEP 3: COMPUTE JOINT FIXED EFFECTS ESTIMATE OF BETA- #
    # ------------------------------------------------------- #
    
    # Set optimization parameters
    if silent:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # low tolerance levels, and smaller number of allowed iterations.
        # Hide iteration output.
        options_set = {'xtol': 1e-8, 'maxiter': 1000, 'disp': False}
    else:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # high tolerance levels, and larger number of allowed iterations.
        # Show iteration output.
        options_set = {'xtol': 1e-12, 'maxiter': 10000, 'disp': True}
    
    # If starting values not provided use vector of zeros
    if beta_sv is None:
        beta_sv = np.zeros((K,))
    
    # Starting values for heterogeneity parameters
    A0 = np.zeros((N,))
    
    # Initialize n x 1 vector with A_i + A_j as elements
    # NOTE: solved for by fixed point iteration to get concentrated criterion function and defined as
    #       global in order to be accessed by foc and soc evaluation without additional computation
    A_i_p_A_j = np.zeros((n,1))
    
    # Derivative check at starting values
    #grad_norm = sp.optimize.check_grad(dyad_jfe_logit_logl, dyad_jfe_logit_foc, beta_sv, A0, D, W, D_plus, \
    #                                   ij_LowTri, epsilon = 1e-12)
    # print 'Joint fixed effects derivative check (2-norm): ' + "%.8f" % grad_norm
                
    # Solve for beta_JFE
    if silent:
        beta_JFE_results = sp.optimize.minimize(dyad_jfe_logit_logl, beta_sv, args=(A0, D, W_nK, D_plus, ij_LowTri), \
                                                method='Newton-CG', jac=dyad_jfe_logit_foc, hess=dyad_jfe_logit_soc, \
                                                options=options_set)
    else:
        print("-------------------------------------------------------------------------------------------")
        print("- COMPUTE JOINT FIXED EFFECT MLEs                                                         -")
        print("-------------------------------------------------------------------------------------------")
        
        beta_JFE_results = sp.optimize.minimize(dyad_jfe_logit_logl, beta_sv, args=(A0, D, W_nK, D_plus, ij_LowTri), \
                                                method='Newton-CG', jac=dyad_jfe_logit_foc, hess=dyad_jfe_logit_soc, \
                                                callback = dyad_jfe_logit_callback, options=options_set)
    if A_i_p_A_j is not None:
        
        # MLE exists and is successfully computed
        beta_JFE = beta_JFE_results.x           # Extract estimated parameter vector
        
        # ------------------------------------------------------- #
        # - STEP 4: COMPUTE BIAS CORRECTION & COVARIANCE        - #
        # ------------------------------------------------------- #
    
        # Initialize K x K asymptotic variance
        # NOTE: the asymptotic variance is constructed as a by product of the bias correction
        #       calculations. Since these calculations involved fixed point iteration vcov_beta_JFE
        #       it defined as a global to avoid additional fixed point iterations
        vcov_beta_JFE = np.zeros((K,K))
    
        beta_JFE_BC    = sp.optimize.fixed_point(dyad_jfe_logit_biasFP, beta_JFE, \
                                                 args=(beta_JFE, A0, D, W, W_nK, T, D_plus, ij_LowTri), \
                                                 xtol=1e-12, maxiter=1000, method='del2')
        vcov_beta_JFE = vcov_beta_JFE/n   # Divide asymptotic variance by "sample size"
    
        # Compute individual-level degree heterogeneity effects, using bias corrected beta
        # for fixed point iteration
        A_JFE    = sp.optimize.fixed_point(FixedEffectsFP, A0, args=(D_plus, W_nK, beta_JFE_BC, ij_LowTri), \
                                           xtol=1e-8, maxiter=1000, method='iteration')
    
        # ------------------------------------------------------- #
        # - STEP 5: Report estimation results                   - #
        # ------------------------------------------------------- #
    
        if not silent:
            print("")
            print("-------------------------------------------------------------------------------------------")
            print("- BIAS CORRECTED JOINT FIXED EFFECTS LOGIT ESTIMATION RESULTS                             -")
            print("-------------------------------------------------------------------------------------------")
            print("")
            print("Number of agents,           N : " + "{:>15,.0f}".format(N))
            print("Number of dyads,            n : " + "{:>15,.0f}".format(n))
            print("")
            print("-------------------------------------------------------------------------------------------")
            print_coef(beta_JFE_BC, vcov_beta_JFE, W_names)
    
    else:
        
        # MLE does not exists and/or is not successfully computed
        print("MLE does not exist or was not successfully computed.")
        return [None, None, None, None, False]
    
    return [beta_JFE, beta_JFE_BC, vcov_beta_JFE, A_JFE, beta_JFE_results.success]