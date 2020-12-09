hparams_dict = {
	

    "seed" : 11,
    
    ##################################################################
    #  Methods 
    ##################################################################
    # 
    # method (string):
    #       One of ['advi, 'gaussian, 'flows', 'meanfield', 'custom']
    #       choosing a method gives your some default hyper-parameter settings
    #       choosing a method will change the values for other relevant 
    #       hyper-parameters; however, choosing 'custom', will leave them as is.
    "method" : "meanfield",

    ##################################################################
    #  ADVI hparams
    ##################################################################
    #
    # advi_use (bool):
    #       If true, alerts the internal functions that you are running 
    #       ADVI as reported in paper https://www.jmlr.org/papers/volume18/16-107/16-107.pdf.
    #       You can custom use different functionalities of ADVI implementation; however, you
    #       would want to switch 'advi_use' to False in that case. 
    # advi_convergence_threshold (float):
    #       The threshold parameters is used to diagnose convergence. If the 
    #       mean or median of the relative tolerance window is below the 
    #       threshold parameter, then the optimization is stopped.  
    # advi_step_size (float):
    #       parameter eta as defined in the original paper.
    # advi_adapt_step_size (bool):
    #       Default is True. If True, then it uses the heuristic step-size scheme in the 
    #       PyStan's ADVI implementation. 
    # advi_adapt_step_size_range (iterable):t
    #       The range used in the original paper was [100,10,1,0.1,0.01] 
    # advi_adapt_step_size_num_iters (int):
    #       The number of iterations to use for each step-size candidate, ADVI greedily chooses
    #       the maximum step-size based on the final ELBO value after 'advi_adapt_step_size_num_iters'
    #       iterations.
    # advi_callback_iterations (int): 
    #       # of iterations after which ELBO is evaluated. This evaluation is then used to monitor 
    #       convergence by monitoring the mean and the median of relative tolerance within a 
    #       window of values. The length of this window (circular buffer) is heuristically determined. 

    "advi_use": False,
    'advi_convergence_threshold' : 0.001, 
    "advi_step_size": 0.01,
    "advi_adapt_step_size_range": [0.1, 0.01],
    "advi_adapt_step_size": True,
    'advi_adapt_step_size_verbose': False, 
    'advi_adapt_step_size_num_iters': 20, 
    'advi_callback_iteration': 2,

    ##################################################################
    #  Optimization
    ##################################################################
    # step_size_base (float):
    #       Default value is 0.1. 
    #
    # step_size_scale (float):
    #       Default value is 4.0 
    #
    # step_size_exp (int):
    #       Default value is 0 
    #
    # step_size (float):
    #       Default value is 0.1 
    #       step_size = step_size_base/(step_size_scale**step_size_exp)
    #       To try custom step_sizes, use the step_size_base attribute
    #
    # step_size_exp_range (iterable):
    #       Default value is [0,1,2,3]
    #       For comprehensive step-search, we optimize for different step_sizes,
    #       where the different step_sizes are generated using step_size_exp_range.
    #       Comprehensive step-search is not supported right now.
    #
    # max_iters (int):
    #       # of optimization iterations. 
    #       In case of ADVI, if we choose to adapt step-size, then this is only used
    #       for final adapted step_size. 
    #
    # optimizer (string):
    #       One of ['adam', 'sgd', 'rmsprop', 'advi']
    #       'advi' is the custom step-size sequence proposed in the ADVI paper
    #       (see https://www.jmlr.org/papers/volume18/16-107/16-107.pdf page 12.)
    #       Currently, only 'advi' and 'adam' support early exit based on the callback function
    #
    # M_iw_train (int):
    #       # of importance weights during training to optimize the IW-ELBO.
    #       Setting M_iw_train=1 reverts to naive VI (optimizer ELBO).
    #
    # grad_estimator (string): 
    #       One of ['DReG', 'STL', 'Total-gradient', 'closed-form-entropy']
    #       Total gradient refers to the regular IW-ELBO gradient. With M_iw_train = 1
    #       this would be same as regular VI. 'closed-form-entropy' works when
    #       families support closed-form-entropy calculation ("gaussian" or "diagonal".)
    #
    # per_iter_sample_budget (int):
    #       Fixes the # of samples used at each iteration to calculate the gradient and the 
    #       evaluation_fn. If per_iter_sample_budget = 100 and M_iw_train = 10, then we
    #       use 10 copies of IW-ELBO gradient to optimize at each iteration.

    "step_size_base" : 0.1,
    "step_size_scale" : 4.0,
    "step_size_exp" : 0, 
    "max_iters" : 20,
    "optimizer":"adam" ,
    "M_iw_train"  : 2, 
    "grad_estimator": "DReG", 
    "per_iter_sample_budget":100, 

    ##################################################################
    #  Evaluation Function
    ##################################################################
    #
    # evaluation_fn (string):
    #       One of ['IWELBO', 'ELBO-cfe']
    #       When using different gradient estimators like DREG and STL, the objective 
    #       function is a surrogate function that has the same gradient as IW-ELBO
    #       but is different in value. evaluation_fn helps you monitor the correct 
    #       objective. 
    #       If set to IW-ELBO, it uses the same M as M_iw_train
    #       If set to ELBO-cfe, it monitors the ELBO using the closed-form entropy function

    "evaluation_fn" : "IWELBO",

    ##################################################################
    #  VI Families
    ##################################################################
    #
    #  "vi_family" (string): 
    #       One of ["rnvp", "gaussian", "diagonal"]

    "vi_family" : "gaussian",

    ##################################################################
    #  RealNVP
    ##################################################################
    #
    # rnvp_num_transformations (int): 
    #       # of coupling transforms; each transforms includes two transitions
    #
    # rnvp_num_hidden_layers (int): 
    #       # of hidden layers within s-t block
    #
    # rnvp_num_hidden_units (int): 
    #       # of units in each hidden layer
    #
    # params_init_scale (int): 
    #       scaling to help initializes to standard normal approximately

    "rnvp_num_transformations" : 10,
    "rnvp_num_hidden_units" : 32,
    "rnvp_num_hidden_layers" : 2,
    "rnvp_params_init_scale" : 0.01,


    ##################################################################
    #  LI: Laplace's Initialization
    ##################################################################
    #
    # LI (bool): 
    #       Default is False. If True, initializes the Gaussian parameters to 
    #       Laplace's approximation. Do not use LI with real-NVP
    #
    # LI_max_iters (int):
    #       Default is 2000. # of iterations for MAP estimate used by scipy.optimize.minimize
    #
    # LI_epsilon (float): 
    #       Default is 1e-6. A small positive used to calculate Hessian at MAP
    #       estimate using finite differences.

    "LI" : False,
    "LI_max_iters":2000,
    "LI_epsilon":1e-6,
}