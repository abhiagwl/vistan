hparams_dict = {
	

    "seed" : 11,
    
    ##################################################################
    #  Methods 
    ##################################################################
    # choosing a method gives your some default hyper-parameter settings
    # choose from : ['advi, 'gaussian, 'flows', 'meanfield', 'custom']
    # choosing a method will change the values for other relevant 
    # hyper-parameters; however, choosing 'custom', will leave them as is.
    "method" : "meanfield",

    ##################################################################
    #  ADVI hparams
    ##################################################################
    "advi_use": False,
    # The threshold parameters is used to diagnose convergence. If the 
    # mean or median of the relative tolerance window is below the 
    # threshold parameter, then the optimization is stopped.  
    'advi_convergence_threshold' : 0.001, 

    "advi_step_size": 0.01,
    # the range used in the original paper. 
    # Expand this range if the optimization diverges for all these choices
    "advi_adapt_step_size_range": [0.1, 0.01],
    # Whether to adapt the step-size to the problem based on ADVI heuristic or not.
    "advi_adapt_step_size": True,
    # Number of iterations when adapting the step-size. ADVI optimizes each step-size 
    # for these many iterations, looks at the final ELBO, and greedily chooses 
    # the largest step-size with highest final ELBO  
    'advi_adapt_step_size_verbose': False, 
    'advi_adapt_step_size_num_iters': 20, 

    # Number of iterations after which ELBO is calculated. This is coupled with the 
    # size of the circular buffer to detect convergence. During final vi optimization,
    # after each 'advi_callback_iteration', we evaluate ELBO, and append it to a buffer.
    # Then convergence is detected based on the mean or median relative tolerance 
    # over this circular buffer.

    'advi_callback_iteration': 2,

    ##################################################################
    #  Optimization
    ##################################################################
    # step_size = step_size_base/(step_size_scale**step_size_exp)
    # For comprehensive step-search, we optimize for different step_sizes,
    # where the different step_sizes are generated using step_size_exp_range
    "step_size_base" : 0.1,
    "step_size_scale" : 4.0,
    "step_size_exp" : 2, 
    # "step_size_exp_range" : [2], # For comprehensive step_search we suggest [0,1,2,3,4]

    "max_iters" : 20,

    #  Choices for optimizers: "adam", "advi"
    "optimizer":"adam" ,
    #  M_iw_train = 1 corresponds to regular VI -- ELBO is optimized
    #  M_iw_train > 1 corresponds to IW-training
    "M_iw_train"  : 2, 

    #  Four choices for gradient estimator type: "Total-gradient", "STL", "DReG", "closed-form-entropy"
    #  closed-form-entropy will work with distributions like Gaussians.
    #  IWAEDREG defaults to STL when M_iw_train = 1  
    "grad_estimator": "DReG", 

    # Fixing the sample budget. If the M_iw_train = 10, then we make 10 copies of 
    # IW-ELBO with M = 10 at each iteration.
    "per_iter_sample_budget":100, 

    "evaluation_fn" : "IWELBO",

    ##################################################################
    #  VI Families
    ##################################################################

    #  Choices for vi_vi_families: "rnvp", "gaussian", "diagonal"
    "vi_family" : "gaussian",

    ##################################################################
    #  RealNVP
    ##################################################################

    # of transformations : # of coupling transforms; each transforms includes two transitions
    # of hidden layers : # of hidden layers within s-t block
    # of hidden units : # of units in each hidden layer
    # params_init_scale : lower value initializes to standard normal approximately

    "rnvp_num_transformations" : 10,
    "rnvp_num_hidden_units" : 32,
    "rnvp_num_hidden_layers" : 2,
    "rnvp_params_init_scale" : 0.01,


    ##################################################################
    #  LI
    ##################################################################
    #  Do not use LI with real-NVP
    "LI" : False,
    "LI_max_iters":2000,
    "LI_epsilon":1e-6,

    
}