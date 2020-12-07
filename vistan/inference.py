import functools
import autograd
import autograd.numpy.random as npr

import vistan.vi_families as vi_families
import vistan.objectives as objectives
import vistan.interface as interface
import vistan.utilities as utils
# from utilities.result_helper import (save_results_parameters)


def hyperparams(**kwargs):
    hparams = {
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
        #  M_training = 1 corresponds to regular VI -- ELBO is optimized
        #  M_training > 1 corresponds to IW-training
        "M_training"  : 2, 

        #  Four choices for gradient estimator type: "Total-gradient", "STL", "DReG", "closed-form-entropy"
        #  closed-form-entropy will work with distributions like Gaussians.
        #  IWAEDREG defaults to STL when M_training = 1  
        "grad_estimator": "DReG", 

        # Fixing the sample budget. If the M_training = 10, then we make 10 copies of 
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
    
    for k in kwargs.keys():
        if k not in hparams.keys():
            raise KeyError(f"{k} is not an expected hyperparam.")
    hparams.update(kwargs)

    return hparams


def inference(code, data, model_name = None, verbose = True, hparams = hyperparams()):
    npr.seed(hparams['seed'])


    # Stan is particular about model names
    if model_name is not None: 
        model_name = model_name.replace("-", "_")

    model = interface.Model(code, data, model_name, verbose = verbose)

    utils.update_hparams(model, hparams)
    print("Printing the Hyper-param configuration file...")
    utils.print_dict(hparams)

    if hparams['advi_use'] == True:
        utils.advi_asserts(hparams)

    var_dist = vi_families.get_var_dist(hparams)
    init_params = var_dist.initial_params()

    if hparams['LI'] == 1:
        # update the initial parameters to LI params
        assert "gaussian" in hparams['vi_family'] 
        init_params = utils.get_laplaces_init (log_p = model.log_prob, \
                                    z_len = hparams['latent_dim'], \
                                    num_epochs = hparams['LI_max_iters'], \
                                    ε = hparams['LI_epsilon'],\
                                    model_name = model_name,
                                    model_code = code)

    log_p = model.log_prob
    log_q = var_dist.log_prob
    sample_q = var_dist.sample

    objective, eval_function = objectives.get_objective_eval_fn(log_p, var_dist,\
                                                                    hparams)
    objective_grad = autograd.grad(objective) 

    optimizer = utils.get_optimizer(hparams)
    callback = utils.get_callback(hparams)
    print("Starting optimization......")
    results, optimized_params = utils.optimization_handler(objective_grad = objective_grad,
                                                    eval_function = eval_function,
                                                    init_params = init_params,
                                                    optimizer = optimizer,
                                                    num_epochs = hparams['max_iters'],
                                                    step_size = utils.get_step_size(hparams),
                                                    callback = functools.partial(callback,
                                                                    model = model, 
                                                                    eval_function = eval_function),
                                                    verbose = verbose,
                                                    hparams = hparams)
    posterior = vi_families.get_posterior(model = model, var_dist = var_dist,\
                                                M_sampling = hparams['M_training'], params = optimized_params)

    return posterior, model, results