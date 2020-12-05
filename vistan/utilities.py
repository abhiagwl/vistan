import os
from pickle import load, dump
from time import time
from functools import (partial, reduce)
from operator import mul
from collections import deque
from traceback import print_exc 
import autograd.numpy as np 
import autograd.numpy.random as npr
from autograd import grad, value_and_grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam

########################################################################################
######################### Generic Inference Utilities ###############################
########################################################################################

S = np.log(np.exp(1) - 1)

def good_iter(i):
    a = 10 ** np.floor(np.log10(i*1.0))
    return (i%a)==0

def log_add_exp(x1, x2):
    return np.maximum(x1,x2) + np.log1p(np.exp(-np.abs(x1-x2)))

def log_sub_exp(x1, x2):
    assert (all(x1>x2))
    return x1 + np.log1p(-np.exp(x2-x1))

def pos_diag(x):
    return log_add_exp(x, 0)

def pos_tril(x):
    return np.tril(x, -1) + np.diag(pos_diag(np.diag(x)))

def inv_pos_diag(x):
    return log_sub_exp(np.diag(x), 0)

def inv_pos_tril(x):
    return np.tril(x, -1) + np.diag(inv_pos_diag(x)) 

def mul_iterable(x):
    return reduce(mul, x, 1)
    
########################################################################################
######################### Laplace's Initialization Utilities ###########################
########################################################################################

def Hessian_finite_differences(z, grad_f, ε):

    D = len(z)
    H = np.zeros((D,D))

    for d in range(D):
        z_pos = z*1.0
        z_pos[d] += ε
        z_neg = z*1.0
        z_neg[d] -= ε

        H[:,d] = (grad_f(z_pos) - grad_f(z_neg))/(2*ε)

    return H

def get_laplaces_init_params(log_p, z_len, num_epochs, ε = 1e-4):        

    # Initialize to Laplace's method 
    # Conduct MAP inference using BFGS method
    # Set μ = z_MAP
    # Use finite difference to calculate Hessian matrix
    # Set L = cholesky(inv(-Hessian(μ)))

    z_0 = npr.rand(z_len)
    val_and_grad = value_and_grad(lambda z: -log_p(z)) # using minimize to maximize

    rez = minimize(val_and_grad, z_0, \
                    method='BFGS', jac=True,\
                    options={'maxiter':num_epochs, 'disp':True})

    μ = rez.x
    H = Hessian_finite_differences(z = μ, grad_f = grad(lambda z : log_p(z)),\
                                    ε = ε)

    try :
        neg_H_inv = np.linalg.inv(-H)
        L = np.linalg.cholesky(neg_H_inv) # -H_inv = inv(-H)
        L = inv_pos_tril(L) # modification to adjust for pos_tril function 

    except:
        print ('Using noisy unit covariance...')
        L = np.tril(0.1*npr.randn(z_len, z_len), -1) + np.eye(z_len)
        L = inv_pos_tril(L) # modification to adjust for pos_tril function 

    return mu, L


def get_laplaces_init(log_p, z_len, num_epochs, ε, model_name):
    if check_laplaces_init_saved(model_name):
        return load_saved_laplaces_init(model_name)
    else:
        LI_params = get_laplaces_init_params(log_p, z_len, num_epochs, ε)
        save_laplaces_init(LI_params, model_name)
        return LI_params

def laplaces_init_dir(model_name):
    return 'data/laplaces_init/' + model_name + "/"

def laplaces_init_file(dir_name):
    return dir_name + "params.pkl"

def check_laplaces_init_saved(model_name):
    """
    Check if stored Laplaces Initialization parameters  are available.
    """    
    file_name = laplaces_init_file(laplaces_init_dir(model_name))
    if os.path.exists(file_name):
        return True
    return False

def load_saved_laplaces_init(model_name):
    file_name = laplaces_init_file(laplaces_init_dir(model_name))
    if not os.path.exists(file_name):
        raise ValueError
    return open_pickled_files(file_name)

    

def save_laplaces_init(params,  model_name):  
    dir_name = laplaces_init_dir(model_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dump_pickled_files(filename = laplaces_init_file(dir_name = dir_name),
        objects = params)


########################################################################################
######################### Optimization Utilities             ###########################
########################################################################################

def stan_model_batch_logp(samples, log_p, z_len):
    """
    A simple function to get batched sample evaluation for models.
    """
    orig_samples_shape = samples.shape
    assert(orig_samples_shape[-1] == z_len)
    lp = np.array([log_p(s) for s in samples.reshape(-1, z_len)])

    return lp.reshape(orig_samples_shape[:-1]) 




def advi_baseline_asserts(hyper_params):

    assert(hyper_params['advi_use'] == 1)
    assert(hyper_params['vi_family'] == 'gaussian')
    assert(hyper_params['M_training'] == 1)
    assert(hyper_params['LI_use'] == 0)
    assert(hyper_params['grad_estimator_type'] == 'closed-form-entropy')


def get_callback_arg_dict(hyper_params):

    if hyper_params['advi_use'] == True:

        buffer_len = np.int(max(0.01*hyper_params['num_epochs']/hyper_params['advi_callback_iteration'] , 2))
        delta_results = deque(maxlen = buffer_len)

        return {"delta_results" : delta_results}

    else:

        return {}

def run_optimization(objective_grad, init_params, step_size,\
             num_epochs, callback, optimizer):

    optimized_params = optimizer(objective_grad, 
                                init_params, 
                                step_size = step_size,
                                num_iters = num_epochs, 
                                callback = callback)

    return optimized_params

def get_adapted_step_size(objective_grad, eval_function, init_params, \
                        optimizer, num_epochs, hyper_params):

    """
    Implements the adaptive step-size scheme from the PyStan version of ADVI.
    """

    init_elbo = eval_function(params = init_params)

    best_elbo = -1.0*np.inf

    best_step_size = 0

    print("############################################ ")
    print(f"########## Initial elbo: {init_elbo}")
    print("############################################ ")

    for i, step_size in enumerate(hyper_params['advi_adapt_step_size_range']):

        results = []

        print("############################################ ")
        print(f"########## Checking the step_size: {step_size}")
        print("############################################ ")

        try: 
            optimized_params = run_optimization(objective_grad, 
                                                init_params, 
                                                step_size, 
                                                hyper_params['advi_adapt_step_size_num_iters'], 
                                                None, 
                                                advi_optimizer)

        except Exception:
            print(f"Error occured during the optimization with step-size {step_size}...")
            print(print_exc())
            print(f"Using initial parameters instead for {step_size}...")
            optimized_params = init_params

        candidate_elbo  = eval_function(optimized_params)

        if np.isnan(candidate_elbo):

            candidate_elbo = -1.0*np.inf 

        if  (candidate_elbo < best_elbo) & \
            (best_elbo > init_elbo):

            assert(best_step_size!= 0)

            print("Best step_size found, best step_size : ", best_step_size)
            print("Best step_size found, best elbo : ", best_elbo)

            return best_step_size

        else:

            if  ((i+1) < len(hyper_params['advi_adapt_step_size_range'])):

                best_elbo = candidate_elbo
                best_step_size = step_size

            else:

                if candidate_elbo > init_elbo:

                    print("Best step_size found, best step_size : ", best_step_size)
                    print("Best step_size found, best elbo : ", best_elbo)

                    return best_step_size

                else :

                    print("ELBO value diverged for all step_sizes. Update step_size range")
                    exit()




def optimization_handler(objective_grad, eval_function, init_params, optimizer,\
                            num_epochs, step_size, callback, hyper_params, 
                            advi_use = False, adapt_step_size = False, **kwargs):

    if  (adapt_step_size) & (advi_use):

        step_size = get_adapted_step_size(objective_grad, eval_function,\
                                            init_params, optimizer, num_epochs, 
                                            hyper_params)

    results = []

    t0 = time()

    optimized_params = run_optimization(objective_grad = objective_grad,
                                        init_params = init_params,
                                        num_epochs = num_epochs,
                                        step_size = step_size,
                                        callback = partial (callback, 
                                                            results = results,
                                                            t0 = t0,
                                                            **get_callback_arg_dict(hyper_params)),
                                        optimizer = optimizer)

    tn = time() - t0
    
    return results, optimized_params

def checkpoint(params, model, hyper_params, results, t0, n):

    if hyper_params['check_point_use']==1:

        if (n+1) in hyper_params['check_point_num_epochs']:

            tn = time() - t0

            save_results_parameters(hyper_params = hyper_params,
                                    params = params,
                                    model = model,
                                    uniq_name = hyper_params['uniq_name']+"_check_point_"+str(n+1),
                                    results = results,
                                    time = tn/(n+1))

def callback(params, t, g, results, model, eval_function, hyper_params, t0):

    results.append(eval_function(params))

    if good_iter(t+1):

        if np.isnan(results[-1]): 

            print("exiting optimization because nan encountered.")

            raise ValueError

        print("Iteration {} IWELBO (RUNNING AVERAGE) {}".format(t+1, np.mean(results)))
        print("Iteration {} IWELBO (CURRENT ESTIMATE) {}".format(t+1, results[-1]))

    checkpoint(params, model, hyper_params, results, t0 = t0, n = t)

def relative_difference(curr, prev):

    return np.abs((curr-prev)/prev)

def advi_callback(params, t, g, results, delta_results, model, \
                            eval_function, hyper_params, t0):

    results.append(eval_function(params))

    if (t+1)%hyper_params['advi_callback_iteration']==0:

        print(f"Iteration {t+1} log likelihood IWELBO (RUNNING AVERAGE) {np.nanmean(results)}")
        print(f"Iteration {t+1} log likelihood IWELBO (CURRENT AVERAGE) {(results[-1])}")

        if len(results) > hyper_params['advi_callback_iteration']:

            previous_elbo = results[-(hyper_params['advi_callback_iteration']+1)] 

        else: 

            previous_elbo = 0.0 

        current_elbo = results[-1]

        delta_results.append(relative_difference(previous_elbo, current_elbo))

        delta_elbo_mean = np.nanmean(delta_results)

        delta_elbo_median = np.nanmedian(delta_results)

        print(f"Iteration {t+1} Δ mean {delta_elbo_mean}")
        print(f"Iteration {t+1} Δ median {delta_elbo_median}")


        if  (   (delta_elbo_median <= hyper_params['advi_convergence_threshold'])|
                (delta_elbo_mean <= hyper_params['advi_convergence_threshold'])
            ):

            print("Converged according to ADVI metrics for Median/Mean")

            tn = time() - t0

            save_results_parameters (hyper_params = hyper_params,
                                    params = params,
                                    model = model,
                                    uniq_name = hyper_params['uniq_name'] + str("_delta_convergence_"),
                                    results = results,
                                    time = tn/(t+1))
            exit()

    checkpoint(params, model, hyper_params, results, t0 = t0, n = t)

def advi_optimizer(grad, x, callback, num_iters, step_size,\
                                 epsilon = 1e-16, tau = 1, alpha = 0.1):

    x, unflatten = flatten(x)

    s = np.zeros(len(x))

    for i in range(num_iters):

        g = flatten(grad(unflatten(x), i))[0]

        if callback: callback(unflatten(x), i, unflatten(g))

        if i==0:

            s = g**2

        else:

            s = alpha*(g**2) + (1-alpha)*s

        x = x -  (step_size / np.sqrt(i+1.))*g/ (tau + np.sqrt(s))

    return unflatten(x)

def get_step_size(hyper_params):

    if hyper_params['advi_use'] == True:

        return hyper_params['advi_step_size']

    return hyper_params['step_size']


def get_optimizer(hyper_params):

    if hyper_params['optimizer']=="adam":
      
        return adam

    elif hyper_params['optimizer']=="advi":

        return advi_optimizer

    else: 
        raise NotImplementedError


def get_callback(hyper_params):

    if hyper_params['advi_use'] == 0:
    
        return callback
    
    else: 
    
        return advi_callback


########################################################################################
######################### Generic Flow Utilities #######################################
########################################################################################


def coupling_layer_specifications(num_hidden_units, num_hidden_layers, z_len):
    """
    We specify the FNN based networks over here. A single network produce both s and t parts.
    Coupling Layer currently comprises of 2 transforms.
    """

    d_1 = np.int(z_len//2) 
    d_2 = np.int(z_len - d_1)
    
    coupling_layer_sizes = []
    coupling_layer_sizes.append([d_1] + num_hidden_layers*[num_hidden_units] + [2*d_2])
    coupling_layer_sizes.append([d_2] + num_hidden_layers*[num_hidden_units] + [2*d_1])
    return coupling_layer_sizes


########################################################################################
######################### Generic NN Utilities #########################################
########################################################################################

def relu(x):       
    return np.maximum(0, x)

def leakyrelu(x, slope = 0.01):       
    return np.maximum(0, x) + slope*np.minimum(0,x)

def tanh(x): 
    return np.tanh(x)

def softmax_matrix(x):
    assert x.ndim <=2
    y = x.T
    z = y - np.max(y, 0)
    z = np.exp(z)
    return (z/np.sum(z, 0)).T


########################################################################################
######################### Generic file Utilities #######################################
########################################################################################


def dump_pickled_files(filename, objects, protocol = None):
    with open(filename,"wb") as f:
        if protocol is None:
            dump(objects,f)
        else:
            dump(objects,f, protocol = protocol)

def open_pickled_files(filename, protocol = None):
    with open(filename,"rb") as f:
        if protocol is None:
            objects = load(f)
        else:
            objects = load(f, protocol = protocol)
        return objects


########################################################################################
######################### Generic Print Utilities #######################################
########################################################################################

def print_dict(dic):
    
    # hyper_params['step_size'] = hyper_params['step_size']/hyper_params['latent_dim']

    # using a fixed sample budget--we need to adjust the no. of training samples based on M 

    [print(f"{k} : {i}") for k, i in dic.items()];


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self, verbose):
        self.verbose = verbose
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):

        if not self.verbose:

        # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0],1)
            os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        
        # Re-assign the real stdout/stderr back to (1) and (2)
        if not self.verbose:

            os.dup2(self.save_fds[0],1)
            os.dup2(self.save_fds[1],2)
            # Close all file descriptors
            for fd in self.null_fds + self.save_fds:
                os.close(fd)
