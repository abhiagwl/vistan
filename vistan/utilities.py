import os
import pickle
import time
import functools
import operator
import collections
import warnings
import traceback
import hashlib
import re
import scipy
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.misc.optimizers as optim
from tqdm.autonotebook import tqdm
import contextlib
import joblib
import logging
from vistan.recipes import cook_book
logging.getLogger("pystan").propagate = False


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def warning_on_one_line(message, category, filename, lineno, file=None,
                        line=None):
    return '%s:%s\n' % (category.__name__, message)


warnings.formatwarning = warning_on_one_line

###############################################################################
# Generic Inference Utilities
###############################################################################


def update_hparams(model, hparams):
    hparams['latent_dim'] = model.zlen

    # update_hparams_method(hparams)
    if hparams['fix_sample_budget']:
        hparams['num_copies_training'] = (
                                    hparams['per_iter_sample_budget']
                                    // hparams['M_iw_train'])
    else:
        hparams['num_copies_training'] = hparams['per_iter_sample_budget']
    # If advi is not used, then we follow the the step-size scaling
    # scheme of Agrawal et al. 2020
    if hparams['full_step_search_scaling']:
        hparams['step_size'] = hparams['step_size_base'] /\
                                    (hparams['step_size_scale']
                                        ** hparams['step_size_exp'])
        hparams['step_size'] = hparams['step_size']/hparams['latent_dim']


def get_recipe_hparams(method, hparams):
    try:
        hparams.update(cook_book[method])
    except KeyError as e:
        raise (f"""Method not unsupported.
            Expected one of
            ['advi', 'fullrank', 'meanfield', 'flows',
            'method 0', 'method 1', 'method 2', 'method 3b', 'method 3a',
            'method 4a', 'method 4b', 'method 4c', method 4d'],
            but got '{method}'
            """)


S = np.log(np.exp(1) - 1)


def good_iter(i):
    a = 10 ** np.floor(np.log10(i*1.0))
    return (i % a) == 0


def log_add_exp(x1, x2):
    return np.maximum(x1, x2) + np.log1p(np.exp(-np.abs(x1-x2)))


def log_sub_exp(x1, x2):
    assert (all(x1 > x2))
    return x1 + np.log1p(-np.exp(x2-x1))


def pos_diag(x):
    assert x.ndim == 1
    return log_add_exp(x, 0)


def pos_tril(x):
    assert x.ndim == 2
    return np.tril(x, -1) + np.diag(pos_diag(np.diag(x)))


def inv_pos_diag(x):
    assert x.ndim == 1
    return log_sub_exp(x, 0)


def inv_pos_tril(x):
    assert x.ndim == 2
    return np.tril(x, -1) + np.diag(inv_pos_diag(np.diag(x)))


def mul_iterable(x):
    return functools.reduce(operator.mul, x, 1)


###############################################################################
# Laplace's Initialization Utilities
###############################################################################


def Hessian_finite_differences(z, grad_f, ε):

    D = len(z)
    H = np.zeros((D, D))

    for d in range(D):
        z_pos = z*1.0
        z_pos[d] += ε
        z_neg = z*1.0
        z_neg[d] -= ε

        H[:, d] = (grad_f(z_pos) - grad_f(z_neg))/(2*ε)

    return H


def get_laplaces_init_params(log_p, z_len, num_epochs, ε=1e-4):
    """ A function to generate Laplaces approximation

    Args:
        log_p ([function]): [An autograd differentiable function]
        z_len ([int]): # of latent dimensions
        num_epochs ([int]): # of iterations

    Returns:
        [tuple]: [MAP estimate and the cholesky factor of \
                        inverse of negative Hessian at MAP estimate]
    """

    z_0 = npr.rand(z_len)
    # using minimize to maximize
    val_and_grad = autograd.value_and_grad(lambda z: -log_p(z))

    rez = scipy.optimize.minimize(
                                    val_and_grad, z_0,
                                    method='BFGS', jac=True,
                                    options={
                                            'maxiter': num_epochs,
                                            'disp': True})

    mu = rez.x
    H = Hessian_finite_differences(
                                    z=mu,
                                    grad_f=autograd.grad(lambda z: log_p(z)),
                                    ε=ε)
    try:
        neg_H_inv = np.linalg.inv(-H)
        L = np.linalg.cholesky(neg_H_inv)  # -H_inv = inv(-H)
        # modification to adjust for pos_tril function
        L = inv_pos_tril(L)
    except Exception as e:
        print('Using noisy unit covariance...')
        L = np.tril(0.1*npr.randn(z_len, z_len), -1) + np.eye(z_len)
        # modification to adjust for pos_tril function
        L = inv_pos_tril(L)

    return mu, L


def standardize_code(text):
    # text = re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    pat = re.compile(r'\s+')
    text = pat.sub('', text)
    return text


def get_cache_fname(model_name, model_code):
    model_code_stripped = standardize_code(model_code)
    code_hash = hashlib.md5(model_code_stripped.encode('ascii')).hexdigest()
    if model_name is None:
        return code_hash
    else:
        return model_name + "_" + code_hash


def get_laplaces_init(log_p, z_len, num_epochs, ε, model_name, model_code):
    if check_laplaces_init_saved(model_name, model_code):
        return load_saved_laplaces_init(model_name, model_code)
    else:
        LI_params = get_laplaces_init_params(log_p, z_len, num_epochs, ε)
        save_laplaces_init(LI_params, model_name, model_code)
        return LI_params


def laplaces_init_dir():
    return 'data/laplaces_init/'


def laplaces_init_file(dir_name, model_name, model_code):
    return dir_name + get_cache_fname(model_name, model_code)\
                                                + "_LI_params.pkl"


def check_laplaces_init_saved(model_name, model_code):
    """
    Check if stored Laplaces Initialization parameters  are available.
    """
    file_name = laplaces_init_file(laplaces_init_dir(), model_name, model_code)
    if os.path.exists(file_name):
        return True
    return False


def load_saved_laplaces_init(model_name, model_code):
    file_name = laplaces_init_file(laplaces_init_dir(), model_name, model_code)
    if not os.path.exists(file_name):
        raise ValueError
    return open_pickled_files(file_name)


def save_laplaces_init(params,  model_name, model_code):
    dir_name = laplaces_init_dir()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dump_pickled_files(
                        filename=laplaces_init_file(
                                                    dir_name,
                                                    model_name, model_code),
                        objects=params)


############################################################################
# Optimization Utilities
############################################################################


def advi_asserts(hparams):

    assert(hparams['advi_use'] == 1)
    assert(hparams['vi_family'] == 'gaussian')
    assert(hparams['M_iw_train'] == 1)
    assert(hparams['LI'] == 0)
    assert(hparams['grad_estimator'] == 'closed-form-entropy')
    assert(hparams['optimizer'] == 'advi')


def get_callback_arg_dict(hparams):

    if hparams['advi_use'] is True:

        buffer_len = np.int(max(
                                0.01*hparams['max_iters'] /
                                hparams['advi_callback_iteration'], 2))
        delta_results = collections.deque(maxlen=buffer_len)

        return {"delta_results": delta_results, "hparams": hparams}

    else:

        return {}


def run_optimization(
                        objective_grad, init_params, step_size,
                        num_epochs, callback, optimizer):

    optimized_params = optimizer(
                                grad=objective_grad,
                                x0=init_params,
                                step_size=step_size,
                                num_iters=num_epochs,
                                callback=callback)

    return optimized_params


def get_adapted_step_size(
                            objective_grad, eval_function, init_params,
                            optimizer, num_epochs, hparams):

    """
    Implements the adaptive step-size scheme from the PyStan version of ADVI.
    """

    init_elbo = eval_function(params=init_params)
    best_elbo = -1.0*np.inf
    best_step_size = 0

    try:
        print(f" Initial elbo: {init_elbo}")
        for i, step_size in enumerate(hparams['advi_adapt_step_size_range']):

            results = []
            print(f" Checking the step_size: {step_size}")
            try:
                optimized_params = run_optimization(
                        objective_grad=objective_grad,
                        init_params=init_params,
                        step_size=step_size,
                        num_epochs=hparams['advi_adapt_step_size_num_iters'],
                        callback=None,
                        optimizer=advi_optimizer)

            except Exception:
                print(f"Error occured during the optimization\
                                with step-size {step_size}...")
                print(traceback.print_exc())
                print(f"Using initial parameters instead for {step_size}...")
                optimized_params = init_params

            candidate_elbo = eval_function(optimized_params)

            if np.isnan(candidate_elbo):

                candidate_elbo = -1.0*np.inf

            if (candidate_elbo < best_elbo) & \
               (best_elbo > init_elbo):

                assert(best_step_size != 0)

                print("Best step_size found, best step_size:", best_step_size)
                print("Best step_size found, best elbo:", best_elbo)

                return best_step_size

            else:

                if ((i+1) < len(hparams['advi_adapt_step_size_range'])):

                    best_elbo = candidate_elbo
                    best_step_size = step_size

                else:

                    if candidate_elbo > init_elbo:
                        print("Best step_size found, \
                            best step_size:", best_step_size)
                        print("Best step_size found, best elbo:", best_elbo)
                        return best_step_size
                    else:
                        raise ValueError("ELBO value diverged \
                                for all step_sizes. Update step_size range")
    except Exception as e:
        print("Error occurred during adapting step_size for ADVI")
        raise e


def optimization_handler(
                        objective_grad, eval_function,
                        init_params, optimizer,
                        num_epochs, step_size, callback, hparams):

    if(hparams['advi_adapt_step_size']) & (hparams['advi_use']):
        with suppress_stdout_stderr(hparams['advi_adapt_step_size_verbose']):
            step_size = get_adapted_step_size(
                                objective_grad=objective_grad,
                                eval_function=eval_function,
                                init_params=init_params,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                hparams=hparams)

    if hparams.get('full_step_search', False) is False:
        return get_optim_results(
                                step_size=step_size,
                                objective_grad=objective_grad,
                                init_params=init_params,
                                num_epochs=num_epochs,
                                callback=callback,
                                optimizer=optimizer,
                                hparams=hparams)

    else:
        if hparams.get('full_step_search_scaling', False) is True:
            steps = [
                        hparams['step_size_base']/(
                                hparams['latent_dim']
                                * hparams['step_size_scale']**e)
                        for e in reversed(
                                    sorted(hparams['step_size_exp_range']))]
        else:
            steps = hparams.get('step_size_range', [0.01, 0.001, 0.0001])

        results = []
        partial_func = functools.partial(
            get_optim_results,
            objective_grad=objective_grad,
            init_params=init_params,
            num_epochs=num_epochs,
            callback=callback,
            optimizer=optimizer,
            hparams=hparams)
        warnings.filterwarnings('ignore')
        with tqdm_joblib(tqdm(
                        desc="Optimizing full-step-search",
                        total=len(steps),
                        colour='green')) as progress_bar:
            results = joblib.Parallel(n_jobs=len(steps))(
                                joblib.delayed(partial_func)(s) for s in steps)
        warnings.filterwarnings('default')
        return get_full_step_search_results(results)


def get_full_step_search_results(results):
    best_mean = -np.inf
    best_result = None
    for r in results:
        (trace, _, _), _ = r
        candidate_mean = np.mean(trace)
        if candidate_mean > best_mean:
            best_mean = candidate_mean
            best_result = r
    if best_result is None:
        warnings.warn(
            "All optimization points either diverged."
            + "Returning the least step-size results."
            + "Please, model maybe poorly specified, or"
            + "the choose bigger step-size range."
            )
        warnings.warn(
            "Returning results from the smallest step-size.")
        return results[-1]
    return best_result


def get_optim_results(
                    step_size, objective_grad,
                    init_params, num_epochs,
                    callback, optimizer, hparams):
    results = []
    t0 = time.time()
    optimized_params = run_optimization(
                            objective_grad=objective_grad,
                            init_params=init_params,
                            num_epochs=num_epochs,
                            step_size=step_size,
                            callback=functools.partial(
                                            callback,
                                            results=results,
                                            **get_callback_arg_dict(hparams)),
                            optimizer=optimizer)

    tn = time.time() - t0
    return (results, tn, len(results)), optimized_params


def callback(params, t, g, results, model, eval_function):

    results.append(eval_function(params))

    if good_iter(t+1):
        if np.isnan(results[-1]):
            print("exiting optimization because nan encountered.")
            return "exit"
    return None


def relative_difference(curr, prev):
    return np.abs((curr-prev)/prev)


def advi_callback(
                params, t, g, results, delta_results, model,
                eval_function, hparams):
    results.append(eval_function(params))

    if (t+1) % hparams['advi_callback_iteration'] == 0:
        tqdm.write(f"Iteration {t+1}")
        tqdm.write(f"ELBO, running mean: {np.nanmean(results)}")
        tqdm.write(f"ELBO, current value: {results[-1]}")

        if len(results) > hparams['advi_callback_iteration']:
            previous_elbo = results[-(hparams['advi_callback_iteration']+1)]
        else:
            previous_elbo = 0.0

        current_elbo = results[-1]
        delta_results.append(relative_difference(previous_elbo, current_elbo))
        delta_elbo_mean = np.nanmean(delta_results)
        delta_elbo_median = np.nanmedian(delta_results)

        tqdm.write(f"Tolerance Δ, mean: {delta_elbo_mean}")
        tqdm.write(f"Tolerance Δ, median: {delta_elbo_median}")
        if((delta_elbo_median <= hparams['advi_convergence_threshold']) |
                (delta_elbo_mean <= hparams['advi_convergence_threshold'])):
            tqdm.write("Converged according to ADVI \
                metrics for Median/Mean")
            return "exit"
    return None


def adam(grad, x0, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    x, unflatten = autograd.misc.flatten(x0)
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    # for i in tqdm(range(num_iters), desc="Optimizing"):
    for i in range(num_iters):
        g = autograd.misc.flatten(grad(unflatten(x), i))[0]
        if callback:
            flag = callback(unflatten(x), i, unflatten(g))
            if flag == "exit":
                return unflatten(x)

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)


def advi_optimizer(
                    grad, x0, callback, num_iters, step_size,
                    epsilon=1e-16, tau=1, alpha=0.1):

    """ADVI optimizer as described
     in https://dl.acm.org/doi/pdf/10.5555/3122009.3122023.
    """
    x, unflatten = autograd.misc.flatten(x0)
    s = np.zeros(len(x))
    for i in tqdm(range(num_iters), desc="Optimizing"):
        g = autograd.misc.flatten(grad(unflatten(x), i))[0]
        if callback:
            flag = callback(unflatten(x), i, unflatten(g))
            if flag == "exit":
                return unflatten(x)
        if i == 0:
            s = g**2
        else:
            s = alpha*(g**2) + (1-alpha)*s
        x = x - (step_size / np.sqrt(i+1.))*g / (tau + np.sqrt(s))
    return unflatten(x)


def get_step_size(hparams):

    if hparams['advi_use'] is True:
        return hparams['advi_step_size']
    return hparams['step_size']


def get_LI_params(hparams, model, code, model_name):
    # update the initial parameters to LI params
    assert "gaussian" in hparams['vi_family']
    try:
        with suppress_stdout_stderr(False):
            init_params = get_laplaces_init(
                                    log_p=model.log_prob,
                                    z_len=hparams['latent_dim'],
                                    num_epochs=hparams['LI_max_iters'],
                                    ε=hparams['LI_epsilon'],
                                    model_name=model_name,
                                    model_code=code)
        return init_params
    except Exception as e:
        print("Error occurred trying to generate\
             Laplace's Initialization parameters.")


def get_optimizer(hparams):

    if hparams['optimizer'] == "adam":
        return adam
    elif hparams['optimizer'] == "advi":
        return advi_optimizer
    elif hparams['optimizer'] == 'sgd':
        warnings.warn("Using autograd's SGD. \
            Will not exit optimization if NaN occurs.")
        return optim.sgd
    elif hparams['optimizer'] == 'rmsprop':
        warnings.warn("Using autograd's RMSprop. \
            Will not exit optimization if NaN occurs.")
        return optim.rmsprop
    else:
        raise NotImplementedError


def get_callback(hparams):

    if hparams['advi_use'] == 0:
        return callback
    else:
        return advi_callback


##############################################################################
# Generic Flow Utilities
##############################################################################


def coupling_layer_specifications(num_hidden_units, num_hidden_layers, z_len):
    """
    We specify the FNN based networks over here. A single network
    produce both s and t parts.
    Coupling Layer currently comprises of 2 transforms.
    """

    d_1 = np.int(z_len//2)
    d_2 = np.int(z_len - d_1)
    coupling_layer_sizes = []
    coupling_layer_sizes.append([d_1] +
                                num_hidden_layers*[num_hidden_units] + [2*d_2])
    coupling_layer_sizes.append([d_2] +
                                num_hidden_layers*[num_hidden_units] + [2*d_1])
    return coupling_layer_sizes


##############################################################################
# Generic NN Utilities
##############################################################################

def relu(x):
    return np.maximum(0, x)


def leakyrelu(x, slope=0.01):
    return np.maximum(0, x) + slope*np.minimum(0, x)


def tanh(x):
    return np.tanh(x)


def softmax_matrix(x):
    assert x.ndim <= 2
    y = x.T
    z = y - np.max(y, 0)
    z = np.exp(z)
    return (z/np.sum(z, 0)).T


############################################################################
# Generic file Utilities
############################################################################


def dump_pickled_files(filename, objects, protocol=None):
    with open(filename, "wb") as f:
        if protocol is None:
            pickle.dump(objects, f)
        else:
            pickle.dump(objects, f, protocol=protocol)


def open_pickled_files(filename, protocol=None):
    with open(filename, "rb") as f:
        if protocol is None:
            objects = pickle.load(f)
        else:
            objects = pickle.load(f, protocol=protocol)
        return objects


############################################################################
# Generic Print Utilities
############################################################################

def print_hparams(hparams):
    # hparams['step_size'] = hparams['step_size']/hparams['latent_dim']
    # using a fixed sample budget--we need to adjust the no. of
    # training samples based on M
    common_hparams = [
                        'num_copies_training', 'per_iter_sample_budget',
                        'latent_dim', 'evaluation_fn', 'grad_estimator',
                        'vi_family', 'max_iters', 'optimizer',
                        'M_iw_train', 'method']

    # print the common relevant hparams
    for k, i in hparams.items():
        if k in common_hparams:
            print(f"{k}:{i}")

    if hparams['method'] == 'advi':
        # print the advi relevant hparams
        for k, i in hparams.items():
            if ('advi' in k):
                print(f"{k}:{i}")
    elif hparams['method'] in ['gaussian', 'flows', 'meanfield']:
        # print the other relevant hparams
        for k, i in hparams.items():
            if (k.startswith("step")):
                print(f"{k}:{i}")
            if (k.startswith("rnvp")) & (hparams['method'] == 'flows'):
                print(f"{k}:{i}")
    elif hparams['method'] == 'custom':
        # print the remaining hparams for custom
        for k, i in hparams.items():
            if k not in common_hparams:
                print(f"{k}:{i}")

    else:
        raise ValueError("Method should be one of \
            ['custom', 'advi', 'gaussian', 'rnvp', 'meanfield']")


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
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):

        if not self.verbose:
            # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):

        # Re-assign the real stdout/stderr back to (1) and (2)
        if not self.verbose:

            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            # Close all file descriptors
            for fd in self.null_fds + self.save_fds:
                os.close(fd)
