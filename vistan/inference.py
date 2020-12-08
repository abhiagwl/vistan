import functools
import autograd
import autograd.numpy.random as npr

import vistan.vi_families as vi_families
import vistan.objectives as objectives
import vistan.interface as interface
import vistan.utilities as utils
import vistan.hyperparams as hyperparams


def algorithm(**kwargs):
    hparams = hyperparams.hparams_dict.copy()

    for k in kwargs.keys():
        if k not in hparams.keys():
            raise KeyError(f"{k} is not an expected hyperparam.")

    # update method arguments before other updates
    if 'method'  in kwargs.keys():
        hparams['method'] = kwargs['method']
    utils.update_hparams_method(hparams)

    # update rest of the arguments to override the default method arguments.
    hparams.update(kwargs)

    return functools.partial(inference, hparams = hparams)



def inference(code, data, hparams, *, model_name = "default_model_name", verbose = False, print_hparams = False ):
    npr.seed(hparams['seed'])
    # exit()
    if not code.isascii():
        raise ValueError(f"""Found ascii character in code. 
            PyStan currently does not support non-ascii characters. 
            See https://github.com/alashworth/stan-monorepo-stage1/issues/87 for discussion.
            Please, remove the non-ascii characters from the source code.
            {code}""")

    # Stan is particular about model names
    if model_name is not None: 
        model_name = model_name.replace("-", "_")

    model = interface.Model(code, data, model_name, verbose = verbose)

    utils.update_hparams(model, hparams)
    if print_hparams:
        print("printing the Hyper-param configuration file...")
        utils.print_hparams(hparams)

    var_dist = vi_families.get_var_dist(hparams)
    init_params = var_dist.initial_params()

    if hparams['LI'] == True:
        # update the initial parameters to LI params
        assert "gaussian" in hparams['vi_family'] 
        try:
            with utils.suppress_stdout_stderr(False):
                init_params = utils.get_laplaces_init (log_p = model.log_prob, \
                                            z_len = hparams['latent_dim'], \
                                            num_epochs = hparams['LI_max_iters'], \
                                            ε = hparams['LI_epsilon'],\
                                            model_name = model_name,
                                            model_code = code)
        except:
            print("Error occurred trying to generate Laplace's Initialization parameters.")

    objective, eval_function = objectives.get_objective_eval_fn(model.log_prob, var_dist,\
                                                                    hparams)
    objective_grad = autograd.grad(objective) 
    optimizer = utils.get_optimizer(hparams)
    callback = utils.get_callback(hparams)
    results, optimized_params = utils.optimization_handler(objective_grad = objective_grad,
                                                    eval_function = eval_function,
                                                    init_params = init_params,
                                                    optimizer = optimizer,
                                                    num_epochs = hparams['max_iters'],
                                                    step_size = utils.get_step_size(hparams),
                                                    callback = functools.partial(callback,
                                                                    model = model, 
                                                                    eval_function = eval_function),
                                                    hparams = hparams)
    posterior = vi_families.get_posterior(model = model, var_dist = var_dist,\
                                            M_iw_sample = hparams['M_iw_train'], params = optimized_params, results = results)
    return posterior