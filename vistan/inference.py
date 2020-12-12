import functools
import autograd
import autograd.numpy.random as npr

import vistan.vi_families as vi_families
import vistan.objectives as objectives
import vistan.interface as interface
import vistan.utilities as utils
import vistan.hyperparams as hyperparams

import pickle


def recipe(method='meanfield', *, max_iters=100):
    """
        A function to easily run default variational methods--recipes.

        Args
        ----------
        method (string):
            One of ['advi', 'fullrank', 'meanfield', 'flows',
                    'method 0', 'method 1', 'method 2', 'method 3b',
                    'method 4a', 'method 4b', 'method 4d']
            'advi':
                Runs our implementation of ADVI
            'fullrank':
                Runs a version of fullrank Gaussian VI
            'meanfield':
                Runs a version of diagonal Gaussian VI
            'flows':
                Runs a version of flow-based VI using realNVP
            'method x':
                Runs a vi method as described in
                the paper https://arxiv.org/pdf/2006.10343.pdf
        max_iters (int):
            Changes the default number of iterations in recipes.
        Returns
        -------
        'inference' function with modified hyper-parameters locked into it.
    """
    default_hparams = hyperparams.default_hparams_dict.copy()
    utils.get_recipe_hparams(method, default_hparams)
    default_hparams["max_iters"] = max_iters
    return functools.partial(inference, hparams=default_hparams)


def algorithm(**kwargs):
    """
        A function to easily run custom variational methods.

        Args
        ----------

        kwargs (dict):

            Can contain the following attributes.
            ---------------
            ADVI hyperparams
            ---------------

            advi_use (bool):
                If true, alerts the internal functions that you are running
                ADVI as reported in
                paper https://www.jmlr.org/papers/volume18/16-107/16-107.pdf.
                You can custom use different functionalities of
                ADVI implementation; however, you would want to switch
                'advi_use' to False in that case.

            advi_convergence_threshold (float):
                The threshold parameters is used to diagnose convergence.
                If the mean or median of the relative tolerance window
                is below the threshold parameter, then
                the optimization is stopped.

            advi_step_size (float):
                parameter eta as defined in the original paper.

            advi_adapt_step_size (bool):
                Default to True. If True, then it uses the
                heuristic step-size scheme in the PyStan's ADVI implementation.

            advi_adapt_step_size_range (iterable):
                The range used in the original paper was [100,10,1,0.1,0.01]

            advi_adapt_step_size_num_iters (int):
                The number of iterations to use for each step-size candidate,
                ADVI greedily chooses the maximum step-size based on the final
                ELBO value after 'advi_adapt_step_size_num_iters' iterations.

            advi_callback_iterations (int):
                # of iterations after which ELBO is evaluated. This evaluation
                is then used to monitor convergence by monitoring the mean
                and the median of relative tolerance within a window of values.
                The length of this window (circular buffer)
                is heuristically determined.

            ---------------
            Optimization
            ---------------

            step_size (float):
                Defaults to 0.1
                If full_step_search_scaling is True, overridden
                by step_size = step_size_base/(step_size_scale**step_size_exp).
                Else, step-size is used as is.

            full_step_search (bool):
                Default to False.
                If True, will optimize for each step-size parallely
                for 'max_iters' iterations, where the step-sizes
                are selected from a range. Range can be selected in two ways:
                First, follow full-step-search scheme from the paper
                https://arxiv.org/pdf/2006.10343.pdf. For this, set
                'full_step_search_scaling' to True,
                and define 'step_size_exp_range'.
                Second, user can define 'step_size_range' hyperparam and
                set the 'full_step_search_scaling' to False.

                If False, will optimize using the 'step_size' parameter,
                as allowed by the 'full_step_search_scaling'.

            full_step_search_scaling (bool):
                Default to False.
                If True, will scale the step-size using the heuristic
                used in the paper https://arxiv.org/pdf/2006.10343.pdf,
                where step_size = step_size_base/step_size_scale**step_size_exp
                If False, it will use 'step_size' as is.

            step_size_base (float):
                Defaults to 0.1.

            step_size_scale (float):
                Defaults to 4.0

            step_size_exp (int):
                Defaults to 0

            step_size_exp_range (iterable):
                Defaults to [0,1,2,3].
                For full step-search, we optimize for
                different step_sizes, where the different step_sizes are
                generated using step_size_exp_range.

            step_size_range (iterable):
                Defaults to [0.01, 0.001, 0.0001]
                Will be used in case full_step_search_scaling is set to False,
                and full_step_search is True. The aim is to allow for custom
                step-search schemes without using step-size scaling.

            max_iters (int):
                Defaults to 100.
                # of optimization iterations.
                In case of ADVI, if we choose to adapt step-size, then
                this is only used for final adapted step_size. See ADVI
                related hyperparams for control over other ADVI parameters.

            optimizer (string):
                One of ['adam', 'sgd', 'rmsprop', 'advi']
                'advi' is the custom step-size sequence
                proposed in the ADVI paper
                (see https://www.jmlr.org/papers/volume18/16-107/16-107.pdf
                page 12.)
                Currently, only 'advi' and 'adam' support early exit
                based callback functions

            M_iw_train (int):
                # of importance weights during training to optimize
                the IW-ELBO. Setting M_iw_train=1 reverts to
                naive VI (optimizer ELBO).

            grad_estimator (string):
                One of ['DReG', 'STL', 'Total-gradient', 'closed-form-entropy']
                Total gradient refers to the regular IW-ELBO gradient.
                With M_iw_train = 1 this would be same as regular VI.
                'closed-form-entropy' works when families support
                closed-form-entropy calculation ("gaussian" or "diagonal".)

            per_iter_sample_budget (int):
                Fixes the # of samples used at each iteration to calculate
                the gradient and the evaluation_fn. If per_iter_sample_budget
                 = 100 and M_iw_train = 10, then we use 10 copies of
                 IW-ELBO gradient to optimize at each iteration.

            fix_sample_budget (bool):
                Default to True.
                If False, will increase the per_iter_sample budget
                by M_iw_train times. The numer of samples drawn at each
                iteration is decided by (num_copies_training, M_iw_train).
                If False, it will set the 'num_copies_training'
                parameter to be same as 'per_iter_sample_budget'.
                If True, num_copies = per_iter_sample_budget//M_iw_train.
            --------------------
            Evaluation Function
            --------------------

            evaluation_fn (string):
                One of ['IWELBO', 'ELBO-cfe']
                When using different gradient estimators like DREG and STL,
                the objective function is a surrogate function that has the
                same gradient as IW-ELBO but is different in value.
                evaluation_fn helps you monitor the correct objective.
                If set to IW-ELBO, it uses the same M as M_iw_train
                If set to ELBO-cfe, it monitors the ELBO using the
                closed-form entropy function
            -----------
            VI Families
            -----------

            vi_family (string):
                One of ["rnvp", "gaussian", "diagonal"]

            -------
            RealNVP
            -------

            rnvp_num_transformations (int):
                # of coupling transforms; each transforms includes
                two transitions

            rnvp_num_hidden_layers (int):
                # of hidden layers within s-t block

            rnvp_num_hidden_units (int):
                # of units in each hidden layer

            params_init_scale (int):
                scaling to help initializes to standard normal approximately

            ----------------------------
            LI: Laplace's Initialization
            ----------------------------

            LI (bool):
                Default to False. If True, initializes the
                Gaussian parameters to Laplace's approximation.
                Do not use LI with real-NVP

            LI_max_iters (int):
                Default to 2000. # of iterations for MAP estimate
                used by scipy.optimize.minimize

            LI_epsilon (float):
                Default to 1e-6. A small positive used to calculate
                Hessian at MAP estimate using finite differences.



        Returns
        -------

        'inference' function with modified hyper-parameters locked into it.
    """

    hparams = hyperparams.default_hparams_dict.copy()

    for k in kwargs.keys():
        if k not in hparams.keys():
            raise KeyError(f"""{k} is not an expected hyperparam.
                Hyper-param should be one of {list(hparams.keys())}""")

    # update the arguments to override the default arguments
    hparams.update(kwargs)

    return functools.partial(inference, hparams=hparams)


def inference(
            code, data, hparams, *, model_name="default_model_name",
            verbose=False, print_hparams=False):
    """
        A function to launch variational inference on the model
        specified by code conditioned on data.

        Args
        ----------

        code (string):
            Stan code specified as a string.

        data (dict):
            A Python dictionary containing the data as required
            by the Stan code.

        hparams (dict):
            A dictionary of hyper-parameters choices.
            This dictionary defined the various hyper-parameters
            used throughout the inference scheme. Please take a look
            at the 'algorithm' function for a list of valid attributes.
            A default hyper-parameter dictionary can be found
            at 'vistan/hyperparams.py'

        model_name (string, keyword only argument):
            Default to 'default_model_name'. Alternatively, you can
            specify your own names. Stan does not allow '-' in names.
            So we internally change '-' to '_'.

        verbose (bool, keyword only argument):
            Helps in printing some suppressed model compile warnings.

        print_hparams (bool, keyword only argument):
            Helps print the hyper-parameter dictionary for easier debugging.

        Returns
        -------
        Posterior instance
            A Posterior instance with optimization results, optimizer
            variational parameters, and Model instance locked in.

    """
    npr.seed(hparams['seed'])
    if not code.isascii():
        raise ValueError(f"""Found ascii character in code.
            PyStan currently does not support non-ascii characters.
            See https://github.com/alashworth/stan-monorepo-stage1/issues/87
            for discussion. Please, remove the non-ascii
            characters from the source code.
            {code}""")

    # Stan is particular about model names
    if model_name is not None:
        model_name = model_name.replace("-", "_")

    model = interface.Model(code, data, model_name, verbose=verbose)

    utils.update_hparams(model, hparams)
    if print_hparams:
        print("printing the Hyper-param configuration file...")
        utils.print_hparams(hparams)

    var_dist = vi_families.get_var_dist(hparams)
    init_params = var_dist.initial_params()

    if hparams['LI'] is True:
        init_params = utils.get_LI_params(
                                        hparams=hparams, model=model,
                                        code=code, model_name=model_name)

    objective, eval_function = objectives.get_objective_eval_fn(
                                                    model.log_prob, var_dist,
                                                    hparams)

    objective_grad = autograd.grad(objective)
    optimizer = utils.get_optimizer(hparams)
    callback = utils.get_callback(hparams)
    results, optimized_params = utils.optimization_handler(
                                    objective_grad=objective_grad,
                                    eval_function=eval_function,
                                    init_params=init_params,
                                    step_size=utils.get_step_size(hparams),
                                    optimizer=optimizer,
                                    num_epochs=hparams['max_iters'],
                                    callback=functools.partial(
                                        callback,
                                        model=model,
                                        eval_function=eval_function),
                                    hparams=hparams)
    return vi_families.get_posterior(
                                    model=model, var_dist=var_dist,
                                    M_iw_sample=hparams['M_iw_train'],
                                    params=optimized_params,
                                    results=results)
