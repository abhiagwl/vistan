recipe = {
    'advi_use': True,
    'advi_convergence_threshold': 0.001,
    'advi_step_size': 1,
    'advi_adapt_step_size': True,
    'advi_adapt_step_size_range': [100, 10, 1, 0.1, 0.01],
    'advi_adapt_step_size_verbose': False,
    'advi_adapt_step_size_num_iters': 200,
    'advi_callback_iteration': 2,

    'vi_family': "gaussian",
    'grad_estimator': "closed-form-entropy",
    'optimizer': 'adam',
    'M_iw_train': 1,

    'LI': False,
    'per_iter_sample_budget': 100,
    'max_iters': 200,

    'evaluation_fn': 'ELBO-cfe',

    'fix_sample_budget': False,
    "full_step_search_scaling": False,


}
