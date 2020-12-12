recipe = {

    'advi_use': False,
    'vi_family': "diagonal",

    "full_step_search": False,
    "full_step_search_scaling": False,
    'step_size': 0.01,
    'max_iters': 100,
    'optimizer': 'adam',
    'M_iw_train': 1,
    'M_iw_sample': -1,
    'grad_estimator': "DReG",
    'per_iter_sample_budget': 100,

    'LI': False,
    'evaluation_fn': "IWELBO",
    'fix_sample_budget': False,

}
