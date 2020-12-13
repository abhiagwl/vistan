recipe = {

    'advi_use': False,
    'vi_family': "rnvp",

    "full_step_search": True,
    "full_step_search_scaling": True,
    'step_size_exp': 0,
    'step_size_exp_range': [0, 1, 2, 3, 4],
    'step_size_base': 0.1,
    'step_size_scale': 4.0,
    'max_iters': 100,
    'optimizer': 'adam',
    'M_iw_train': 1,
    'M_iw_sample': 10,
    'grad_estimator': "DReG",
    'per_iter_sample_budget': 100,
    'fix_sample_budget': True,

    'LI': False,

    'evaluation_fn': "IWELBO",

    'rnvp_num_transformations':  10,
    'rnvp_num_hidden_units':  32,
    'rnvp_num_hidden_layers':  2,
    'rnvp_params_init_scale':  0.01,


}
