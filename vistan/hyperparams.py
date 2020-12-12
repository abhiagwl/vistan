default_hparams_dict = {
    "seed": 11,


    "advi_use": False,
    'advi_convergence_threshold': 0.001,
    "advi_step_size": 0.01,
    "advi_adapt_step_size": True,
    "advi_adapt_step_size_range": [100, 10, 1, 0.1, 0.01],
    'advi_adapt_step_size_verbose': False,
    'advi_adapt_step_size_num_iters': 100,
    'advi_callback_iteration': 2,


    "step_size": 0.01,
    "full_step_search": False,
    "full_step_search_scaling": False,
    "step_size_base": 0.01,
    "step_size_scale": 4.0,
    "step_size_exp": 0,
    "step_size_exp_range": [0, 1, 2, 3, 4],
    "step_size_range": [0.01, 0.001, 0.0001],
    "max_iters": 100,
    "optimizer": "adam",
    "M_iw_train": 1,
    "M_iw_sample": -1,
    "grad_estimator": "DReG",
    "fix_sample_budget": False,
    "per_iter_sample_budget": 100,


    "evaluation_fn": "IWELBO",


    "vi_family": "gaussian",


    "rnvp_num_transformations": 10,
    "rnvp_num_hidden_units": 16,
    "rnvp_num_hidden_layers": 2,
    "rnvp_params_init_scale": 0.01,



    "LI": False,
    "LI_max_iters": 2000,
    "LI_epsilon": 1e-6,

}
