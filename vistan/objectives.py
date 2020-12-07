import functools
import autograd.numpy as np
import autograd.scipy.special as autoscipy
import autograd
import vistan.utilities as utils

def objective_utils(params, log_p, log_q, sample_q, \
                                        M_training, num_copies_training):

    samples_shape = (num_copies_training, M_training)  

    params_stopped = autograd.core.getval(params)

    z = sample_q(params, samples_shape)
    
    lp = log_p(z)
    
    lq = log_q(params, z)

    lq_stopped = log_q(params_stopped, z)

    return z, lp, lq, lq_stopped

def ELBO_cf_entropy(params, log_p, log_q, sample_q, entropy_q, \
                                                M_training, num_copies_training):

    _, lp, _, _ = objective_utils(params, log_p, log_q, sample_q, \
                                                M_training, num_copies_training)

    return np.mean(lp) + entropy_q(params)

def IWELBO(params, log_p, log_q, sample_q, M_training, num_copies_training):

    _, lp, lq, _ = objective_utils(params, log_p, log_q, sample_q,\
                                                 M_training, num_copies_training)

    return np.mean(autoscipy.logsumexp(lp - lq, -1)) - np.log(M_training)

def IWELBO_STL(params, log_p, log_q, sample_q, M_training, num_copies_training):

    _, lp, _, lq_stopped = objective_utils(params, log_p, log_q,\
                                     sample_q, M_training, num_copies_training)

    lR = lp - lq_stopped

    return np.mean(np.sum(utils.softmax_matrix(lR)*lR, -1))

def IWELBO_DREG(params, log_p, log_q, sample_q,\
                                 M_training, num_copies_training):

    _, lp, _, lq_stopped = objective_utils(params, log_p,\
                             log_q, sample_q, M_training, num_copies_training)

    lR = lp - lq_stopped

    return np.mean(np.sum((utils.softmax_matrix(lR)**2)*lR, -1))

def choose_objective_eval_fn(hyper_params):

    if hyper_params['grad_estimator_type'] == "Total-gradient":
        objective = IWELBO 

    elif hyper_params['grad_estimator_type'] == "DReG":
        objective = IWELBO_DREG  

    elif hyper_params['grad_estimator_type'] == "STL":
        objective = IWELBO_STL  

    elif hyper_params['grad_estimator_type'] == "closed-form-entropy":
        assert (hyper_params['M_training'] == 1)
        assert ("gaussian" in hyper_params['vi_family'])
        objective = ELBO_cf_entropy  
    else:
        raise ValueError

    if hyper_params['evaluation_fn'] == "IWELBO":
        evaluation_fn = IWELBO

    elif hyper_params['evaluation_fn'] == "ELBO-cfe":
        evaluation_fn = ELBO_cf_entropy()

    else: 
        raise ValueError

    return objective, evaluation_fn

def modify_objective_eval_fn(objective, evaluation_fn,\
                                             log_p, var_dist, hyper_params):

    m_objective = functools.partial(objective, 
                                log_p = log_p,
                                log_q = var_dist.log_prob, 
                                sample_q = var_dist.sample,
                                M_training = hyper_params['M_training'], 
                                num_copies_training = hyper_params['num_copies_training'])

    m_evaluation_fn = functools.partial(evaluation_fn, 
                                log_p = log_p,
                                log_q = var_dist.log_prob, 
                                sample_q = var_dist.sample,
                                M_training = hyper_params['M_training'], 
                                num_copies_training = hyper_params['num_copies_training'])

    if hyper_params['grad_estimator_type'] == "closed-form-entropy":
        m_objective = functools.partial(m_objective, entropy_q = var_dist.entropy)

    if hyper_params['evaluation_fn'] == "ELBO-cfe":
        m_objective = functools.partial(m_objective, entropy_q = var_dist.entropy)

    # Augment the objective definition to match the Autograd's optimizer template  
    return lambda params,t : -m_objective(params), m_evaluation_fn


def get_objective_eval_fn(log_p, var_dist, hyper_params):

    objective, evaluation_fn = choose_objective_eval_fn(hyper_params)
    
    return  modify_objective_eval_fn(objective, evaluation_fn, log_p, \
                                                        var_dist, hyper_params) 

