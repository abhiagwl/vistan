import pystan
import warnings
import hashlib
import pickle
import re
import os
import collections
import scipy
import warnings
import traceback
import functools
import autograd
import autograd.numpy as np
import vistan.utilities as utils

###########################################################################
#  stuff related to loading stan models and data
###########################################################################

def is_good_model(code, data, model_name = None):
    # TODO : make sure data is deterministic

    try:

        model = Model(code, data, model_name)
        Z_nuts = model.sampling(iter=20)
        Z_mf   = model.mf(iter=100)
        Z_advi = model.advi(iter=100)

        for Z in [Z_mf,Z_advi]:
            if Z.shape[1]!=Z_nuts.shape[1]:
                raise ValueError("NUTS and VI not same dimensions")

        # also check that optimization works
        z,rez  = model.argmax(True,method='BFGS',gtol=1e-3,maxiter=100) 

        if not rez.success:
            print(rez)
            raise ValueError("optimization failed")
        if len(z) != Z_nuts.shape[1]:
            raise ValueError("NUTS and argmax not same dimensions")

        return True

    except Exception:
        print(f"Error during model check...")
        print(traceback.print_exc())

        return False

###########################################################################
#  stuff related to compiling and caching stan models
###########################################################################

# removes whitespace and such to prevent extra compiling
# this is only used for hashing (original code is compiled)

def standardize_code(text):
    # text = re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    pat = re.compile(r'\s+')
    text = pat.sub('', text)
    return text

# import Cython
def get_compiled_model(model_code, model_name=None, verbose = False, **kwargs):
    """Use just as you would `stan` from pystan"""
    if not os.path.exists('data/cached-models'):
        print('creating cached-models/ to save compiled stan models')
        os.makedirs('data/cached-models')

    model_code_stripped = standardize_code(model_code)
    code_hash = hashlib.md5(model_code_stripped.encode('ascii')).hexdigest()


    if model_name is None:
        cache_fn = 'data/cached-models/{}.pkl'.format(code_hash)
    else:
        cache_fn = 'data/cached-models/{}-{}.pkl'.format(model_name, code_hash)

    if os.path.isfile(cache_fn):
        with open(cache_fn, 'rb') as f:
            sm = pickle.load(f)
    else:
        try:
            print("Cached model not found. Recompiling...")            
            with utils.suppress_stdout_stderr(verbose = verbose):
                sm = pystan.StanModel(model_code=model_code, model_name=model_name, **kwargs)
        except:
            print("Error during compilation...")
            print('Could not compile code for ' + model_code, ". Trying turning verbose for better debugging.")
            raise
        print("Caching model...")
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)        
    return sm

###########################################################################
#  here's the object provided to the user-- a logp function that hooks into autograd
###########################################################################

class Model:
    def __init__(self, model_code, data, model_name=None, verbose = True, debugging_verbose = False):
        extra_compile_args = ['-O1','-w','-Wno-deprecated'] # THIS IS SLOW! TURN OPTIMIZATION ON SOMEDAY!

        self.data = data
        self.model_name = model_name
        self.model_code = model_code

        self.sm = get_compiled_model(model_code=self.model_code, extra_compile_args=extra_compile_args,\
                model_name=self.model_name, verbose = verbose)
        try:
            with utils.suppress_stdout_stderr(verbose = debugging_verbose):
                self.fit = self.sm.sampling(data=self.data, iter=100, chains=1, init=0)
        except:
            print('Could not init model for ' + model_code)
            print('Try turning debugging_verbose on for better debugging.')
            raise 

        self.keys = self.fit.unconstrained_param_names()
        self.zlen = len(self.keys)

    def z0(self):
        return np.random.randn(self.zlen)

    def sampling(self,**kwargs):
        try: 
            assert kwargs.get('iter',2) >=2
            warnings.simplefilter('ignore')
            self.fit = self.sm.sampling(data=self.data,**kwargs)
            rez = self.unconstrain(self.fit.extract())
            warnings.simplefilter('default')
        except:
            raise
        return rez

    # get posterior max via BFGS (or a method of your choice) -- passes keywords arguments to method
    def argmax(self,with_rez=False,method='BFGS',**kwargs):
        suggested_solvers = ['CG','BFGS','L-BFGS-B','TNC','SLSQP']
        if not method in suggested_solvers:
            warnings.warn("Solver" + str(method) +
            " passed to argmax not in suggested list" + str(suggested_solvers)+
            '\nsee list at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html')
        z = self.z0()
        obj = autograd.value_and_grad(lambda z : -self.logp(z),0)
        kwargs['disp'] = False
        rez = scipy.optimize.minimize(obj,z,method=method,jac=True,options=kwargs)
        z = rez.x
        if with_rez:
            return rez.x, rez
        else:
            return rez.x

        
    def advi(self, algorithm='fullrank', verbose = True, **kwargs):
        #algorithm='fullrank',iter=100000,tol_rel_obj=.0001
        """
        ADVI returns constrained parameters by default.
        """
        try:
            with utils.suppress_stdout_stderr(verbose = verbose):
                rez = self.sm.vb(data=self.data,algorithm=algorithm,**kwargs)
        except:
            print('ADVI failed with error...\n', traceback.print_exc())
            return np.array([[]])
        samples = np.array(rez['sampler_params']).T[:,:-1]
        return samples

    def mf(self,**kwargs):
        return self.advi(algorithm='meanfield', verbose = True, **kwargs)


    def constrain(self, z):

        assert z.ndim == 2
        assert z.shape[-1] == self.zlen
        return np.array([self.fit.constrain_pars(np.asarray(z_,order='C')) for z_ in z])

    def get_n_samples(self, z):

        if not isinstance(z, dict):
            assert z.ndim == 2
            return z.shape[0]
        else:
            return list(z.values())[0].shape[0]

    def array_to_dict(self, z):

        assert self.get_n_samples(z) >= 1 
        assert z.shape[-1] == self.zlen

        return collections.OrderedDict({k:z[:, i] for i,k in enumerate(self.keys)})

    def dict_to_array(self, z):
        if not isinstance(z, dict):
            return z

        s = np.array([z[k]  for k in z.keys()])
        assert s.shape[-1] == self.zlen
        return s

    def unconstrain(self, z):
        N = self.get_n_samples(z)
        if not isinstance(z, (dict,)):
            z = self.array_to_dict(z)

        return np.array([self.fit.unconstrain_pars({k:v[n] for k,v in z.items()}) \
                                for n in range(N)])


    def log_prob(self, z):
        """
        A simple function to get batched sample evaluation for models.
        """
        orig_samples_shape = z.shape
        assert(orig_samples_shape[-1] == self.zlen)
        lp = np.array([self.logp(z_) for z_ in z.reshape(-1, self.zlen)])

        return lp.reshape(orig_samples_shape[:-1]) 

    @autograd.primitive
    def logp(self,z):
        assert(z.ndim==1)
        assert(len(z)==self.zlen)
        try: # try to evaluate logp with stan. if you fail, just return nan
            return self.fit.log_prob(z,adjust_transform=True)
        except:
            return np.nan
        return rez
        
    def glogp(self,z):
        assert(len(z)==self.zlen)
        rez_from_stan = self.fit.grad_log_prob(z,adjust_transform=True)
        return rez_from_stan.reshape(z.shape)

    def vjpmaker(argnum,rez,stuff,args):
        obj = stuff[0]
        z   = stuff[1]
        if np.isnan(rez):
            return lambda gg : 0*z # special gradient for nan case
        else:
            return lambda gg : gg*Model.glogp(obj,z)

###########################################################################
#  magic that makes autograd treat stan model logp as primitive function
###########################################################################

autograd.extend.defvjp_argnum(Model.logp, Model.vjpmaker)

