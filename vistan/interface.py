import pystan
import warnings
import pickle
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

def is_good_model(code, data, model_name = None, verbose = True):
    if model_name is not None:

        model_name = model_name.replace("-","_")
    # TODO : make sure data is deterministic
    try:

        model = Model(code, data, model_name, verbose = verbose)
        # print(model.zlen)
        print(model.keys)
        exit()
        Z_nuts = model.sampling(iter=20, verbose = verbose)
        Z_advi = model.advi(iter=20, verbose = verbose)
        # Z_mf   = model.mf(iter=20)
        # print(Z_nuts)
        # print(Z_nuts.shape)
        # print(Z_mf)
        # print(Z_mf.shape)
        for Z in [Z_advi]:
            if Z.shape[1]!=Z_nuts.shape[1]:
                raise ValueError("NUTS and VI not same dimensions")

        # also check that optimization works
        z,rez  = model.argmax(True,method='BFGS',gtol=1e-3,maxiter=20) 

        if not rez.success:
            pass
            # print(rez)
            # raise ValueError("optimization failed")
        if len(z) != Z_nuts.shape[1]:
            print("some")
            raise ValueError("NUTS and argmax not same dimensions")

        return True

    except:
        print(f"Error during model check...")
        # print(traceback.print_exc())
        raise
        # return False

###########################################################################
#  stuff related to compiling and caching stan models
###########################################################################

# removes whitespace and such to prevent extra compiling
# this is only used for hashing (original code is compiled)


# import Cython
# get_model_name(model_name, code):
#     model_code_stripped = standardize_code(model_code)
#     code_hash = hashlib.md5(model_code_stripped.encode('ascii')).hexdigest()
#     if model_name is None:
#         return model_name+"-"+
#     else:
#         cache_fn = 'data/cached-models/{}-{}.pkl'.format(model_name, code_hash)

def get_compiled_model(model_code, model_name=None, verbose = False, **kwargs):
    """Use just as you would `stan` from pystan"""
    if not os.path.exists('data/cached-models'):
        print('creating cached-models/ to save compiled stan models')
        os.makedirs('data/cached-models')

    cache_fn = f'data/cached-models/{utils.get_cache_fname(model_name, model_code)}.pkl'

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

        # with utils.suppress_stdout_stderr(verbose = debugging_verbose):
        #     results = self.sm.vb(data = self.data, algorithm = "meanfield", iter = 10)
        #     self.advi_param_names = results['sampler_param_names']
        # except:
        #     print('Could not run advi on model ' + model_code)
        #     print('Try turning debugging_verbose on for better debugging.')
        #     raise 

    def get_constrained_param_shapes(self):
        results = self.fit.extract()
        N = self.get_n_samples(results)
        constrained_param_shapes = collections.OrderedDict()

        for k,v in results.items():
            if k == "lp__":
                continue 
            assert(N == v.shape[0])
            constrained_param_shapes[k] = v.shape[1:]
        return constrained_param_shapes


    def z0(self):
        """
            Get a random initialization for the latent parameters.  Useful for MAP inference.
        """
        return np.random.randn(self.zlen)

    def sampling(self, verbose = True, **kwargs):
        """
            A function to access PyStan MCMC sampling techniques. 
            Returns samples in the unconstrained domain, in the form an array.
            See array_to_dict function to convert into dictionary format.  
        """
        try: 
            assert kwargs.get('iter',2) >=2
            warnings.simplefilter('ignore')
            with utils.suppress_stdout_stderr(verbose):
                self.fit = self.sm.sampling(data=self.data,**kwargs)
            # print(self.fit.extract().keys())
            rez = self.unconstrain(self.fit.extract())
            warnings.simplefilter('default')
        except:
            raise
        return rez

    def argmax(self,with_rez=False,method='BFGS',**kwargs):
        """
            A function to obtain MAP estimate using scipy.optimize.minimize.
            Arguments:
                with_rez :  Boolean
                            If True, returns the entire results object along with z_map.
                method :    String
                            One of suggested_solvers = ['CG','BFGS','L-BFGS-B','TNC','SLSQP'].
                kwargs:     arguments for scipy.optimize.minimize. see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                            for more details 
            Returns:
                z_map:      parameter value at Maximum-a-posteriori
                rez:        result object from scipy.optimize.minimize
        """
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
        """
            A function to run Stan's variational Bayes method (ADVI)
            Arguments:
                algorithm:  "fullrank" or "meanfield"
                verbsoe:    Boolean
                            If True, prints the optimization messages from ADVI
                kwargs:     arguments for vb see https://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel.vb
                            for more details 
            Returns:
                samples:    samples from the final posterior
        """
        try:
            with utils.suppress_stdout_stderr(verbose = verbose):
                rez = self.sm.vb(data=self.data,algorithm=algorithm,**kwargs)
        except:
            print('ADVI failed with error...\n', traceback.print_exc())
            return np.array([[]])
        # def pystan_vb_extract(results):
        samples = np.array(rez["sampler_params"])[:-1,:].T
        # samples = rez['sampler_params']
        # param_specs = rez['sampler_param_names']
        # params1 = self.method1(samples, param_specs)
        # params2 = self.method2(samples)
        # p1 = self.unconstrain(params1)
        # p2 = self.unconstrain(params2)        #returning unconstrain for internal consistency
        # return p1, p2 
        return self.unconstrain(self.method2(samples))

    def method1(self, samples, param_specs):

        N = len(samples[0])


        # first pass, calculate the shape
        param_shapes = collections.OrderedDict()
        for param_spec in param_specs:
            splt = param_spec.split('[')
            name = splt[0]
            if len(splt) > 1:
                idxs = [int(i) for i in splt[1][:-1].split(',')]  # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
            else:
                idxs = ()
            param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

        # create arrays
        params = collections.OrderedDict([(name, np.nan * np.empty((N, ) +\
                                 tuple(shape))) for name, shape in param_shapes.items()])

        # second pass, set arrays
        for param_spec, param_samples in zip(param_specs, samples):
            splt = param_spec.split('[')
            name = splt[0]
            if len(splt) > 1:
                idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]  # -1 because pystan returns 1-based indexes for vb!
            else:
                idxs = ()
            params[name][(..., ) + tuple(idxs)] = param_samples

        return params

    def constrained_array_to_dict(self, samples):
        assert samples.ndim == 2
        N = samples.shape[0]

        constrained_param_shapes = self.get_constrained_param_shapes()
        params = collections.OrderedDict()
        idx = 0
        for param_name, param_shape in constrained_param_shapes.items():
            nparam = int(np.prod(param_shape))
            params[param_name] = np.reshape(samples[:, idx:idx+nparam], (N, *param_shape), order = "F")
            idx+=nparam
        assert idx == samples.shape[1]
        return params

    def mf(self,**kwargs):
        """
            A function to run Stan's variational Bayes method (ADVI) -- meanfield
        """
        return self.advi(algorithm='meanfield', verbose = True, **kwargs)


    def constrain(self, z):
        """
            A function to constrain the parameters from unconstrained to original constrained range
            as defined in the model.
            Arguments:
                z:  samples in array np.ndarray format
            Returns:
                samples: unconstrained samples np.ndarray format
        """

        assert z.ndim == 2
        # assert z.shape[-1] == self.zlen
        return np.array([self.fit.constrain_pars(np.asarray(z_,order='C')) for z_ in z])

    def get_n_samples(self, z):
        """
            A function to get the number of samples. 
            Arguments:
                z:  samples in array np.ndarray or dictionary format
            Returns:
                N: number of samples
        """

        if not isinstance(z, dict):
            assert z.ndim == 2
            return z.shape[0]
        else:
            return list(z.values())[0].shape[0]

    # def array_to_dict(self, z):
    #     """
    #         A function to convert parameters from np.ndarray format to dictionary. 
    #         Keys are inferred from the model definition.
    #         Arguments:
    #             z:  samples in array np.ndarray format
    #         Returns:
    #             samples: samples in dictionary format
    #     """

    #     assert self.get_n_samples(z) >= 1 
    #     assert z.shape[-1] == self.zlen

    #     return collections.OrderedDict({k:z[:, i] for i,k in enumerate(self.keys)})

    # def dict_to_array(self, z):
    #     """
    #         A function to convert parameters from dictionary format to np.ndarray. 
    #         Keys are used from the dictionary.
    #         Arguments:
    #             z: samples in dictionary format
    #         Returns:
    #             samples:  samples in array np.ndarray format
    #     """
    #     if not isinstance(z, dict):
    #         raise ValueError
    #     print(z.keys())
    #     for k in z.keys():
    #         print(z[k].shape)
    #     s = np.array([v  for k.v in z.items()])
    #     assert s.shape[-1] == self.zlen
    #     return s

    def unconstrain(self, z):
        """
            A function to unconstrain the parameters from original constrained range to unconstrained
            domain. Uses the model definition from Stan to transform.
            Arguments:
                z:  samples in array np.ndarray or dictionary format
            Returns:
                samples: unconstrained samples np.ndarray format
        """
        assert isinstance(z, dict)
        N = self.get_n_samples(z)
        # print(N)
        # if not isinstance(z, (dict,)):
        #     z = self.array_to_dict(z)
        # else:
            # not required. Stan takes care of this internally.
            # if "lp___" in list(z.keys()):
            #     del z['lp__']

        return np.array([self.fit.unconstrain_pars({k:v[n] for k,v in z.items()}) \
                                for n in range(N)])


    def log_prob(self, z):
        """
            A simple function to get batched sample evaluation for models.
            Arguments:
                z:  unconstrained samples in array np.ndarray 
            Returns:
                log p: model log p(z,x) evaluation at unconstrained z samples. Adjust transform in used under the hood. 
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

