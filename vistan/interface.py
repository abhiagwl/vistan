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
import logging
# get pystan to stop warning
logging.getLogger("pystan").propagate = False
###########################################################################
#  stuff related to loading stan models and data
###########################################################################


def is_good_model(code, data, model_name="test_model", verbose=True):
    """ A simple function to test if PyStan compiles, samples, optimizes
        the model correctly or not.

    Args:
        code (string): Stan code
        data (dict): Observations in the form of a dictionary
        model_name (str, optional): Defaults to "test_model".
        verbose (bool, optional): Defaults to True.

    Raises:
        ValueError: NUTS and ADVI unconstrained results do not match in
                    # of dimensions.
        ValueError: NUTS and Optimization results do not match in # of
                    dimensions.

    Returns:
        bool: Test result
    """

    if model_name is not None:
        model_name = model_name.replace("-", "_")
    try:

        model = Model(code, data, model_name, verbose=verbose)
        Z_nuts = model.sampling(iter=20, verbose=verbose)
        Z_advi = model.advi(iter=20, verbose=verbose)
        for Z in [Z_advi]:
            if Z.shape[1] != Z_nuts.shape[1]:
                raise ValueError("NUTS and VI not same dimensions")

        # also check that optimization works
        z, rez = model.argmax(
                            True, method='BFGS',
                            gtol=1e-3, maxiter=20)

        if len(z) != Z_nuts.shape[1]:
            raise ValueError("NUTS and argmax not same dimensions")

        return True
    except Exception as e:
        print(f"Error during model check...")
        raise e

###########################################################################
#  Compiling and Caching Stan models
###########################################################################


def get_compiled_model(model_code, model_name=None, verbose=False, **kwargs):
    """ A function to get PyStan compiled model.

    Args:
        model_code (string):
            Stan code
        model_name (string, optional):
            Defaults to None.
        verbose (bool, optional):
            Helps print some PyStan compilation warnings.
            Defaults to False.

    Returns:
        StanModel: Compiled StanModel for the model specified by model_code
    """
    if not os.path.exists('data/cached-models'):
        print('creating cached-models/ to save compiled stan models')
        os.makedirs('data/cached-models')

    cache_fn = f'data/cached-models/\
                        {utils.get_cache_fname(model_name, model_code)}.pkl'

    if os.path.isfile(cache_fn):
        try:
            with open(cache_fn, 'rb') as f:
                sm = pickle.load(f)
            print("Compiled model found...")
        except Exception as e:
            print("Error during re-loading the complied model.")
            print(f"Try recompiling the model. Changed the\
                    name of the model or delete \
                    the saved pickled file at {cache_fn}.")
            raise e
    else:
        try:
            print("Cached model not found. Compiling...")
            with utils.suppress_stdout_stderr(verbose):
                sm = pystan.StanModel(
                                    model_code=model_code,
                                    model_name=model_name, **kwargs)
        except Exception as e:
            print("Error during compilation...")
            print('Could not compile code for ' + model_code)
            raise e

        print("Caching model...")
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm

###########################################################################
#  Class to provide the log_prob function that hooks into autograd
###########################################################################


class Model:
    def __init__(self, model_code, data, model_name=None, verbose=False):
        """
            A class to interface with the autograd.

            Args
            ----------
                model_code (string):
                    Stan code for the model
                data (dict):
                    Data in the dictionary format
                model_name (string):
                    Name of the model for easier identification.
                    This along with code is used to cache compiled models.

                verbose (bool):
                    If True, it will print additional details involving
                    Stan compilation.

            Attributes
            ----------
                data(dict):
                     Model data
                model_name(string):
                     If None, only model_code is used to cache.
                model_code(string):
                     Stan code. Also, used to cache.
                sm(StanModel):
                      Complied StanModel instance.
                fit(StandModelFit4):
                    A StanModelFit4 instance obtained using self.sm.sampling
                keys(list):
                     Names of the unconstrained parameters.
                zlen(list):
                     Number of latent dimensions in the model.

        """
        # THIS IS SLOW! TURN OPTIMIZATION ON SOMEDAY!
        extra_compile_args = ['-O1', '-w', '-Wno-deprecated']

        self.data = data
        self.model_name = model_name
        self.model_code = model_code

        self.sm = get_compiled_model(
                        model_code=self.model_code,
                        extra_compile_args=extra_compile_args,
                        model_name=self.model_name, verbose=verbose)
        try:
            with utils.suppress_stdout_stderr(False):
                self.fit = self.sm.sampling(
                                    data=self.data, iter=100,
                                    chains=1, init=0)
        except Exception as e:
            print('Error occurred during a sampling check for compiled model.')
            print('Could not sample for ' + model_code)
            raise e

        self.keys = self.fit.unconstrained_param_names()
        self.zlen = len(self.keys)

    def get_constrained_param_shapes(self):
        """ A function to get output names and shapes

        Returns:
            dict:
                A dictionary containing the shapes of all the parameters that
                are return my the PyStan's sampling function. This involves
                transformed parameters and generated quantities. This is useful
                to transform the unconstrained results from ADVI and vistan to
                constrained format similar to PyStan's StanModelFit4.extract()
        """
        results = self.fit.extract()
        N = self.get_n_samples(results)
        constrained_param_shapes = collections.OrderedDict()

        for k, v in results.items():
            if k == "lp__":
                continue
            assert(N == v.shape[0])
            constrained_param_shapes[k] = v.shape[1:]
        return constrained_param_shapes

    def z0(self):
        """
        Returns
        ----------
        np.ndarray
            a random initialization used for optimization
        """
        return np.random.randn(self.zlen)

    def sampling(self, verbose=False, **kwargs):
        """
        Args
        ----------
        verbose : bool
            If True, will print the suppressed print statements.
        kwargs : dict
            keyword arguments passed to PyStan's StanModel.sampling

        Returns
        ----------
        np.ndarray
            An array of unconstrained latent variables of the shape
            (num_samples, self.zlen)
        """
        try:
            assert kwargs.get('iter', 2) >= 2
            with utils.suppress_stdout_stderr(verbose):
                logging.getLogger("pystan").propagate = verbose
                self.fit = self.sm.sampling(data=self.data, **kwargs)
                logging.getLogger("pystan").propagate = False
            rez = self.unconstrain(self.fit.extract())
        except Exception as e:
            print("Error during sampling from the model.")
            raise e
        return rez

    def argmax(self, with_rez=False, method='BFGS', **kwargs):
        """
        Args
        ----------
            with_rez :  Boolean
                If True, returns the entire results object along
                with z_map.
            method :    String
                One of suggested_solvers = ['CG','BFGS','L-BFGS-B',
                                                'TNC','SLSQP'].
            kwargs:  dict
                Arguments for scipy.optimize.minimize. See
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                for more details
        Returns
        ----------
            np.ndarray, dict (optional)
            parameter at Maximum-a-posteriori, result object from
            scipy.optimize.minimize
        """
        suggested_solvers = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']
        if method not in suggested_solvers:
            warnings.warn(
                        "Solver" + str(method)
                        + " passed to argmax not in suggested list"
                        + str(suggested_solvers)
                        + '\nsee list at:'
                        + 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html')
        z = self.z0()
        obj = autograd.value_and_grad(lambda z: -self.logp(z), 0)
        kwargs['disp'] = False
        rez = scipy.optimize.minimize(
                            obj, z,
                            method=method, jac=True, options=kwargs)
        z = rez.x
        if with_rez:
            return rez.x, rez
        else:
            return rez.x

    def advi(self, algorithm='fullrank', verbose=False, **kwargs):
        """
            A function to run Stan's variational Bayes method (ADVI)
            Args
            ----------
                algorithm:  "fullrank" or "meanfield\
                                    "
                verbsoe:    Boolean
                            If True, prints the optimization messages from ADVI
                kwargs:     arguments for vb see
                            https://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel.vb
                            for more details
            Returns
            ----------
                samples:    samples from the final posterior
        """
        try:
            with utils.suppress_stdout_stderr(verbose):
                rez = self.sm.vb(
                            data=self.data,
                            algorithm=algorithm, **kwargs)

        except Exception as e:
            print('Error during ADVI...')
            raise e

        samples = np.array(rez["sampler_params"])[:-1, :].T

        return self.unconstrain(self.constrained_array_to_dict(samples))

    def mf(self, verbose=False, **kwargs):
        """
            A function to run Stan's variational Bayes method
            (ADVI) -- meanfield
        """
        return self.advi(algorithm='meanfield', verbose=verbose, **kwargs)

    def constrained_array_to_dict(self, samples):
        """ A function to convert the constrained parameter
            output to dictionary format compatible with
            PyStan's StanModelFit4.extract() format.

        Args:
            samples (np.ndarray): Constrained params

        Returns:
            dict:
                A dictionary that consists of the parameters aranged in
                the same format as PyStan's StanModelFit4.extract()
        """
        assert samples.ndim == 2
        N = samples.shape[0]

        constrained_param_shapes = self.get_constrained_param_shapes()
        params = collections.OrderedDict()
        idx = 0
        for param_name, param_shape in constrained_param_shapes.items():
            nparam = int(np.prod(param_shape))
            params[param_name] = np.reshape(
                                        samples[:, idx:idx+nparam],
                                        (N, *param_shape), order="F")
            idx += nparam
        assert idx == samples.shape[1]
        return params

    def constrain(self, z):
        """
            A function to constrain the parameters from unconstrained
            to original constrained range
            as defined in the model.
            Args
            ----------
                z:  samples in array np.ndarray format
            Returns
            ----------
                np.ndarray:
                    unconstrained samples np.ndarray format
        """

        assert z.ndim == 2
        assert z.shape[-1] == self.zlen
        return np.array([
                        self.fit.constrain_pars(np.asarray(z_, order='C'))
                        for z_ in z])

    def get_n_samples(self, z):
        """
            A function to get the number of samples.
            Args
            ----------
                z:  samples in array np.ndarray or dictionary format
            Returns
            ----------
                N: number of samples
        """

        if not isinstance(z, dict):
            assert z.ndim == 2
            return z.shape[0]
        else:
            return list(z.values())[0].shape[0]

    def unconstrain(self, z):
        """
            A function to unconstrain the parameters from original
            constrained range to unconstrained
            domain. Uses the model definition from Stan to transform.
            Args
            ----------
                z:  samples in array np.ndarray or dictionary format
            Returns
            ----------
                samples: unconstrained samples np.ndarray format
        """
        assert isinstance(z, dict)
        N = self.get_n_samples(z)

        return np.array([
                        self.fit.unconstrain_pars({
                                        k: v[n] for k, v in z.items()})
                        for n in range(N)])

    def log_prob(self, z):
        """
            A simple function to get batched sample evaluation for models.
            Args
            ----------
                z:  unconstrained samples in array np.ndarray
            Returns
            ----------
                log p: model log p(z,x) evaluation at unconstrained z samples.
                Adjust transform in used under the hood.
        """
        orig_samples_shape = z.shape
        assert(orig_samples_shape[-1] == self.zlen)
        lp = np.array([self.logp(z_) for z_ in z.reshape(-1, self.zlen)])

        return lp.reshape(orig_samples_shape[:-1])

    @autograd.primitive
    def logp(self, z):
        """
            A simple function to get log prob on a single sample
            Args
            ----------
                z (np.ndarray):
                single unconstrained sample
            Returns
            ----------
                log p(z,x) evaluation at unconstrained z
        """
        assert(z.ndim == 1)
        assert(len(z) == self.zlen)

        try:
            # try to evaluate logp with stan. if you fail, just return nan
            return self.fit.log_prob(z, adjust_transform=True)
        except Exception as e:
            return np.nan

    def glogp(self, z):
        """
            A simple function to get grad(log prob) on a single sample

            Args
            ----------
                z (np.ndarray):
                single unconstrained sample
            Returns
            ----------
                grad log p(z,x) evaluation at unconstrained z
        """
        assert(len(z) == self.zlen)
        rez_from_stan = self.fit.grad_log_prob(z, adjust_transform=True)
        return rez_from_stan.reshape(z.shape)

    def vjpmaker(argnum, rez, stuff, args):

        """
            A simple function to covert the grad to vjp
        """
        obj = stuff[0]
        z = stuff[1]
        if np.isnan(rez):
            return lambda gg: 0*z  # special gradient for nan case
        else:
            return lambda gg: gg*Model.glogp(obj, z)

###########################################################################
#  magic that makes autograd treat stan model logp as primitive function
###########################################################################


autograd.extend.defvjp_argnum(Model.logp, Model.vjpmaker)
