import autograd.numpy as np
import autograd.numpy.random as npr
import functools 

import utilities as utils

class Dist():

    def __init__(self, zlen):
        
        self.zlen = zlen

    def initial_params(self):
        
        raise NotImplementedError

    def transform_params(self, params):
        
        raise NotImplementedError

    def get_params(self, params, **kwargs):

        return self.transform_params(params)

    def sample(self, params, sample_shape, **kwargs):

        raise NotImplementedError

    def log_prob(self, params, samples, **kwargs):

         raise NotImplementedError


S = np.log(np.exp(1)-1)


class Gaussian(Dist):

    def initial_params(self):
        
        return [np.zeros(self.zlen,), S*np.eye(self.zlen)]

    def transform_params(self, params):

        return params[0], utils.pos_tril(params[1])


    def sample(self, params, sample_shape = (), **kwargs):

        if isinstance(sample_shape, int) : sample_shape = (sample_shape,)

        mu, sig = self.get_params(params, **kwargs)
        
        samples = mu + np.dot(npr.randn(*sample_shape, self.zlen), sig.T)
        
        return samples

    def log_prob(self, params, samples, **kwargs):

        mu, sig = self.get_params(params, **kwargs)

        Λ = np.linalg.inv(sig).T

        M = (samples-mu)

        a = self.zlen*np.log(2*np.pi)

        b = 2*np.sum(np.log((np.diag(sig))))

        c = np.sum(np.matmul(M,np.matmul(Λ, Λ.T)) * M, -1)

        return -0.5*(a+b+c) 

    def entropy(self, params):

        mu, sig = self.get_params(params)

        a = self.zlen*np.log(2*np.pi) + self.zlen

        b = 2*np.sum(np.log((np.diag(sig))))

        return 0.5*(a+b)



class Diagonal(Gaussian):

    def initial_params(self):
        
        return [np.zeros(self.zlen,), S*np.ones(self.zlen,)]

    def transform_params(self, params):
        
        return params[0], np.diag(pos_diag(params[1]))


class Flows(Dist):

    def __init__(self, zlen):

        self.zlen = zlen
        self.base_dist = Gaussian(self.zlen)
        self.base_dist_params = self.base_dist.initial_params()

    def transform_params(self, params):

        return params

    def forward_transform(self, params, z, **kwargs):

        raise NotImplementedError

    def inverse_transform(self, params, x, **kwargs):

        raise NotImplementedError

    def sample(self, params, sample_shape, **kwargs):
        

        params = self.get_params(params, **kwargs)
        
        z_o = self.base_dist.sample(self.base_dist_params, sample_shape)

        samples, neg_log_det_J = self.forward_transform(params, z_o, **kwargs)

        return samples

    def log_prob(self, params, samples, **kwargs):

        params = self.get_params(params, **kwargs)

        z_o, neg_log_det_J = self.inverse_transform(params, samples, **kwargs)

        lq = self.base_dist.log_prob(self.base_dist_params, z_o)

        return lq + neg_log_det_J

def BinaryFlip(z, j):

    return np.concatenate(reversed(BinarySplit(z, j)), -1)

def ReverseBinaryFlip(z, j):

    return np.concatenate(reversed(ReverseBinarySplit(z, j)), -1)
    
def BinarySplit(z, j):

    D = z.shape[-1]

    d = D//2
    
    if D%2 == 1: d+=(np.int(j)%2)

    return np.array_split(z, [d], -1)

def ReverseBinarySplit(z, j):

    return BinarySplit(z, np.int(j)+1)

class RealNVP(Flows):

    def __init__(   self, 
                    num_transformations, 
                    num_hidden_units, 
                    num_hidden_layers, 
                    params_init_scale, 
                    zlen):
        
        self.num_transformations = num_transformations
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.params_init_scale = params_init_scale

        super(RealNVP, self).__init__(zlen = zlen)
    
    def initial_params(self):

        def generate_net_st():

            coupling_layer_sizes = utils.coupling_layer_specifications(self.num_hidden_units, self.num_hidden_layers, self.zlen)

            init_st_params = []

            for layer_sizes in coupling_layer_sizes:

                 init_st_params.append([(self.params_init_scale * npr.randn(m, n),   # weight matrix
                                               self.params_init_scale * npr.randn(n))      # bias vector
                                              for m, n in zip(layer_sizes[:-1], layer_sizes[1:])])
            return init_st_params

        st = [generate_net_st() for i in range(self.num_transformations)]

        return st

    def apply_net_st(self, params, inputs):

        inpW, inpb = params[0]

        inputs = utils.leakyrelu(np.dot(inputs, inpW) + inpb)

        for W, b in params[1:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = utils.leakyrelu(outputs)

        outW, outb = params[-1]
        outputs = np.dot(inputs, outW) + outb

        assert(outputs.shape[:-1] == inputs.shape[:-1])
        assert(outputs.shape[-1]%2 == 0)

        s,t = np.array_split(outputs, 2 , -1)

        assert(s.shape == t.shape)

        return utils.tanh(s),t

    def forward_transform(self, params, z):

        neg_log_det_J = np.zeros(z.shape[:-1])

        st_list = params

        for i in range(self.num_transformations):
            for j in range(2):

                z_1, z_2 = BinarySplit(z,j)

                s, t = self.apply_net_st( params = st_list[i][j], inputs = z_1)
                z = np.concatenate([z_1, z_2*np.exp(s) + t ], axis = -1)
                neg_log_det_J -= np.sum(s,axis=-1)  #Immediately update neg log jacobian with s outputs

                z = BinaryFlip(z, j)

        assert(z.shape[:-1] == neg_log_det_J.shape)
        return z, neg_log_det_J

    def inverse_transform(self, params, x):

        neg_log_det_J = np.zeros(x.shape[:-1])

        st_list = params
        
        for i in reversed(range(self.num_transformations)):
            for j in reversed(range(2)):

                x = ReverseBinaryFlip(x, j)

                x_1, x_2 = BinarySplit(x, j)

                s, t = self.apply_net_st( params = st_list[i][j], inputs = x_1)#output should be of the shape of x_1

                x = np.concatenate([x_1, (x_2 - t)*np.exp(-s) ], axis = -1)
                neg_log_det_J -= np.sum(s,axis=-1)

        assert(x.shape[:-1] == neg_log_det_J.shape)
        
        return x, neg_log_det_J


def get_var_dist(hyper_params):

    if hyper_params['vi_family'] == "gaussian":

        var_dist = Gaussian(hyper_params['latent_dim'])

    elif hyper_params['vi_family'] == "gaussian - diagonal":

        var_dist = Diagonal(hyper_params['latent_dim'])

    elif hyper_params['vi_family'] == "rnvp":

        var_dist = RealNVP  (   num_transformations = hyper_params['rnvp_num_transformations'], 
                                num_hidden_units = hyper_params['rnvp_num_hidden_units'], 
                                num_hidden_layers = hyper_params['rnvp_num_hidden_layers'], 
                                params_init_scale = hyper_params['rnvp_params_init_scale'], 
                                zlen = hyper_params['latent_dim']
                            )     
    else: 
        raise NotImplementedError
    
    return var_dist


class Posterior():

    def __init__(self, M_sampling, log_p, log_q, sample_q, zlen):

        self.M_sampling = M_sampling,
        self.log_p = log_p, 
        self.log_q = log_q,
        self.sample_q = sample_q
        self.zlen = zlen
                                         

    def sample(self, shape, params):

        if isinstance(shape, int): 

            shape = (shape,) 

        if self.M_sampling == 1:

            return self.sample_q(params, shape)

        orig_shape = shape

        shape = (utils.mul_iterable(shape), M_sampling)
        
        samples = self.sample_q(params, shape)

        lp = self.log_p(samples)

        lq = self.log_q(params, samples)

        lR = lp - lq

        final_samples = []

        for i in range(lR.shape[0]):

            j = np.argmax(npr.multinomial(1, softmax_matrix(lR[i])))
            
            final_samples.append(samples[i,j,:])

        return np.array(final_samples).reshape(orig_shape + (self.zlen,))

    def log_prob(self, samples, params):

        if self.M_sampling == 1:

            return self.log_q(params, samples)

        else :

            raise NotImplementedError

def get_posterior(model, var_dist, params, M_sampling):

    q = Posterior(M_sampling, model.log_prob, var_dist.log_prob, var_dist.sample, var_dist.zlen)

    q.sample = functools.partial(q.sample, params = params)

    q.log_prob = functools.partial(q.log_prob, params = params)

    return q
