# vistan

`vistan` is a simple library to run variational inference algorithms on Stan models. 

![Screenshot](vistan-example.png)

`vistan` uses [autograd](https://github.com/HIPS/autograd) and [PyStan](https://github.com/stan-dev/pystan) under the hood, and aims to help you quickly run different variational methods from [Advances in BBVI](https://proceedings.neurips.cc/paper/2020/file/c91e3483cf4f90057d02aa492d2b25b1-Paper.pdf) on Stan models. 
## Features

> - **Initialization:** Laplace's method to initialize full-rank Gaussian
> - **Gradient Estimators:** Total-gradient, STL, DReG, closed-form entropy   
> - **Variational Families:** Full-rank Gaussian, Diagonal Gaussian, RealNVP
> - **Objectives:** ELBO, IW-ELBO
> - **IW-sampling:** Posterior samples using importance weighting

## Installation

```
pip install vistan
```

## Usage
The typical usage of the package would have three steps:
- Use default variational recipes as `vistan.recipe("meanfield")`. There are various options: 
    + `advi`: Run our implementation of ADVI's PyStan.
    + `meanfield`: Full-factorized Gaussian
    + `fullrank`: Full-rank Gaussian
    + `flows`: RealNVP flow-based VI
    + `method x`: Use methods from the paper [Advances in BBVI](https://proceedings.neurips.cc/paper/2020/file/c91e3483cf4f90057d02aa492d2b25b1-Paper.pdf) where x is one of `[0,1,2,3a,3b,4a,4b,4c,4d]`
- Create an algorithm as `algo=vistan.algorithm()`. There are numerous optional arguments:
    + `vi_family`: This can be one of `['gaussian', 'diagonal', 'rnvp']` (Default: `gaussian`)
    + `max_iter`: The maximum number of optimization iterations. (Default: 100)
    + `optimizer`: This can be `adam` or `advi`. (Default: `adam`)
    + `grad_estimator`: What gradient estimator to use. Can be `Total-gradient`, `STL`, `DReG`, or `closed-form-entropy`. (Default: `DReG`)
    + `M_iw_train`: The number of importance samples. Use 1 for standard variational inference or more for importance-weighted variational inference. (Default: 1)
    + `per_iter_sample_budget`: The total number of evaluations to use in each iteration. (Default: 100)
- Get an approximate posterior as `posterior=algo(code, data)`. This runs the algorithm on Stan model given by the string `code` with observations given by the `data`.
- Draw samples from the approximate posterior as `samples=posterior.sample(100)`. You can also evaluate the log-probability of the posterior as `posterior.log_prob(latents)`. You can also draw samples using importance weighting as `posterior.sample(100, M_iw_sample=10)`.

## Recipes
### Meanfield Gaussian 
We provide some default VI algorithm choices which can accessed using `vistan.recipe`   

```python
import vistan 
import matplotlib.pyplot as plt
import numpy as np 
import scipy
code = """
data {
    int<lower=0> N;
    int<lower=0,upper=1> x[N];
}
parameters {
    real<lower=0,upper=1> p;
}
model {
    p ~ beta(1,1);
    x ~ bernoulli(p);
}
"""
data = {"N":5, "x":[0,1,0,0,0]}
algo = vistan.recipe() # runs Meanfield VI by default
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```

### Full-rank Gaussian 
```python
algo = vistan.recipe(method = 'fullrank')
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```

### Flow-based VI
```python
algo = vistan.recipe(method = 'flows')
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```

### ADVI
Our implementation of PyStan's ADVI.
```python
algo = vistan.recipe(method = 'advi')
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```

## Custom algorithms
You can also specify custom VI algorithms to work with your Stan models using `vistan.algorithm`.
```python
algo = vistan.algorithm(M_iw_train = 2,
                        grad_estimator = "DReG",
                        vi_family = "gaussian",
                        per_iter_sample_budget = 10,
                        max_iters = 100)
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```
### IW-sampling
We provide support to use IW-sampling at inference time (see [Advances in BBVI](https://proceedings.neurips.cc/paper/2020/file/c91e3483cf4f90057d02aa492d2b25b1-Paper.pdf) for more information.) This importance weights `M_iw_sample` candidate samples and picks one. IW-sampling is a post-hoc step and can be used with any variational scheme.
```python
samples = posterior.sample(100000, M_iw_sample = 10)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```
### Initialization
We provide support to use Laplace's method to initialize the parameters for Gaussian VI.
```python
algo = vistan.algorithm(vi_family = 'gaussian', LI = True)
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```
### Building your own inference algorithms
We provide access to the `model.log_prob` function we use internally for optimization. This allows you to evaluate the log density in the unconstrained space for your Stan model. Also, this function is differentiable in `autograd`.
```python
log_prob = posterior.model.log_prob

```


## Limitations

- We currently only support inference on all latent parameters in the model
- No support for data sub-sampling.