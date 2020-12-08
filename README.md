# vistan

`vistan` is a simple library to run variational inference algorithms on Stan models. Our primary aim is to help you quickly run variational methods from [Advances in BBVI](https://proceedings.neurips.cc/paper/2020/file/c91e3483cf4f90057d02aa492d2b25b1-Paper.pdf) on any Stan model. `vistan` uses [autograd](https://github.com/HIPS/autograd) and [PyStan](https://github.com/stan-dev/pystan) under the hood.

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

### Meanfield VI

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
algo = vistan.algorithm() # runs Meanfield VI by default
posterior = algo(code, data) 
samples = posterior.sample(100000)

points = np.arange(0,1,.01)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```

### Gaussian VI 
We provide some default VI algorithm choices which can accessed using `method` argument.   
```python
algo = vistan.algorithm(method = 'gaussian')
posterior = algo(code, data) 
samples = posterior.sample(100000)

plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```

### Flow-based VI
```python
algo = vistan.algorithm(method = 'flows')
posterior = algo(code, data) 
samples = posterior.sample(100000)

plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```

### ADVI

```python
algo = vistan.algorithm(method = 'advi')
posterior = algo(code, data) 
samples = posterior.sample(100000)

plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```

### Custom
You can also specify custom VI algorithms to work with your Stan models, just set the `method='custom'` and provide customized arguments.
```python
algo = vistan.algorithm(method = 'custom', 
                        M_iw_train = 2,
                        grad_estimator = "DReG",
                        vi_family = "gaussian",
                        per_iter_sample_budget = 10,
                        max_iters = 100)
posterior = algo(code, data) 
samples = posterior.sample(100000)

plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()
```
### IW-sampling
We provide support to use IW-sampling at inference time (see [Advances in BBVI](https://proceedings.neurips.cc/paper/2020/file/c91e3483cf4f90057d02aa492d2b25b1-Paper.pdf) for more information.) This importance weights `M_iw_samples` candidate samples and picks one final sample. 
```
samples = posterior.sample(100000, M_iw_samples = 10)
plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```
### Initialization
We provide support to use Laplace's method to initialize the parameters for Gaussian VI.
```
algo = vistan.algorithm(method = 'gaussian', LI = True)
posterior = algo(code, data) 
samples = posterior.sample(100000)

plt.hist(samples['p'], 200, density = True, histtype = 'step')
plt.plot(points,scipy.stats.beta(2,5).pdf(points),label='True Posterior')
plt.legend()
plt.show()

```
### Building your own inference algorithms
We provide access to the `model.log_prob` function we use internally for optimization. This allows you to evaluate the log density in the unconstrained space for your Stan mode. Also, this function is differentiable in `autograd`. 
```
posterior = vistan.algorithm(max_iters = 1)(code, data) 
log_prob = posterior.model.log_prob

```



## Limitations

- We currently only support inference on all latent parameters in the model
