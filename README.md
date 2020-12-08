# vistan

`vistan` is a simple library to run variational inference algorithms on Stan models.

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

code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""

data = {"J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}

posterior, model, results = vistan.inference(code = code, data = data) # runs Meanfield VI by default

samples = posterior.sample(1000)
for i in range(samples['eta'].shape[1]):
    plt.plot(samples["eta"][:,i], label = "eta[i]")
plt.show()

```

### Gaussian VI 
You can specify the hyper-parameters using the `vistan.hyperparams` function. We provide some default VI hyper-parameter choices which can accessed using `method` argument.   
```python
hyperparams = vistan.hyperparams(method = 'gaussian')

posterior, model, results = vistan.inference(code = code, data = data, 
                        hparams = hyperparams, verbose = True)

samples = posterior.sample(1000)
for i in range(samples['eta'].shape[1]):
    plt.plot(samples["eta"][:,i], label = "eta[i]")
plt.show()

```

### Flow-based VI
```python
hyperparams = vistan.hyperparams(method = 'flows')

posterior, model, results = vistan.inference(code = code, data = data, 
                        hparams = hyperparams, verbose = True)

samples = posterior.sample(1000)
for i in range(samples['eta'].shape[1]):
    plt.plot(samples["eta"][:,i], label = "eta[i]")
plt.show()

```

### ADVI

```python
hyperparams = vistan.hyperparams(method = 'advi')

posterior, model, results = vistan.inference(code = code, data = data, 
                                hparams = hyperparams, verbose = True)

samples = posterior.sample(1000)
for i in range(samples['eta'].shape[1]):
    plt.plot(samples["eta"][:,i], label = "eta[i]")
plt.show()

```

### Custom
You can also specify custom VI algorithms to work with your Stan models, just set the `method='custom'` and provide customized arguments.
```python
hyperparams = vistan.hyperparams(   method = 'custom', 
                                    vi_family = "gaussian",
                                    M_iw_train = 10,
                                    grad_estimator = "DReG",
                                    LI = True)

posterior, model, results = vistan.inference(code = code, data = data, 
                                hparams = hyperparams, verbose = True)

samples = posterior.sample(1000)
for i in range(samples['eta'].shape[1]):
    plt.plot(samples["eta"][:,i], label = "eta[i]")
plt.show()

```

## Limitations

- We currently only support inference on all latent parameters in the model
