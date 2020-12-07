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

```
code = """
    data {
        int<lower=0> N; 
        int<lower=0,upper=1> switc[N];
    }
    parameters {
         real<lower=0,upper=1> beta1;
         real<lower=2,upper=2.4> beta2;
    } 
    model {
        switc ~ bernoulli(beta1);
    }
    """
data = {
    "N" : 2, 
    "switc": [1,0]
}
# runs by default
posterior, model, results = vistan.infer(code = code, data = data)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

### Gaussian VI 
```
hyperparams = vistan.hyper_params(method = 'gaussian')

posterior, model, results = vistan.infer(code = code, data = data, 
                        hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

### Flow-based VI
```
hyperparams = vistan.hyper_params(method = 'flows')

posterior, model, results = vistan.infer(code = code, data = data, 
                        hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

### ADVI

```
hyperparams = vistan.hyperparams(method = 'advi')

posterior, model, results = vistan.infer(code = code, data = data, 
                                hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

### Custom

```
hyperparams = vistan.hyperparams(   method = 'custom', 
                                    vi_family = "gaussian",
                                    M_training = 10,
                                    grad_estimator = "DReG",
                                    LI = True)

posterior, model, results = vistan.infer(code = code, data = data, 
                                hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000, M_sampling = 20)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

## Limitations

- We will currently only support inference on all latent parameters in the model
