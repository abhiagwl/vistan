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

### Default VI

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


posterior, model, trace = vistan.infer(code = code, data = data)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

samples = posterior.sample(1000, M_sampling = 10)
plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```


### Flow-based VI
```
hyperparams = vistan.hyper_params(max_iter = 1000, family = "rnvp",  
                                    M_training = 2, estimator = "DReG", optimizer = "adam",
                                    rnvp_num_transformations = 10, 
                                    rnvp_num_hidden_units = 32, 
                                    rnvp_num_hidden_layers = 2, 
                                    rnvp_param_init_scale = 0.001)

posterior, model, trace = vistan.infer(code = code, data = data, 
                        hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

samples = posterior.sample(1000, M_sampling = 10)
plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

### ADVI

```
hyperparams = vistan.hyperparams( max_iter = 1000, family = "gaussian",  
                        "advi_use" = True, advi_adapt_eta = True,
                        advi_adapt_eta_max_iters = 100)

posterior, model, trace = vistan.infer(code = code, data = data, 
                                hyperparams = hyperparams, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()
posterior.M_sampling = 10
samples = posterior.sample(1000)
plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

## Limitations

- We will currently only support inference on all latent parameters in the model
