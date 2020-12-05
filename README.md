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

### Basic VI methods

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

method = vistan.method(max_iter = 1000, family = "full-rank gaussian", 
                        optimizer = "Adam")

posterior, _ = vistan.infer(code = code, data = data, method = method,
                         verbose = True, model_name = "test-model")

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
method = vistan.method( max_iter = 1000, family = "rnvp",  
                        M_training = 2, estimator = "DReG", optimizer = "Adam",
                        rnvp_num_transformations = 10, 
                        rnvp_num_hidden_units = 32, 
                        rnvp_num_hidden_layers = 2, 
                        rnvp_param_init_scale = 0.001
                        )

posterior, _ = vistan.infer(code = code, data = data, 
                                method = method, verbose = True)

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
method = vistan.method( max_iter = 1000, family = "gaussian",  
                        method = "advi", advi_adapt_eta = True,
                        advi_adapt_eta_max_iters = 100)

posterior, _ = vistan.infer(code = code, data = data, 
                                method = method, verbose = True)

samples = posterior.sample(1000)

plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

samples = posterior.sample(1000, M_sampling = 10)
plt.plot(samples["beta1"], label = "beta1")
plt.plot(samples["beta2"], label = "beta2")
plt.show()

```

## Limitations

- We will currently only support inference on all latent parameters in the model
