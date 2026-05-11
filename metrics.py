import numpy as np 
import pandas as pd
import jax.numpy as jnp


def WinklerScore(y_val, pi_lower, pi_upper, alpha=0.1, returnCoverage=False):
    y_val = jnp.asarray(y_val)
    pi_lower = jnp.asarray(pi_lower)
    pi_upper = jnp.asarray(pi_upper)

    interval_length = pi_upper - pi_lower
    factor = 2.0 / alpha

    below = y_val < pi_lower
    above = y_val > pi_upper
    inside = ~(below | above)

    score = jnp.where(
        below,
        interval_length + factor * (pi_lower - y_val),
        interval_length
    )

    score = jnp.where(
        above,
        interval_length + factor * (y_val - pi_upper),
        score
    )

    coverage = jnp.mean(inside)

    return (jnp.mean(score), coverage) if returnCoverage else jnp.mean(score)



def GaussianNll(y, mu, sigma, eps=1e-6): # the lower the better, we want to minimize it. also wrong confidence, penalized. too little confidence also
    y = jnp.asarray(y)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    
    sigma = jnp.maximum(sigma, eps)
    var = sigma ** 2

    nll = 0.5 * (jnp.log(2 * jnp.pi * var) + ((y - mu) ** 2) / var)
    return jnp.mean(nll)
