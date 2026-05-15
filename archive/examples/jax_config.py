import jax.numpy as jnp
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
from jax import pmap
