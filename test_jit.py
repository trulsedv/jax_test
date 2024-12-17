import timeit

import jax.numpy as jnp
from jax import jit, random


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


def print_timeit(times):
    times = jnp.array(times)
    mean = jnp.mean(times) / 1000 * 1e6  # convert to microseconds
    std = jnp.std(times) / 1000 * 1e6

    print(f"{mean:.0f} μs ± {std:.2f} μs per loop")


if __name__ == '__main__':
    x = jnp.arange(5.0)
    print(selu(x))

    key = random.key(1701)
    x = random.normal(key, (1_000_000,))

    times = timeit.repeat(lambda: selu(x).block_until_ready(), repeat=7, number=1000)
    print_timeit(times)

    selu_jit = jit(selu)
    _ = selu_jit(x)
    times = timeit.repeat(lambda: selu_jit(x).block_until_ready(), repeat=7, number=1000)
    print_timeit(times)
