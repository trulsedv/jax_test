import timeit

import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap

from test_jit import print_timeit

key = random.key(1701)
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))


def apply_matrix(x):
    return jnp.dot(mat, x)


def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])


@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, mat.T)


@jit
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)


print('Naively batched', flush=True)
times = timeit.repeat(lambda: naively_batched_apply_matrix(batched_x).block_until_ready(), repeat=7, number=1000)
print_timeit(times)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)

print('Manually batched', flush=True)
times = timeit.repeat(lambda: batched_apply_matrix(batched_x).block_until_ready(), repeat=7, number=1000)
print_timeit(times)


np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)

print('Auto-vectorized with vmap', flush=True)
times = timeit.repeat(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(), repeat=7, number=1000)
print_timeit(times)
