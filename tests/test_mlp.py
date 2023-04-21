from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from model import GPT as GPT_jax

def test_mlp():
    key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
    x = random.randint(key1, (BATCH_SIZE, config_jax["block_size"]), 0, config_jax["vocab_size"])
    model_jax = GPT_jax(**config_jax)
    params_jax = model_jax.init({"params": key2, 'dropout' : dropout_key}, x)
    param_count = sum(x.size for x in jax.tree_leaves(params_jax))

