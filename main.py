from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn

from model import GPT

BATCH_SIZE = 4
config = {
    "n_layers": 4,
    "n_head": 7,
    "n_embd": 21,
    "sequence_length": 10,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1
}

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
print(key1, key2)
x = random.randint(key1, (BATCH_SIZE, config["sequence_length"]), 0, config["vocab_size"])
model = GPT(**config) #CausalSelfAttention(n_head=7, n_embd=3, sequence_length=10)
params = model.init(key2, x)

y = model.apply(params, x, rngs = {'dropout' : dropout_key})


print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
print(f"Output shape: {y.shape}")

print(model.generate(params, x, 5, 1))