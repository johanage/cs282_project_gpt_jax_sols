from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn

from model import GPT

BATCH_SIZE = 4
config = {
    "n_layers": 6,
    "n_head": 6,
    "n_embd": 192,
    "vocab_size": 4,
    "block_size": 11,
    "embd_pdrop": 0.1,
    "train": True
}

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)

x = random.randint(key1, (BATCH_SIZE, config["block_size"]), 0, config["vocab_size"])

model = GPT(**config)
params = model.init({"params": key2, 'dropout' : dropout_key}, x)

y = model.apply(params, x, rngs = {'dropout' : dropout_key})


#print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
#print('output:\n', y)
#print(f"Output shape: {y.shape}")

print(f"input: {x}")
x_2 = model.generate(params, x, 5, 1, rngs = {'dropout' : dropout_key})
print(f"output: {x_2}")
