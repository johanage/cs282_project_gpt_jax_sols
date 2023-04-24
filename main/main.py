from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn

import sys
import os
cwd = os.getcwd()
dir_proj = cwd + "/.."
sys.path.append(dir_proj)
from cs282_project_gpt_jax.model import GPT

BATCH_SIZE = 4
config = {
    "n_layers": 1,
    "n_head": 7,
    "n_embd": 21,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1
}

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)

x = random.randint(key1, (BATCH_SIZE, config["block_size"]), 0, config["vocab_size"])

model = GPT(**config)
print(model)

params = model.init({"params": key2, 'dropout' : dropout_key}, x)
param_count = sum(x.size for x in jax.tree_leaves(params))
print("Number of parameters: ", param_count)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes


y = model.apply(params, x, rngs = {'dropout' : dropout_key})


#print('output:\n', y)
#print(f"Output shape: {y.shape}")

print(f"input: {x}")
x_2 = model.generate(params, x, 5, random_key = dropout_key) #{'dropout' : dropout_key})
print(f"output: {x_2}")
