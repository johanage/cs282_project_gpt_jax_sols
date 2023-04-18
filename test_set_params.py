"""
This script will test the output of the jax implementation of the GPT model with the original torch implementation
by setting the biases and weights manually after initialization of the models.

We use the weights and biases initiatied by GPT.

For simplicity we set the architecture to have 1 layer. If it works for one layer it should work for several.
Should test for 2 layers after to check multi-layer correctness.

"""
from mingpt.model import CausalSelfAttention, Block, GPT
from mingpt.utils import CfgNode
import torch

from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from model import GPT as GPT_jax

BATCH_SIZE = 4
config_jax = {
    "n_layers": 1,
    "n_head": 7,
    "n_embd": 21,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1,
    "train": True
}

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
x = random.randint(key1, (BATCH_SIZE, config_jax["block_size"]), 0, config_jax["vocab_size"])
model_jax = GPT_jax(**config_jax)
params_jax = model_jax.init({"params": key2, 'dropout' : dropout_key}, x)
param_count = sum(x.size for x in jax.tree_leaves(params_jax))
print("Number of parameters: ", param_count)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(lambda x: x.shape, params_jax)) # Checking output shapes


config_gpt = {
    "n_layer": 1,
    "n_head": 7,
    "n_embd": 21,
    "sequence_length": 10,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "train": True,
	"model_type" : None
}

def count_parameters(model, full=False):
    if full:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_config = CfgNode(**config_gpt)
print(model_config)

model = GPT(model_config)
print(model)
gpt_param_count = count_parameters(model)
print("Number of parameters in 1-layer GPT model: ", gpt_param_count)
assert gpt_param_count == param_count, "jax implementation does not have the same amount of parameters as OG GPT"


"""
===========================================================================================================0
Here is where we set the weights and biases following the model surgery instructionson:
https://flax.readthedocs.io/en/latest/guides/model_surgery.html

Example from model surgery section on flax website:
===========================================================================================================0
# Somehow modify a layer
dense_kernel = flat_params['Dense_1/kernel']
flat_params['Dense_1/kernel'] = dense_kernel / jnp.linalg.norm(dense_kernel)
===========================================================================================================0
"""

# Get a flattened key-value list.
flat_params = traverse_util.flatten_dict(params_jax, sep='/')
print("Flattened parameter tree: \n", jax.tree_util.tree_map(jnp.shape, flat_params) )

# Get the weights from the OG GPT implementation
# put it into a dict
gpt_param_dict = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        gpt_param_dict[name] = param
print("GPT dict of named parameters that req grad: \n",  list(gpt_param_dict.keys()) )


# Unflatten.
unflat_params = traverse_util.unflatten_dict(flat_params, sep='/')
# Refreeze.
unflat_params = freeze(unflat_params)
jax.tree_util.tree_map(jnp.shape, unflat_params)

"""
===========================================================================================================
Here we produce output and check for equivalence between the two implementations.
===========================================================================================================
"""

# out jax implementation
y = model_jax.apply(params_jax, x, rngs = {'dropout' : dropout_key})

# out OG gpt
y_gpt = model(torch.tensor(x.tolist()))


