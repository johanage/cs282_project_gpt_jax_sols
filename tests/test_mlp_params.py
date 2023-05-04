# script for testing the mlp architecture

import torch

from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util
import sys
import os

from mingpt_pytorch.model import CausalSelfAttention, Block, GPT
from mingpt_pytorch.utils import CfgNode

from model import GPT as GPT_jax 
from model import MLP as MLP_jax
from model import CausalSelfAttention as CausalSelfAttention_jax 
from model import Block as Block_jax 

from tests.config import config_gpt, config_jax, BATCH_SIZE

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
x = random.randint(key1, (BATCH_SIZE, config_jax["block_size"]), 0, config_jax["vocab_size"])
csa_seq_len = 10
x_mlp = random.uniform(key1, (BATCH_SIZE, config_jax["block_size"], config_jax['n_embd']), dtype=float, minval=0., maxval=1.)
mlp_jax = MLP_jax(config_jax['n_embd'])
params_jax_mlp = mlp_jax.init({"params" : key2}, x_mlp)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_jax_mlp))
print("Number of parameters in the MLP: ", param_count)
#print('initialized parameter shapes:\n', jax.tree_util.tree_map(lambda x: x.shape, params_jax)) # Checking output shapes

def count_parameters(model, full=False):
    if full:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_config = CfgNode(**config_gpt)

block = Block(model_config)
block.eval()
gpt_param_count = count_parameters(block)
#print("Number of parameters in 1-layer GPT model: ", gpt_param_count)

# Get the weights from the OG GPT implementation
# put it into a dict
mlp_param_dict = {}
for name, param in block.named_parameters():
    if param.requires_grad:
        # store the params as jax numpy arrays in the dict
        if "mlp" in name:
            mlp_param_dict[name] = jnp.array(param.detach().numpy())

"""
=================================================================================

Checking the MLP architecture implementation.
=================================================================================
"""
# Get a flattened key-value list.
mlp_flat_params = traverse_util.flatten_dict(params_jax_mlp, sep='/')
#print("MLP flattened parameter tree: \n", jax.tree_util.tree_map(jnp.shape, mlp_flat_params) )
mlp_flat_params['params/c_project/bias']   = mlp_param_dict['mlp.c_proj.bias']
mlp_flat_params['params/c_project/kernel'] = mlp_param_dict['mlp.c_proj.weight'].T
mlp_flat_params['params/fc/bias']          = mlp_param_dict['mlp.c_fc.bias']
mlp_flat_params['params/fc/kernel']        = mlp_param_dict['mlp.c_fc.weight'].T
# Unflatten.
unflat_params_mlp = traverse_util.unflatten_dict(mlp_flat_params, sep='/')
# Refreeze.
unflat_params_mlp = freeze(unflat_params_mlp)
jax.tree_util.tree_map(jnp.shape, unflat_params_mlp)

y_mlp_set_params = mlp_jax.apply(unflat_params_mlp, x_mlp)
y_mlp_gpt        = block.mlpf( torch.tensor(x_mlp.tolist(), dtype=torch.float) )

#print("y jax\n", y_mlp_set_params.shape)
#print("y GPT\n", y_mlp_gpt.size())
print(" If the number below is below 10e-6 then the MLP works!:")
print(jnp.mean(jnp.abs( (y_mlp_set_params - y_mlp_gpt.detach().numpy() ))))
