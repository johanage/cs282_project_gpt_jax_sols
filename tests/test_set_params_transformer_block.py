# script for testing the mlp architecture
import sys
import os
cwd = os.getcwd()
dir_proj = cwd + "/../.."
sys.path.append(dir_proj)

from cs282_project_gpt_jax.mingpt_pytorch.model import CausalSelfAttention, Block, GPT
from cs282_project_gpt_jax.mingpt_pytorch.utils import CfgNode
import torch

from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from cs282_project_gpt_jax.model import GPT as GPT_jax 
from cs282_project_gpt_jax.model import MLP as MLP_jax
from cs282_project_gpt_jax.model import CausalSelfAttention as CausalSelfAttention_jax 
from cs282_project_gpt_jax.model import Block as Block_jax 

from cs282_project_gpt_jax.tests.config import config_gpt, config_jax, BATCH_SIZE

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
x_csa = random.uniform(key1, (BATCH_SIZE, config_jax["block_size"], config_jax['n_embd']), dtype=float, minval=0., maxval=1.)
block_jax = Block_jax(config_jax['n_head'], config_jax['n_embd'], config_jax['block_size'])
params_jax_block = block_jax.init({"params" : key2}, x_csa)

model_config = CfgNode(**config_gpt)
block = Block(model_config)
block.eval()

# Get the weights from the OG GPT implementation
# put it into a dict
block_param_dict = {}
mlp_param_dict = {}
for name, param in block.named_parameters():
    if param.requires_grad:
        # store the params as jax numpy arrays in the dict
        block_param_dict[name] = jnp.array(param.detach().numpy())

"""
=================================================================================

Checking the Block architecture implementation.
=================================================================================
"""
# Get a flattened key-value list.
block_flat_params = traverse_util.flatten_dict(params_jax_block, sep='/')
#print("Transformer block flattened parameter tree: \n", jax.tree_util.tree_map(jnp.shape, block_flat_params) )
att_w = block.get_parameter('attn.c_attn.weight').T
att_b = block.get_parameter('attn.c_attn.bias')
q_w, k_w, v_w = att_w.split(att_w.size()[0], dim=1)
q_b, k_b, v_b = att_b.split(att_w.size()[0], dim=0)
block_flat_params['params/attn/c_proj/bias']      = block_param_dict['attn.c_proj.bias'] 
block_flat_params['params/attn/c_proj/kernel']    = block_param_dict['attn.c_proj.weight'].T
block_flat_params['params/attn/kdense/bias']      = jnp.array(k_b.detach().numpy())
block_flat_params['params/attn/kdense/kernel']    = jnp.array(k_w.detach().numpy())
block_flat_params['params/attn/qdense/bias']      = jnp.array(q_b.detach().numpy())
block_flat_params['params/attn/qdense/kernel']    = jnp.array(q_w.detach().numpy())
block_flat_params['params/attn/vdense/bias']      = jnp.array(v_b.detach().numpy())
block_flat_params['params/attn/vdense/kernel']    = jnp.array(v_w.detach().numpy())
block_flat_params['params/ln_1/bias']             = block_param_dict['ln_1.bias']
block_flat_params['params/ln_1/scale']            = block_param_dict['ln_1.weight']
block_flat_params['params/ln_2/bias']             = block_param_dict['ln_2.bias']
block_flat_params['params/ln_2/scale']            = block_param_dict['ln_2.weight']
block_flat_params['params/mlpf/c_project/bias']   = block_param_dict['mlp.c_proj.bias']
block_flat_params['params/mlpf/c_project/kernel'] = block_param_dict['mlp.c_proj.weight'].T
block_flat_params['params/mlpf/fc/bias']          = block_param_dict['mlp.c_fc.bias']
block_flat_params['params/mlpf/fc/kernel']        = block_param_dict['mlp.c_fc.weight'].T
# Unflatten.
unflat_params_block = traverse_util.unflatten_dict(block_flat_params, sep='/')
# Refreeze.
unflat_params_block = freeze(unflat_params_block)
jax.tree_util.tree_map(jnp.shape, unflat_params_block)

y_block_set_params = block_jax.apply(unflat_params_block, x_csa)
y_block_gpt        = block( torch.tensor(x_csa.tolist() ) )

#print("y block jax\n", y_block_set_params.shape)
#print("y block GPT\n", y_block_gpt.size())
print(" If the number below is below 10e-6 then the Transformer block works!:")
print(jnp.mean(jnp.abs( ( y_block_set_params - y_block_gpt.detach().numpy()) ) ) )

