# script for testing the mlp architecture
from mingpt_pytorch.model import CausalSelfAttention, Block, GPT
from mingpt_pytorch.utils import CfgNode
import torch

from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from model import GPT as GPT_jax 
from model import MLP as MLP_jax
from model import CausalSelfAttention as CausalSelfAttention_jax 
from model import Block as Block_jax 

from tests.config import config_gpt, config_jax, BATCH_SIZE

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
x_csa = random.uniform(key1, (BATCH_SIZE, config_jax["block_size"], config_jax['n_embd']), dtype=float, minval=0., maxval=1.)
csa_jax = CausalSelfAttention_jax(config_jax['n_head'], config_jax['n_embd'], config_jax['block_size'])
params_jax_csa = csa_jax.init({"params" : key2}, x_csa)

model_config = CfgNode(**config_gpt)
csa = CausalSelfAttention(model_config)
csa.eval()

# Get the weights from the OG GPT implementation
csa_param_dict = {}
for name, param in csa.named_parameters():
    if param.requires_grad:
        csa_param_dict[name] = jnp.array(param.detach().numpy())


"""
=================================================================================

Checking the Causal Self-Attention architecture implementation.
=================================================================================
"""
# Get a flattened key-value list.
csa_flat_params = traverse_util.flatten_dict(params_jax_csa, sep='/')
#print("CSA flattened parameter tree: \n", jax.tree_util.tree_map(jnp.shape, csa_flat_params) )

att_w = csa.get_parameter('c_attn.weight').T
att_b = csa.get_parameter('c_attn.bias')
#print("csa att_w size : ", att_w.size())
#print("csa att_b size : ", att_b.size())
q_w, k_w, v_w = att_w.split(att_w.size()[0], dim=1)
q_b, k_b, v_b = att_b.split(att_w.size()[0], dim=0)

csa_flat_params['params/c_proj/bias']   = csa_param_dict['c_proj.bias']
csa_flat_params['params/c_proj/kernel'] = csa_param_dict['c_proj.weight'].T
csa_flat_params['params/kdense/bias']   = jnp.array(k_b.detach().numpy()) 
csa_flat_params['params/kdense/kernel'] = jnp.array(k_w.detach().numpy())
csa_flat_params['params/qdense/bias']   = jnp.array(q_b.detach().numpy())
csa_flat_params['params/qdense/kernel'] = jnp.array(q_w.detach().numpy())
csa_flat_params['params/vdense/bias']   = jnp.array(v_b.detach().numpy())
csa_flat_params['params/vdense/kernel'] = jnp.array(v_w.detach().numpy())


# Unflatten.
unflat_params_csa = traverse_util.unflatten_dict(csa_flat_params, sep='/')
# Refreeze.
unflat_params_csa = freeze(unflat_params_csa)
jax.tree_util.tree_map(jnp.shape, unflat_params_csa)

y_csa_set_params = csa_jax.apply(unflat_params_csa, x_csa)
y_csa_gpt        = csa( torch.tensor( x_csa.tolist() ) )
#print("y csa jax\n", y_csa_set_params.shape)
#print("y csa GPT\n", y_csa_gpt.size())
print(" If the number below is below 10e-6 then the Causal Self-Attention works!:")
print( jnp.mean(jnp.abs( (y_csa_set_params - y_csa_gpt.detach().numpy()) ) ) )

