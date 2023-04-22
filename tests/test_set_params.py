"""
This script will test the output of the jax implementation of the GPT model with the original torch implementation
by setting the biases and weights manually after initialization of the models.

We use the weights and biases initiatied by GPT.

For simplicity we set the architecture to have 1 layer. If it works for one layer it should work for several.
Should test for 2 layers after to check multi-layer correctness.

"""
from mingpt_pytorch.model import CausalSelfAttention, Block, GPT
from mingpt_pytorch.utils import CfgNode
import torch

from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from model import GPT as GPT_jax

from config import config_gpt, config_jax, BATCH_SIZE

key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)
x = random.randint(key1, (BATCH_SIZE, config_jax["block_size"]), 0, config_jax["vocab_size"])
model_jax = GPT_jax(**config_jax)
params_jax = model_jax.init({"params": key2, 'dropout' : dropout_key}, x)
param_count = sum(x.size for x in jax.tree_leaves(params_jax))
print("Number of parameters: ", param_count)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(lambda x: x.shape, params_jax)) # Checking output shapes

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
        # store the params as jax numpy arrays in the dict
        gpt_param_dict[name] = jnp.array(param.detach().numpy())
print("GPT dict of named parameters that req grad: \n",list(gpt_param_dict.keys()) )

flat_params['params/wte/embedding'] = gpt_param_dict['transformer.wte.weight']
flat_params['params/wpe/embedding'] = gpt_param_dict['transformer.wpe.weight']
flat_params['params/blocks_0/ln_1/scale'] = gpt_param_dict['transformer.h.0.ln_1.weight']
flat_params['params/blocks_0/ln_1/bias'] = gpt_param_dict['transformer.h.0.ln_1.bias']
att_w = model.get_parameter('transformer.h.0.attn.c_attn.weight')
att_b = model.get_parameter('transformer.h.0.attn.c_attn.bias')
q_w, k_w, v_w = att_w.split(att_w.size()[-1], dim=0)
q_b, k_b, v_b = att_b.split(att_w.size()[-1], dim=0)
flat_params['params/blocks_0/attn/qdense/kernel']    = jnp.array(q_w.detach().numpy())
flat_params['params/blocks_0/attn/qdense/bias']      = jnp.array(q_b.detach().numpy())
flat_params['params/blocks_0/attn/vdense/kernel']    = jnp.array(v_w.detach().numpy())
flat_params['params/blocks_0/attn/vdense/bias']      = jnp.array(v_b.detach().numpy())
flat_params['params/blocks_0/attn/kdense/kernel']    = jnp.array(k_w.detach().numpy())
flat_params['params/blocks_0/attn/kdense/bias']      = jnp.array(k_b.detach().numpy())
flat_params['params/blocks_0/attn/c_proj/kernel']    = gpt_param_dict['transformer.h.0.attn.c_proj.weight']
flat_params['params/blocks_0/attn/c_proj/bias']      = gpt_param_dict['transformer.h.0.attn.c_proj.bias']
flat_params['params/blocks_0/ln_2/scale']            = gpt_param_dict['transformer.h.0.ln_2.weight']
flat_params['params/blocks_0/ln_2/bias']             = gpt_param_dict['transformer.h.0.ln_2.bias']
flat_params['params/blocks_0/mlpf/fc/kernel']        = gpt_param_dict['transformer.h.0.mlp.c_fc.weight'].T
flat_params['params/blocks_0/mlpf/fc/bias']          = gpt_param_dict['transformer.h.0.mlp.c_fc.bias']
flat_params['params/blocks_0/mlpf/c_project/kernel'] = gpt_param_dict['transformer.h.0.mlp.c_proj.weight'].T
flat_params['params/blocks_0/mlpf/c_project/bias']   = gpt_param_dict['transformer.h.0.mlp.c_proj.bias']
flat_params['params/ln_f/scale']                     = gpt_param_dict['transformer.ln_f.weight']
flat_params['params/ln_f/bias']                      = gpt_param_dict['transformer.ln_f.bias']
flat_params['params/lm_head/kernel']                 = gpt_param_dict['lm_head.weight'].T


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
#y = model_jax.apply(params_jax, x, rngs = {'dropout' : dropout_key})
y = model_jax.apply(unflat_params, x, rngs = {'dropout' : dropout_key})

# out OG gpt
y_gpt = model(torch.tensor(x.tolist()))

#print(y == y_gpt)
