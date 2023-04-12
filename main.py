from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn

from model.model import LinearBlock, CausalSelfAttention, Block

key1, key2, dropout_key = random.split(random.PRNGKey(0), 3)
print(key1, key2)
x = random.uniform(key1, (4, 10, 21))
model = Block(n_head=7, n_embd=3, sequence_length=10) #CausalSelfAttention(n_head=7, n_embd=3, sequence_length=10)
params = model.init(key2, x)

y = model.apply(params, x)


print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
print(f"Output shape: {y.shape}")