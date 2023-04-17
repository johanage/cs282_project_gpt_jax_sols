from jax import lax, random, numpy as jnp
import jax
from flax.core import freeze, unfreeze
from flax import linen as nn
from model import LinearBlock, CausalSelfAttention, Block
key1, key2, dropout_key = random.split(random.PRNGKey(0), 3)
print(key1, key2)
x = random.uniform(key1, (2, 3, 21))
print(" type of x is : ", type(x))
model = Block(n_head=7, n_embd=3, sequence_length=3) #CausalSelfAttention(n_head=7, n_embd=3, sequence_length=10)
params = model.init(key2, x)
# for the dropout to work a rng key with prng_collective (name) ´dropout´ 
# has to be added to the arguments as a dictionary with key prng_collective and 
# value the prng key
y = model.apply(params, x, rngs = {'dropout' : dropout_key})
print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
print(f"Output shape: {y.shape}")
