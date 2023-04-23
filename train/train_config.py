import jax

max_epochs = 3
batch_size = 64
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_norm_clip = 1.0
weight_decay = 0.1 
lr_decay = False
grad_norm_clip = 1.0
num_workers = 0
prng = jax.random.PRNGKey(42)