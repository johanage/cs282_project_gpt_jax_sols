import jax

max_epochs = 10
batch_size = 64
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_norm_clip = 1.0
weight_decay = 0.1 
lr_decay = False
step_tokens = None 
grad_norm_clip = 1.0
warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
final_tokens = 260e9 
ckpt_path = None
num_workers = 0
prng = jax.random.PRNGKey(42)