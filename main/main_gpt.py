# a main that is supposed to be identical in nature as main.py for the min/nano-GPT implementation

import sys
import os
cwd = os.getcwd()
dir_proj = cwd + "/.."
sys.path.append(dir_proj)
from cs282_project_gpt_jax.mingpt_pytorch.model import CausalSelfAttention, Block, GPT
from cs282_project_gpt_jax.mingpt_pytorch.utils import CfgNode

BATCH_SIZE = 4
config = {
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


model_config = CfgNode(**config)
print(model_config)

model = GPT(model_config)
print(model)
gpt_param_count = count_parameters(model)
print("Number of parameters in 1-layer GPT model: ", gpt_param_count)
