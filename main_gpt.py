# a main that is supposed to be identical in nature as main.py for the min/nano-GPT implementation
from mingpt.model import CausalSelfAttention, Block, GPT
from mingpt.utils import CfgNode

BATCH_SIZE = 4
config = {
    "n_layer": 4,
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


model_config = CfgNode(**config)
print(model_config)

model = GPT(model_config)
print(model)
