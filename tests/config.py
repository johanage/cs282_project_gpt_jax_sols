config_gpt = {
    "n_layer": 1,
    "n_head": 7,
    "n_embd": 21,
    "sequence_length": 10,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "train": False,
	"model_type" : None
}

BATCH_SIZE = 4
config_jax = {
    "n_layers": 1,
    "n_head": 7,
    "n_embd": 21,
    "vocab_size": 10,
    "block_size": 10,
    "embd_pdrop": 0.1,
#    "train": True
}
