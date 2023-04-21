import jax
import torch
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn, traverse_util

from model import GPT as GPT_jax
from model import MLP
from mingpt.model import CausalSelfAttention, Block, GPT
from mingpt.utils import CfgNode

from config import config_gpt, config_jax, BATCH_SIZE

class TestMLP():
    def __init__(self):
        self.mlp_train = False
        self.mlp_embd = 3
        self.mlp_do_rate = 0.2
        self.test_mlp_forward_input = 0.5
        self.mlp = MLP(self.mlp_embd, self.mlp_train)
        self.mlp.setup(0.2)

    def test_mlp_init(self):
        assert self.mlp.fc == nn.Dense(4*self.mlp_embd)
        assert self.mlp.c_project == nn.Dense(self.mlp_embd)
        assert self.mlp.act == nn.gelu
        assert self.mlp.mlp_dropout == nn.Dropout(rate=self.mlp_do_rate)