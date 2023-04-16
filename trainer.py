# File to train minGPT model
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import jax
import jax.numpy as jnp
import haiku as hk

from jax.experimental import optimizers
import optax
from optax import chain, clip_by_global_norm, scale_by_adam, scale, scale_by_schedule, add_decayed_weights
from jax import local_device_count

from tqdm import tqdm
import math
import numpy as np
from typing import Mapping
import functools
from functools import partial
import pickle 

import torch
from torch.utils.data import Dataset, DataLoader

from train_config import *

class Trainer:
    def __init__(self, loss_fn, train_dataset, test_dataset):
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_training_params(self):
        prng, subkey = jax.random.split(prng)
        train_dl = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers)
        batch = next(iter(train_dl))
        x_batch, y_batch = map(jnp.array, batch)
        params = self.loss_fn.init(subkey, x_batch, y_batch)
        return params

    def run_trainer(self):
        params = self.get_training_params()
        optimizer = optax.adamw(learning_rate=learning_rate, b1=beta1, b2=beta2, weight_decay=weight_decay)
        loss_fn = self.loss_fn.apply
        it = 0 # counter used for learning rate decay
        for epoch in range(max_epochs):
            params, opt_state, it = self.run_epoch(params, opt_state, it, 'train')
            if self.test_dataset is not None:
                test_loss = self.run_epoch(params, opt_state, 0, 'test')

    def run_epoch(self, params, opt_state, it, mode):
        dataset = self.train_dataset if mode == 'train' else self.test_dataset
        loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=batch_size, num_workers=num_workers)
        losses = []


