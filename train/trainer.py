# File to train minGPT model
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils.data import DataLoader

from train.train_config import *

class Trainer:
    def __init__(self, loss_fn, train_dataset, test_dataset):
        self.loss_fn = loss_fn.apply
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_training_params(self):
        prng = jax.random.split(prng)
        loader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=num_workers)
        batch = next(iter(loader))
        x_batch, y_batch = map(jnp.array, batch)
        params = self.loss_fn.init(prng, x_batch, y_batch)
        return params

    def run_trainer(self):
        params = self.get_training_params()
        self.optimizer = optax.adamw(learning_rate=learning_rate, b1=beta1, b2=beta2, weight_decay=weight_decay)
        for epoch in range(max_epochs):
            params, optimizer_update = self.run_epoch(params, optimizer_update, 'train')
            if self.test_dataset:
                test_loss = self.run_epoch(params, optimizer_update, 0, 'test')
        return params, optimizer_update, test_loss

    def run_epoch(self, params, optimizer_update, mode):
        if mode == 'train':
            loader = DataLoader(self.train_dataset, shuffle=False, 
                                sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)), 
                                pin_memory=True, batch_size=batch_size, num_workers=num_workers)
        else: 
            loader = DataLoader(self.test_dataset, shuffle=True, pin_memory=True,
                                batch_size=batch_size,
                                num_workers=num_workers)
                            
        losses = []
        for batch in loader:
            x_batch, y_batch = map(jnp.array, batch)
            if mode == 'train':
                loss, params, optimizer_update = self.update(params, prng, x_batch, y_batch, optimizer_update)
            else:
                loss = self.get_loss(params, prng, x_batch, y_batch)

            loss = loss[0]    
            losses.append(loss)
        
        if mode == 'test':
            test_loss = float(jnp.mean(jnp.array(losses)))
            return test_loss
            
        return params, optimizer_update

    def get_loss(self, params, prng, x_batch, y_batch):
        loss = self.loss_fn(params, prng, x_batch, y_batch)
        return jax.lax.pmean(loss)

    def update(self, params, prng, x_batch, y_batch, optimizer_update):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, prng, x_batch, y_batch)
            
        grads = jax.lax.pmean(grads)
        loss = jax.lax.pmean(loss)
        
        updates, optimizer_update = self.optimizer.update(grads, optimizer_update, params)
        params = optax.apply_updates(params, updates)
        return loss, params, optimizer_update