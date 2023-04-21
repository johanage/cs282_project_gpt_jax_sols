# File to train minGPT model
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from train.train_config import *
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax
import jax
from jax import numpy as jnp, lax
from tqdm import tqdm
@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics
    key: jax.random.KeyArray

def create_train_state(module, rng, learning_rate, momentum, config, key):
    x = jnp.ones((1, config["block_size"]), dtype=int)
    params = module.init(rng, x, training=False)  # initialize parameters by passing a template image
    tx = optax.adamw(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty(), key=key)


@jax.jit
def compute_metrics(*, state, batch):
    x, y = batch
    y_hat = state.apply_fn(state.params, x, training=False)
    individual_loss = optax.softmax_cross_entropy_with_integer_labels(y_hat.reshape(-1, y_hat.shape[-1]),
                                                                      y.reshape(-1))
    zeros = jnp.zeros_like(y, dtype=float)

    individual_loss = individual_loss.reshape(*y.shape)
    individual_loss = lax.select(y == -1, zeros, individual_loss)
    loss = individual_loss.mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=y_hat, labels=y, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    dropout_train_key = jax.random.fold_in(key=state.key, data=state.step)
    x, y = batch

    def loss_fn(params):
        y_hat = state.apply_fn(params, x, rngs={"dropout": dropout_train_key})
        individual_loss = optax.softmax_cross_entropy_with_integer_labels(y_hat.reshape(-1, y_hat.shape[-1]),
                                                                          y.reshape(-1))
        zeros = jnp.zeros_like(y, dtype=float)

        individual_loss = individual_loss.reshape(*y.shape)
        individual_loss = lax.select(y == -1, zeros, individual_loss)
        loss = individual_loss.mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

class Trainer:
    def __init__(self, train_dataset, test_dataset, train_state):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_state = train_state
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        self.metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

    def run_trainer(self, epochs):
        for epoch in range(epochs):
            self.run_epoch('train')
            if self.test_dataset:
                self.run_epoch("test")
        return self.metrics_history

    def run_epoch(self, mode):
        if mode == 'train':
            loader = tqdm(self.train_loader)
        else:
            loader = self.test_loader

        state = self.train_state
        for batch in loader:
            #batch = map(jnp.array, batch)
            x, y = batch
            x, y = jnp.array(x), jnp.array(y)
            batch = (x, y)
            if mode == 'train':
                state = train_step(state=state, batch=batch)  # get updated train state (which contains the updated parameters)
                state = compute_metrics(state=state, batch=batch)
            else:
                state = compute_metrics(state=state, batch=batch)


        for metric, value in state.metrics.compute().items():  # compute metrics
            self.metrics_history[f'{mode}_{metric}'].append(value)  # record metrics
        if mode == "train":
            self.train_state = state.replace(
                metrics=state.metrics.empty())  # reset train_metrics for next training epoch