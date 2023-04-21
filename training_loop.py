from jax import lax, random, numpy as jnp
from train.trainer import create_train_state, Trainer
from model import GPT
import jax
from torch.utils.data import Dataset
import numpy as np

config = {
    "n_layers": 1,
    "n_head": 7,
    "n_embd": 21,
    "vocab_size": 3,
    "block_size": 5,
    "embd_pdrop": 0.1
}
key1, key2, dropout_key = random.split(random.PRNGKey(1), 3)

BATCH_SIZE = 4
init_rng = {"params": key2, 'dropout' : dropout_key}


learning_rate = 0.01
momentum = 0.99
model = GPT(**config)

state = create_train_state(model, init_rng, learning_rate, momentum, config, key=dropout_key)

class TestDataset(Dataset):
    def __init__(self):
        self.key = key1

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        key = jax.random.fold_in(key=self.key, data=idx)
        x = random.randint(key, (config["block_size"],), 0, config["vocab_size"])
        return np.array(x), np.array(x)


trainer = Trainer(TestDataset(), TestDataset(), train_state=state)

trainer.run_trainer(3)
print(trainer.metrics_history)