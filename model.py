import flax.linen as nn
from jax import lax, random, numpy as jnp
from tqdm import tqdm
import numpy as np


class CausalSelfAttention(nn.Module):
    """
    The Causal self-attention block.

    n_head     - number of heads in in the multi-headed attention
    n_embd     - embedding dimension
    block_size - the block 
    """

    n_head: int
    n_embd: int
    block_size: int

    def setup(self, do_rate=0.1):
        """
        Sets up the attention block as illustrated in the model graph from the notebook.

        do_rate - float, dropout probability
        """
        total_number_of_features = self.n_embd
        self.qdense = nn.Dense(total_number_of_features)
        self.vdense = nn.Dense(total_number_of_features)
        self.kdense = nn.Dense(total_number_of_features)
        self.c_proj = nn.Dense(total_number_of_features)
        self.mask = 1 - jnp.tril(jnp.ones((self.block_size, self.block_size)))
        self.attn_dropout = nn.Dropout(rate=do_rate)
        self.resid_dropout = nn.Dropout(rate=do_rate)

    def __call__(self, x, training=False):
        """
        x        - torch.tensor, input dat
        training - bool, indicating whether training is activated or not
        """

        out = None
        batch_size, sequence_length, _ = x.shape
        # ======================================================
        # Start by applying the qdense, vdense, kdense to the x input to get values for q, k and v.

        #YOUR CODE HERE.

        # ======================================================

        # ======================================================
        # Use the values for q, k, v and self.mask to calculate causal self-attention

        # YOUR CODE HERE.

        # ======================================================

        # ======================================================
        # Add the final projection and dropout layers

        # YOUR CODE HERE.

        # ======================================================
        return out


class MLP(nn.Module):
    n_embd: int

    def setup(self, do_rate=0.1):
        self.fc = nn.Dense(self.n_embd * 4)
        self.c_project = nn.Dense(self.n_embd)
        self.act = nn.gelu
        self.mlp_dropout = nn.Dropout(rate=do_rate)

    def __call__(self, x, training=False):
        out = None
        # ======================================================
        # Use the layers defined in setup to implement the MLP part of the transformer block

        #YOUR CODE HERE.

        # ======================================================

        return out


class Block(nn.Module):
    n_head: int
    n_embd: int
    block_size: int

    def setup(self, do_rate=0.1) -> None:
        self.ln_1 = nn.LayerNorm(1e-5)
        self.attn = CausalSelfAttention(n_head=self.n_head,
                                        n_embd=self.n_embd,
                                        block_size=self.block_size)
        self.ln_2 = nn.LayerNorm(1e-5)
        self.mlpf = MLP(self.n_embd)

    def __call__(self, x, training=False):
        out = None
        # ======================================================
        # Add a normalized, residually connected attention layer

        #YOUR CODE HERE.

        # ======================================================
        # ======================================================
        # Add a normalized, residually connected MLP layer

        #YOUR CODE HERE.

        # ======================================================

        return out


class GPT(nn.Module):
    n_layers: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    embd_pdrop: float

    def setup(self):
        self.wte = nn.Embed(self.vocab_size, self.n_embd)
        self.wpe = nn.Embed(self.block_size, self.n_embd)

        self.drop = nn.Dropout(self.embd_pdrop)
        self.ln_f = nn.LayerNorm(1e-5)

        self.blocks = [Block(
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size

        )
            for _ in range(self.n_layers)]

        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, x, training=False):
        out = None
        # ======================================================
        # Embed the input vectors x, and the positional vector, and add them together
        #YOUR CODE HERE.

        # ======================================================
        # ======================================================
        # Run the Transformer blocks
        # YOUR CODE HERE.

        # ======================================================
        # ======================================================
        # Run the final LayerNorm and lm_head layers
        # YOUR CODE HERE.

        # ======================================================

        return out

    def generate(self, params, x, max_new_tokens, random_key, temperature=1.0):
        sequence = [int(x_i) for x_i in x[0]]
        for i in tqdm(range(max_new_tokens)):
            key = random.fold_in(random_key, i)
            x = jnp.array(x)
            pred = self.apply(params, x, training=False)

            pred_token = int(random.categorical(key, pred[0][-1] / temperature))

            sequence.append(pred_token)

            x = np.array(x)

            x[0, 0:-1] = x[0, 1:]
            x[0, -1] = pred_token
        return sequence
