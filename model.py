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
        embds_pr_head = self.n_embd // self.n_head
        # ======================================================

        raise NotImplementedError # remove this when you implement the code

        # Start by applying the qdense, vdense, kdense to the x input to get values for q, k and v.
        # q, v, k =
        # shape of q, k, and v should be (batch size, sequence length, # heads, embeddings per head)
        # YOUR CODE HERE.
        # q = q.reshape(batch_size, sequence_length, ?, ?).transpose(0, 2, 1, 3)
        # v = v.reshape(batch_size, sequence_length, ?, ?).transpose(0, 2, 1, 3)
        # k = k.reshape(batch_size, sequence_length, ?, ?).transpose(0, 2, 1, 3)
        # ======================================================

        # ======================================================
        # Use the values for q, k, v and self.mask to calculate causal self-attention
        # att = (q @ k.transpose(0, 1, 3, 2)) * (1 / ?) # ? = normalization
        # minfs = lax.broadcast(? , ?) # JAX array of negative infinities. Shape should be that of attention
        # mask = lax.broadcast(, (att.shape[0], att.shape[1]))
        # YOUR CODE HERE.
        # att = # Apply causal mask to attention (HINT: Use lax.select)
        # att = # Softmax attention output
        # att = self.attn_dropout(att, deterministic=not training)
        # ======================================================

        # ======================================================
        # Get the outputs by multiplying attention with values
        # y = 
        # y = y.reshape((?, ? ,?)) # y should be of shape (batch size, sequence length, number of embeddings)
        
        # Apply dropout to the projection of y
        # y = self.?(?, deterministic=not training)
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

        raise NotImplementedError # remove this when you implement the code

        # Use the layers defined in setup to implement the MLP part of the transformer block

        # YOUR CODE HERE.
        # fc_out = ? # Apply fully connected layer to x
        # act_out = ? # Apply activation
        # mlp_out = ? # Apply projection
        # out = # Apply dropout [Make sure deterministic is not training]
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

        raise NotImplementedError # remove this when you implement the code

        # Add a normalized, residually connected attention layer to x

        # YOUR CODE HERE.
        # attention_output = ? 

        # ======================================================
        # ======================================================
        # Add a normalized, residually connected MLP layer

        # YOUR CODE HERE.
        # out = attention_output + ?

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
        self.wte = nn.Embed(self.vocab_size, self.n_embd) # Token Embedding Layer
        self.wpe = nn.Embed(self.block_size, self.n_embd) # Position Embedding Layer

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
        bath_size, sequence_length = x.shape
        # ======================================================

        raise NotImplementedError # remove this when you implement the code

        # Embed the input vectors x, and the positional vector, and add them together
        # YOUR CODE HERE.
        # pos = ? # Hint: Use jnp.arange. Shape should be (1, sequence length)
        # tok_emb = ? 
        # pos_emb = ?
        # x = ? # Apply dropout to sum, [Make sure deterministic when not training]

        # ======================================================
        # ======================================================
        # Run the Transformer blocks
        # YOUR CODE HERE.
        # for _ in _:
            # x = some_block_func()

        # ======================================================
        # ======================================================
        # Run the final LayerNorm and lm_head layers
        # YOUR CODE HERE.
        # x = ?
        # out = ?
        # ======================================================

        return out

    def generate(self, params, x, max_new_tokens, random_key, temperature=1.0, stop_tokens=None) -> List[int]:
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
            if stop_tokens is not None and pred_token in stop_tokens:
                break
        return sequence

    def verbose_generate(self, params, x, max_new_tokens, random_key, token_to_char, temperature=1.0, stop_tokens=None) -> str:
        sequence = [int(x_i) for x_i in x[0]]

        word = "".join(token_to_char[x_i] for x_i in sequence)
        print(word, end="")
        for i in range(max_new_tokens):
            key = random.fold_in(random_key, i)
            x = jnp.array(x)
            pred = self.apply(params, x, training=False)

            pred_token = int(random.categorical(key, pred[0][-1] / temperature))

            sequence.append(pred_token)

            x = np.array(x)
            chr = token_to_char[pred_token]
            word += chr
            print(chr, end="")

            x[0, 0:-1] = x[0, 1:]
            x[0, -1] = pred_token
            if stop_tokens is not None and pred_token in stop_tokens:
                break
        print()
        return word
