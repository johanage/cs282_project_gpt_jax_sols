import flax.linen as nn
from jax import lax, random, numpy as jnp
from tqdm import tqdm
import numpy as np


class LinearBlock(nn.Module):
    dense1_features: int
    dense2_features: int

    def setup(self):
        self.dense1 = nn.Dense(self.dense1_features)
        self.dense2 = nn.Dense(self.dense2_features)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x


class CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    block_size: int

    def setup(self, do_rate=0.1):
        """
        Sets up the attention block
        do_rate - dropout probability, float
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
        Args:
        x    - input, jnp tensor
        """
        batch_size, sequence_length, _ = x.shape
        q, v, k = self.qdense(x), self.vdense(x), self.kdense(x)
        embds_pr_head = self.n_embd // self.n_head
        q = q.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) * (1 / jnp.sqrt(embds_pr_head))
        # masking :  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # replacing Tensor.masked_fill(mask, value)
        # mask  - boolean mask
        # value - the vbalue to fill in with
        # jax.lax.select(pred, on_true, on_false)
        minfs = lax.broadcast(-jnp.inf, att.shape)
        mask = lax.broadcast(self.mask[:sequence_length, :sequence_length], (att.shape[0], att.shape[1]))
        att = lax.select(mask == 0, att, minfs)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)
        y = att @ v
        y = y.transpose(0, 2, 1, 3)
        # MISSING: making y contiguous
        y = y.reshape((batch_size, sequence_length, self.n_embd))
        y = self.resid_dropout(self.c_proj(y), deterministic=not training)
        return y


class MLP(nn.Module):
    n_embd: int

    def setup(self, do_rate=0.1):
        self.fc = nn.Dense(self.n_embd * 4)
        self.c_project = nn.Dense(self.n_embd)
        self.act = nn.gelu
        self.mlp_dropout = nn.Dropout(rate=do_rate)

    def __call__(self, x, training=False):
        mlp_out = self.c_project(self.act(self.fc(x)))
        mlp_do_out = self.mlp_dropout(mlp_out, deterministic=not training)
        return mlp_do_out


class Block(nn.Module):
    n_head: int
    n_embd: int
    block_size: int

    def setup(self, do_rate=0.1) -> None:
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(n_head=self.n_head,
                                        n_embd=self.n_embd,
                                        block_size=self.block_size)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlpf = MLP(self.n_embd)

    def __call__(self, x, training=False):
        attention_output = x + self.attn(self.ln_1(x), training=training)
        mlp_out = attention_output + self.mlpf(self.ln_2(attention_output), training=training)
        return mlp_out


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
        self.ln_f = nn.LayerNorm(self.n_embd)

        self.blocks = [Block(
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size

        )
            for _ in range(self.n_layers)]

        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
        # TODO: Special init

    def __call__(self, x, training=False):
        bath_size, sequence_length = x.shape
        pos = jnp.arange(0, sequence_length).reshape(1, sequence_length)
        tok_emb = self.wte(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not training)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

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
