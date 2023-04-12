import flax.linen as nn
from jax import lax, random, numpy as jnp


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
    sequence_length: int

    def setup(self):
        total_number_of_features = self.n_embd * self.n_head
        self.qdense = nn.Dense(total_number_of_features)
        self.vdense = nn.Dense(total_number_of_features)
        self.kdense = nn.Dense(total_number_of_features)
        self.mask = jnp.tril(jnp.ones((self.sequence_length, self.sequence_length)))
        self.attn_dropout = nn.Dropout(rate = 0.3)

    def __call__(self, x):
        # TODO: DROPOUT
        # TODO: MASKING
        batch_size, sequence_length, _ = x.shape
        q, v, k = self.qdense(x), self.vdense(x), self.kdense(x)
        q = q.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1 / jnp.sqrt(self.n_embd))

        # att = lax.select(self.mask==0, att, lax.broadcast(-jnp.inf, att.shape))

        att = nn.softmax(att, axis=3)

        #att = self.attn_dropout(att, deterministic=not training, rngs=rngs)
        y = att @ v
        y = y.transpose(0,2,1,3)
        # MISSING: making y contiguous
        y = y.reshape((batch_size, sequence_length, self.n_embd*self.n_head))
        return y

class Block(nn.Module):
    n_head: int
    n_embd: int
    sequence_length: int

    def setup(self) -> None:
        # TODO: DROPOUT
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(n_head=self.n_head, n_embd=self.n_embd, sequence_length=self.sequence_length)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.fc = nn.Dense(self.n_embd*self.n_head*4)
        self.c_project = nn.Dense(self.n_embd*self.n_head)
        self.act = nn.gelu

    def __call__(self, x):
        attention_output = x + self.attn(self.ln_1(x))
        # TODO Dropout
        mlp_output = attention_output + self.c_project(self.act(self.fc(attention_output)))

        return mlp_output