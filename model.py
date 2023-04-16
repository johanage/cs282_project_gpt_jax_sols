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
        total_number_of_features = self.n_embd
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
        embds_pr_head = self.n_embd//self.n_head
        q = q.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, sequence_length, self.n_head, embds_pr_head).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1 / jnp.sqrt(embds_pr_head))

        # att = lax.select(self.mask==0, att, lax.broadcast(-jnp.inf, att.shape))

        att = nn.softmax(att, axis=3)

        #att = self.attn_dropout(att, deterministic=not training, rngs=rngs)
        y = att @ v
        y = y.transpose(0,2,1,3)
        # MISSING: making y contiguous
        y = y.reshape((batch_size, sequence_length, self.n_embd))
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
        self.fc = nn.Dense(self.n_embd*4)
        self.c_project = nn.Dense(self.n_embd)
        self.act = nn.gelu

    def __call__(self, x):
        attention_output = x + self.attn(self.ln_1(x))
        # TODO Dropout
        mlp_output = attention_output + self.c_project(self.act(self.fc(attention_output)))

        return mlp_output

class GPT(nn.Module):
    n_layers: int
    n_head: int
    n_embd: int
    sequence_length: int
    vocab_size: int
    block_size: int
    embd_pdrop: float

    def setup(self):
        self.wte = nn.Embed(self.vocab_size, self.n_embd)
        self.wpe = nn.Embed(self.block_size, self.n_embd)
        self.drop = lambda x: x #;nn.Dropout(self.emdb_pdrop)
        self.ln_f = nn.LayerNorm(self.n_embd)

        self.blocks = [Block(
                                n_head=self.n_head,
                                n_embd=self.n_embd,
                                sequence_length=self.sequence_length
                            )
                       for _ in range(self.n_layers)]

        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

        # TODO: Special init

    def __call__(self, x):
        bath_size, sequence_length = x.shape
        pos = jnp.arange(0, sequence_length).reshape(1, sequence_length)
        tok_emb = self.wte(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits