import flax.linen as nn
from jax import lax, random, numpy as jnp
import optax

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
    train: bool = False

    def setup(self, do_rate = 0.3):
        """
        Sets up the attention block
        do_rate - dropout probability, float
        """
        total_number_of_features = self.n_embd
        self.qdense = nn.Dense(total_number_of_features)
        self.vdense = nn.Dense(total_number_of_features)
        self.kdense = nn.Dense(total_number_of_features)
        self.c_proj = nn.Dense(total_number_of_features)
        self.mask = 1-jnp.tril(jnp.ones((self.block_size, self.block_size)))
        self.attn_dropout  = nn.Dropout(rate = do_rate)
        self.resid_dropout = nn.Dropout(rate = do_rate)

    def __call__(self, x):
        """
        Args:
        x    - input, jnp tensor
        """
        batch_size, sequence_length, _ = x.shape
        q, v, k = self.qdense(x), self.vdense(x), self.kdense(x)
        embds_pr_head = self.n_embd//self.n_head
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
        #print("shape minfs : ", minfs.shape, "shape mask : ", self.mask.shape)
        mask = lax.broadcast(self.mask[:sequence_length,:sequence_length], (att.shape[0], att.shape[1]))
        #print(mask)
        att = lax.select(mask == 0, att, minfs)


        #print(att[0,0])
        # apply softmax
        att = nn.softmax(att, axis=-1)

        # the manual way
        #key = random.PRNGKey(2023) # can/should be changed
        #dropout = random.bernoulli(key, p = p_do, shape = att.size())
        #att_do = att * dropout

        # using the nn module and jax
        # possibly use module.apply to get full jax functionality
        # may have to inherit something from a flax/jax module class
        att = self.attn_dropout(att, deterministic=not self.train)

        y = att @ v
        y = y.transpose(0,2,1,3)
        # MISSING: making y contiguous
        y = y.reshape((batch_size, sequence_length, self.n_embd))
        # residual dropout
        y = self.resid_dropout(self.c_proj(y), deterministic = not self.train)
        return y

class MLP(nn.Module):
    n_embd: int
    train : bool = False

    def setup(self, do_rate = 0.3):
        self.fc = nn.Dense(self.n_embd*4)
        self.c_project = nn.Dense(self.n_embd)
        self.act = nn.gelu
        self.mlp_dropout = nn.Dropout(rate=do_rate)
    def __call__(self,x):
        mlp_out = self.c_project(self.act(self.fc( x ) ) )
        mlp_do_out = self.mlp_dropout(x, deterministic=not self.train)
        return mlp_do_out

class Block(nn.Module):
    n_head: int
    n_embd: int
    block_size: int
    train : bool = False
    def setup(self, do_rate = 0.3) -> None:
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(n_head=self.n_head,
                                        n_embd=self.n_embd,
                                        block_size=self.block_size,
                                        train=self.train)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlpf = MLP(self.n_embd, self.train)
        #self.fc = nn.Dense(self.n_embd*4)
        #self.c_project = nn.Dense(self.n_embd)
        #self.act = nn.gelu
        #self.block_dropout = nn.Dropout(rate=do_rate)

    def __call__(self, x):
        attention_output = x + self.attn(self.ln_1(x))
        mlp_out = attention_output + self.mlpf( self.ln_2(attention_output) )
		#mlp_output = attention_output + self.c_project(self.act(self.fc( self.ln_2(attention_output) ) ) )
        return mlp_out

class GPT(nn.Module):
    n_layers: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    embd_pdrop: float
    train: bool = False

    def setup(self):
        self.wte = nn.Embed(self.vocab_size, self.n_embd)
        self.wpe = nn.Embed(self.block_size, self.n_embd)

        self.drop = nn.Dropout(self.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.n_embd)

        self.blocks = [Block(
                                n_head=self.n_head,
                                n_embd=self.n_embd,
                                block_size=self.block_size,
                                train=self.train

                            )
                       for _ in range(self.n_layers)]

        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
        # TODO: Special init

    def __call__(self, x):
        bath_size, sequence_length = x.shape
        pos = jnp.arange(0, sequence_length).reshape(1, sequence_length)
        tok_emb = self.wte(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not self.train)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def loss_fn(self, params, x, y, rngs):
        y_hat = self.apply(params, x, rngs=rngs)
        loss = optax.softmax_cross_entropy(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
        return loss

    def generate(self, params, x, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, rngs=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            x = x if x.shape[1] <= self.block_size else x[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.apply(params, x, rngs=rngs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = lax.top_k(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.softmax(logits, axis=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = random.categorical(rngs["params"], probs, shape=(1,))[:, 0]
            else:
                _, idx_next = lax.top_k(probs, k=1)
            # append sampled index to the running sequence and continue
            x = jnp.concatenate((x, idx_next), axis=1)

        return x
