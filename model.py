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
	train : bool = False
	
	def setup(self, do_rate = 0.3, block_size = None):
		"""
		Sets up the attention block
		do_rate - dropout probability, float
		
		"""
		total_number_of_features = self.n_embd * self.n_head
		self.qdense = nn.Dense(total_number_of_features)
		self.vdense = nn.Dense(total_number_of_features)
		self.kdense = nn.Dense(total_number_of_features)
		self.attn_dropout = nn.Dropout(rate = do_rate)
		if block_size is None:
			self.block_size = self.sequence_length
		else: self.block_size = block_size
		# this should be buffered such that update under SGD is avoided
		# see nn.Module.register_buffer for reference
		mask_inner = jnp.tril(jnp.ones((self.sequence_length, self.sequence_length)))
		self.mask = mask_inner
		#self.mask = lax.broadcast(mask_inner, (self.n_embd+1, self.n_head)) #,self.sequence_length, self.sequence_length))


	def __call__(self, x):
		"""
		Args:
		x    - input, torch tensor
		"""
		batch_size, sequence_length, _ = x.shape
		q, v, k = self.qdense(x), self.vdense(x), self.kdense(x)
		q = q.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)
		v = v.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)
		k = k.reshape(batch_size, sequence_length, self.n_head, self.n_embd).transpose(0, 2, 1, 3)

		att = (q @ k.transpose(0, 1, 3, 2)) * (1 / jnp.sqrt(self.n_embd))

		# masking :  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
		# replacing Tensor.masked_fill(mask, value)
		# mask  - boolean mask
		# value - the vbalue to fill in with
		# jax.lax.select(pred, on_true, on_false)
		minfs = lax.broadcast(-jnp.inf, att.shape)
		print("shape minfs : ", minfs.shape, "shape mask : ", self.mask.shape)
		mask = lax.broadcast(self.mask[:sequence_length,:sequence_length], (att.shape[0], att.shape[1]))
		att = lax.select(mask == 0, att, minfs)

		# apply softmax
		att = nn.softmax(att, axis=-1)

		# the manual way
		#key = random.PRNGKey(2023) # can/should be changed
		#dropout = random.bernoulli(key, p = p_do, shape = att.size())
		#att_do = att * dropout

		# using the nn module and jax
		# possibly use module.apply to get full jax functionality
		# may have to inherit something from a flax/jax module class
		att = self.attn_dropout(att, deterministic = not self.train)

		y = att @ v
		y = y.transpose(0,2,1,3)
		# MISSING: making y contiguous
		y = y.reshape((batch_size, sequence_length, self.n_embd*self.n_head))
		return y

class Block(nn.Module):
	n_head: int
	n_embd: int
	sequence_length: int
	train : bool = False

	def setup(self, do_rate = 0.3) -> None:
		self.ln_1 = nn.LayerNorm(self.n_embd)
		self.attn = CausalSelfAttention(n_head=self.n_head, 
										n_embd=self.n_embd, 
										sequence_length=self.sequence_length)
		self.ln_2 = nn.LayerNorm(self.n_embd)
		self.fc = nn.Dense(self.n_embd*self.n_head*4)
		self.c_project = nn.Dense(self.n_embd*self.n_head)
		self.act = nn.gelu
		self.block_dropout = nn.Dropout(rate = do_rate)
	
	def __call__(self, x):
		attention_output = x + self.attn(self.ln_1(x))
		mlp_output = attention_output + self.c_project(self.act(self.fc(attention_output)))
		mlp_output_do = self.block_dropout(mlp_output, deterministic = not self.train)

		return mlp_output_do
