import jax.numpy as jnp
from encoding.bpe import BPETokenizer as student
from encoding.sol import BPETokenizer as sol

class TestEncoder():
    def __init__(self):
        pass

    def autograde(self):
        self.test_encoder()
        self.test_decoder()
    
    def test_encoder(self):
        student_encoder = student()
        solution_encoder = sol()

        text = "Hello, this is a test of the min-GPT tokenizer :) returns TRUE if 100% of your code is correct!"
        actual_encoded = student_encoder(text)
        expected_encoded = solution_encoder(text)

        assert jnp.allclose(actual_encoded, expected_encoded)

    def test_decoder(self):
        student_encoder = student()
        solution_encoder = sol()

        text = "Hello, this is a test of the min-GPT tokenizer :) returns TRUE if 100% of your code is correct!"
        encoded = student_encoder(text)

        actual_decoded = student_encoder.decode(encoded)
        expected_decoded = solution_encoder.decode(encoded)

        assert actual_decoded == expected_decoded
 