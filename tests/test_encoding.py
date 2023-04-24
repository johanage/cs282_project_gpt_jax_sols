import jax.numpy as jnp
from encoding.bpe import BPETokenizer as student

class TestEncoder():
    def __init__(self):
        pass

    def autograde(self):
        self.test_encoder()
        self.test_decoder()
    
    def test_encoder(self):
        student_encoder = student()

        text = "Hello, this is a test of the min-GPT tokenizer :) returns TRUE if 100% of your code is correct!"
        
        actual_encoded = student_encoder(text)
        expected_encoded = jnp.array([15496, 11, 428, 318, 257, 1332, 286, 262, 949, 12, 38, 11571, 11241, 7509, 14373, 5860, 26751, 611, 1802, 4, 286, 534, 2438, 318, 3376, 0])

        assert jnp.allclose(actual_encoded, expected_encoded)

    def test_decoder(self):
        student_encoder = student()

        text = "Hello, this is a test of the min-GPT tokenizer :) returns TRUE if 100% of your code is correct!"

        encoded = student_encoder(text)
        actual_decoded = student_encoder.decode(encoded)

        assert actual_decoded == text
 
if __name__ == "__main__":
    TestEncoder().autograde()