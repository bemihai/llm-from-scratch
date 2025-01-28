"""
Byte-pair encoding tokenizer used for training the GPT models.
    - vocabulary size for the original GPT-2 model is 52_257
    - BPE tokenizer can handle any unknown word
    - BPE breaks down the words into subword units or even individual characters
"""

import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(f"Encoded: {integers}")

decoded_text = tokenizer.decode(integers)
print(f"Decoded: {decoded_text}")

assert text == decoded_text