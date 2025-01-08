"""Simple tokenizer that encodes the text into integers based on a provided vocabulary."""

import re


def create_vocab(text: str) -> dict[str, int]:
    """
    Creates a vocabulary from the input text by splits the text into words.
    Returns a dictionary with the (unique) words as keys and their index as values.
    """
    words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    words = [item.strip() for item in words if item.strip()]
    # Add special tokens: end of text, unknown token, etc.
    words.extend(["<|endoftext|>", "<|unk|>"])
    return {token: idx for idx, token in enumerate(sorted(set(words)))}


class SimpleTokenizerV1:
    """Simple tokenizer that encodes the text into integers based on the vocabulary."""

    def __init__(self, vocab: dict[str, int]):
        self.token_to_int = vocab
        self.int_to_token = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encodes the text into integers."""
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item.strip() for item in words if item.strip()]
        # Replace unknown words with <|unk|> token
        words = [word if word in self.token_to_int else "<|unk|>" for word in words]
        return [self.token_to_int[word] for word in words]

    def decode(self, tokens: list[int]) -> str:
        """Decodes the tokens into text."""
        text = ' '.join([self.int_to_token[token] for token in tokens])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


if __name__ == '__main__':

    with open("/Users/bemihai/projects/llm-from-scratch/data/the-verdict.txt", "r") as f:
        inputs = f.read()

    vocab = create_vocab(inputs)
    tokenizer = SimpleTokenizerV1(vocab)

    text_1 = "What struck me now was that, for the first time, he resented the tone. Evrika!"
    text_2 = "The verdict was a shock to many, but not to all. The judge was known for his harsh sentences."
    input_text = " <|endoftext|> ".join([text_1, text_2])

    encoded_text = tokenizer.encode(input_text)
    print(f"Encoded: {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded: {decoded_text}")

    assert input_text == decoded_text
