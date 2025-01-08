import re


def create_vocab(text: str) -> dict[str, int]:
    """
    Creates a vocabulary from the input text by splits the text into words.
    Returns a dictionary with the (unique) words as keys and their index as values.
    """
    words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    words = [item.strip() for item in words if item.strip()]
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
        return [self.token_to_int[word] for word in words]

    def decode(self, tokens: list[int]) -> str:
        """Decodes the tokens into text."""
        text = ' '.join([self.int_to_token[token] for token in tokens])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


if __name__ == '__main__':

    with open("/Users/bemihai/projects/llm-from-scratch/data/the-verdict.txt", "r") as f:
        inputs = f.read()

    vocab = create_vocab(inputs)
    tokenizer = SimpleTokenizerV1(vocab)

    input_text = "What struck me now was that, for the first time, he resented the tone."

    encoded_text = tokenizer.encode(input_text)
    print(f"Encoded: {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded: {decoded_text}")

    assert input_text == decoded_text
