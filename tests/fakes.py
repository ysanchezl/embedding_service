"""Test doubles shared across test modules."""


class FakeTokenizer:
    """One token per whitespace-separated word, like the HF tokenizer call the service uses."""

    def __call__(self, units, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [u.split() for u in units]}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeSentenceTransformer:
    """Stands in for SentenceTransformer: records how it was built and what it encoded."""

    max_seq_length = 21  # 21 - 2 special tokens - 1 prefix token = 18 per chunk
    tokenizer = FakeTokenizer()

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.encoded = []  # one entry per encode() call
        self.options = []

    def encode(self, texts, normalize_embeddings, batch_size):
        self.encoded.append(list(texts))
        self.options.append({"normalize": normalize_embeddings, "batch_size": batch_size})

        class Vectors:
            def tolist(self_inner):
                return [[float(len(t.split()))] for t in texts]

        return Vectors()

    def get_sentence_embedding_dimension(self):
        return 1
