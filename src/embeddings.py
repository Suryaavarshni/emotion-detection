import numpy as np



def load_glove_embeddings(path="data/raw/glove.6B.100d.txt"):
    """
    Load GloVe embeddings from a .txt file into a dict: {word: vector}
    """
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} GloVe word vectors.")
    return embeddings


def load_emoji2vec(path="data/raw/emoji2vec.bin"):
    """
    Manually parse a word2vec-format binary file (used by Emoji2Vec).
    Returns a dict: {emoji: vector}
    """
    vectors = {}
    with open(path, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())

        for _ in range(vocab_size):
            word = b""
            while True:
                ch = f.read(1)
                if ch == b" ":
                    break
                if ch != b"\n":
                    word += ch
            word = word.decode("utf-8")

            vector = np.frombuffer(f.read(4 * vector_size), dtype=np.float32).copy()
            vectors[word] = vector

    print(f"Loaded Emoji2Vec with {len(vectors)} emoji vectors (dim={vector_size}).")
    return vectors


def get_word_vector(word, glove_embeddings, dim=100):
    """Return GloVe vector for a word, or a zero-vector if not found."""
    return glove_embeddings.get(word, np.zeros(dim, dtype="float32"))


def get_emoji_vector(emoji_char, emoji2vec_model, dim=300):
    """Return Emoji2Vec vector for an emoji, or a zero-vector if not found."""
    return emoji2vec_model.get(emoji_char, np.zeros(dim, dtype="float32"))


def get_text_embedding_matrix(tokens, glove_embeddings, dim=100):
    """Convert a list of tokens into a matrix of GloVe vectors (seq_len x dim)."""
    return np.array([get_word_vector(t, glove_embeddings, dim) for t in tokens])


def get_emoji_embedding_matrix(emojis, emoji2vec_model, dim=300):
    """Convert a list of emojis into a matrix of Emoji2Vec vectors (num_emojis x dim)."""
    if len(emojis) == 0:
        return np.zeros((1, dim), dtype="float32")  # placeholder if no emojis
    return np.array([get_emoji_vector(e, emoji2vec_model, dim) for e in emojis])


if __name__ == "__main__":
    # Quick test
    glove = load_glove_embeddings()
    e2v = load_emoji2vec()

    sample_tokens = ["i", "am", "happy", "today"]
    sample_emojis = ["😄", "🎉"]

    text_matrix = get_text_embedding_matrix(sample_tokens, glove)
    emoji_matrix = get_emoji_embedding_matrix(sample_emojis, e2v)

    print("Text embedding matrix shape:", text_matrix.shape)
    print("Emoji embedding matrix shape:", emoji_matrix.shape)