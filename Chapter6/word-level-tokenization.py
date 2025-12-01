import re

def word_tokenize(text):
    """Tokenize text at word level."""
    # Split on whitespace and punctuation
    # Keep punctuation as separate tokens
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens

# Example usage
text = "Neural networks learn patterns."
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# For building vocabulary, we'd collect all unique words
vocabulary = set(tokens)
print(f"Vocabulary size (this sentence): {len(vocabulary)}")
