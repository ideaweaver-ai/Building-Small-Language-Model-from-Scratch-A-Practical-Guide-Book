def character_tokenize(text):
    """Tokenize text at character level."""
    # Convert to list of characters
    tokens = list(text)
    return tokens

# Example usage
text = "Neural networks learn patterns."
tokens = character_tokenize(text)
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
print(f"Vocabulary size: {len(set(tokens))}")
