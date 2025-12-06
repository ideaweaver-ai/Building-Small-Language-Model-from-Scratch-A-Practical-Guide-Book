import re

def word_tokenize(text):
    """Tokenize text at word level."""
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens

# Training vocabulary might include "learn" but not "learns" or "learning"
training_vocab = {"learn", "network", "pattern"}

# At inference time, we encounter "learns" which isn't in vocabulary
text = "The model learns patterns"
tokens = word_tokenize(text)

print(f"Text: '{text}'")
print(f"Tokens: {tokens}")
print(f"\nChecking which tokens are in vocabulary:")
for token in tokens:
    if token in training_vocab:
        print(f"  '{token}' -> IN vocabulary")
    else:
        print(f"  '{token}' -> NOT in vocabulary (unknown word problem!)")
