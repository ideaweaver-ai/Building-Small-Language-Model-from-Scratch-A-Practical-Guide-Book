def simple_bpe_tokenize(text, vocab):
    """
    Simplified BPE tokenization example.
    In practice, you'd use libraries like tiktoken or sentencepiece.
    """
    # Start with word level tokenization
    words = text.split()
    tokens = []
    
    for word in words:
        # Try to find the word in vocabulary
        if word in vocab:
            tokens.append(word)
        else:
            # Break into subwords (simplified - real BPE is more complex)
            # Try common prefixes and suffixes
            if word.startswith("un") and word[2:] in vocab:
                tokens.extend(["un", word[2:]])
            elif word.endswith("ly") and word[:-2] in vocab:
                tokens.extend([word[:-2], "ly"])
            elif word.endswith("ness") and word[:-4] in vocab:
                tokens.extend([word[:-4], "ness"])
            else:
                # Fall back to character level for unknown words
                tokens.extend(list(word))
    
    return tokens

# Example vocabulary (in practice, this would be learned from training data)
vocab = {
    "Neural", "networks", "learn", "patterns", 
    "efficient", "ly", "un", "happy", "happiness", "ness"
}

text = "Neural networks learn patterns efficiently. Unhappiness is complex."
tokens = simple_bpe_tokenize(text, vocab)
print(f"Tokens: {tokens}")
