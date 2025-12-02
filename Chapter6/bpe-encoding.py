#Install tiktoken first !pip install tiktoken
# Using tiktoken with modern GPT-3.5/GPT-4 tokenizer
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-3.5 and GPT-4

text = "Transformer architectures enable efficient language modeling."
tokens = enc.encode(text)
print(tokens)
decoded_tokens = [enc.decode([token]) for token in tokens]

print(f"Original text: {text}")
print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {decoded_tokens}")
