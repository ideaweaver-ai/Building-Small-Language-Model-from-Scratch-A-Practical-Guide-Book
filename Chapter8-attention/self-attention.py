import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.key_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.value_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.d_k = output_dim

    def forward(self, token_embeddings):
        key_vectors = self.key_projection(token_embeddings)
        query_vectors = self.query_projection(token_embeddings)
        value_vectors = self.value_projection(token_embeddings)
        
        similarity_scores = query_vectors @ key_vectors.T
        
        attention_weights = torch.softmax(
            similarity_scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)), dim=-1
        )
        
        context_vectors = attention_weights @ value_vectors
        return context_vectors
