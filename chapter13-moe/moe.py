"""
Mixture of Experts (MoE) Implementation from Scratch

This module implements a complete MoE layer including:
- Expert networks (feedforward networks)
- Gating network (routing mechanism)
- Top-k routing
- Load balancing loss
- Complete model example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert network - a feedforward neural network."""
    
    def __init__(self, d_model, d_ff, activation='relu'):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = activation
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.w1(x)  # Expand to d_ff dimensions
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        x = self.w2(x)  # Project back to d_model
        return x


class GatingNetwork(nn.Module):
    """Gating network that routes tokens to experts."""
    
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Compute logits for each expert
        logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        # Apply softmax to get probability distribution
        probs = F.softmax(logits, dim=-1)
        return probs, logits


def top_k_routing(gate_logits, k=2):
    """
    Select top k experts for each token.
    
    Args:
        gate_logits: (batch_size, seq_len, num_experts) - logits from gating network
        k: number of experts to select per token
    
    Returns:
        top_k_probs: (batch_size, seq_len, k) - probabilities of selected experts
        top_k_indices: (batch_size, seq_len, k) - indices of selected experts
    """
    # Get top k values and their indices
    top_k_probs, top_k_indices = torch.topk(
        F.softmax(gate_logits, dim=-1), 
        k=k, 
        dim=-1
    )
    return top_k_probs, top_k_indices


class MoELayer(nn.Module):
    """Complete Mixture of Experts layer."""
    
    def __init__(self, d_model, d_ff, num_experts, k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Number of experts to activate per token
        self.capacity_factor = capacity_factor
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(d_model, num_experts)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Compute gating scores
        gate_probs, gate_logits = self.gate(x)
        # gate_probs shape: (batch_size, seq_len, num_experts)
        
        # Step 2: Select top k experts
        top_k_probs, top_k_indices = top_k_routing(gate_logits, k=self.k)
        # top_k_probs shape: (batch_size, seq_len, k)
        # top_k_indices shape: (batch_size, seq_len, k)
        
        # Step 3: Process tokens through selected experts
        # Flatten batch and sequence dimensions for easier processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.k)  # (batch_size * seq_len, k)
        top_k_probs_flat = top_k_probs.view(-1, self.k)  # (batch_size * seq_len, k)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each token through its selected experts
        for token_idx in range(batch_size * seq_len):
            token_expert_indices = top_k_indices_flat[token_idx]  # (k,)
            token_expert_probs = top_k_probs_flat[token_idx]  # (k,)
            token_input = x_flat[token_idx:token_idx+1]  # (1, d_model)
            
            # Get outputs from selected experts
            expert_outputs = []
            for expert_idx in token_expert_indices:
                expert_output = self.experts[expert_idx](token_input)
                expert_outputs.append(expert_output)
            
            # Combine expert outputs with weighted sum
            expert_outputs = torch.cat(expert_outputs, dim=0)  # (k, d_model)
            # Weight by gating probabilities
            weighted_output = (expert_outputs * token_expert_probs.unsqueeze(-1)).sum(dim=0)
            output[token_idx] = weighted_output
        
        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, d_model)
        
        return output, gate_probs


def load_balancing_loss(gate_probs, num_experts):
    """
    Compute load balancing loss to encourage uniform expert usage.
    
    Args:
        gate_probs: (batch_size, seq_len, num_experts) - gating probabilities
        num_experts: number of experts
    
    Returns:
        loss: scalar tensor
    """
    # Compute average probability per expert across all tokens
    # gate_probs shape: (batch_size, seq_len, num_experts)
    expert_usage = gate_probs.mean(dim=[0, 1])  # (num_experts,)
    
    # Ideal usage is uniform: 1/num_experts for each expert
    ideal_usage = 1.0 / num_experts
    
    # Compute coefficient of variation (standard deviation / mean)
    # Higher variance = more imbalanced = higher loss
    variance = torch.var(expert_usage)
    mean_usage = expert_usage.mean()
    cv_squared = variance / (mean_usage ** 2 + 1e-10)
    
    return cv_squared * num_experts


class SimpleMoEModel(nn.Module):
    """Simple model with MoE layer."""
    
    def __init__(self, vocab_size, d_model, d_ff, num_experts, k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.moe_layer = MoELayer(d_model, d_ff, num_experts, k=k)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Embed tokens
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # Process through MoE layer
        x, gate_probs = self.moe_layer(x)
        
        # Project to vocabulary
        logits = self.output(x)
        
        return logits, gate_probs


# Example usage
if __name__ == "__main__":
    # Create model
    model = SimpleMoEModel(
        vocab_size=10000,
        d_model=512,
        d_ff=2048,
        num_experts=8,
        k=2
    )
    
    # Forward pass
    input_ids = torch.randint(0, 10000, (2, 128))  # (batch_size=2, seq_len=128)
    logits, gate_probs = model(input_ids)
    
    # Compute load balancing loss
    lb_loss = load_balancing_loss(gate_probs, num_experts=8)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Load balancing loss: {lb_loss.item():.4f}")
