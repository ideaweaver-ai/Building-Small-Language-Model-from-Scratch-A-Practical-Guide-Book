import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt

# Load Qwen3 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
except:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Load TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")


def process(example):
    ids = tokenizer.encode(example['text'], add_special_tokens=False)
    out = {'ids': ids, 'len': len(ids)}
    return out

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        )
    
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint32 if tokenizer.vocab_size > 65536 else np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

# Training Configuration (needed for get_batch function)
batch_size = 32
block_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Optimized batch loading for Qwen3
def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint32, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint32, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Qwen3 RoPE implementation

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    
    return x_rotated.to(dtype=x.dtype)

# Qwen3 RMSNorm implementation
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float()).to(input_dtype)

# Qwen3 Grouped Query Attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # Qwen3 uses QK normalization
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.scaling = head_dim ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # QK normalization (Qwen3 feature)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V for GQA
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        queries = queries * self.scaling

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

# Qwen3 SwiGLU FeedForward
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, dtype=dtype)

    def forward(self, x):
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# Qwen3 Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, d_ff, dtype=None):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_groups, dtype=dtype)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dtype=dtype)
        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # Pre-norm attention
        attn_out = self.attention(self.norm1(x), mask, cos, sin)
        x = x + attn_out
        
        # Pre-norm feedforward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

# Qwen3 Model
class Qwen3Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_groups, n_layers, d_ff, max_seq_len, dtype=None):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_groups, d_ff, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=1e-6)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)
        
        # RoPE parameters
        head_dim = d_model // n_heads
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            context_length=max_seq_len,
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def _create_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.d_model ** 0.5)
        mask = self._create_causal_mask(seq_len, x.device)

        for block in self.blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Qwen3 Configuration
QWEN3_CONFIG = {
    "vocab_size": 151646,  # Qwen3 vocab size
    "d_model": 640,
    "n_heads": 4,
    "n_kv_groups": 1,  # Same as n_heads for this config
    "n_layers": 18,
    "d_ff": 2048,
    "max_seq_len": 32768,
    "dtype": torch.bfloat16,
}

torch.manual_seed(123)
model = Qwen3Model(**QWEN3_CONFIG)

# Loss estimation function
def estimate_loss(model, eval_iters=200):
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# Additional Training Configuration
learning_rate = 1e-4
max_iters = 150000
warmup_steps = 1000
min_lr = 5e-4
eval_iters = 500
gradient_accumulation_steps = 32
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Training Loop
best_val_loss = float('inf')
best_model_params_path = "qwen3_model.pt"
train_loss_list, validation_loss_list = [], []

model = model.to(device)

for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list += [losses['train']]
        validation_loss_list += [losses['val']]

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)

    X, y = get_batch("train")
    X, y = X.to(device), y.to(device)

    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()

# Plot training and validation loss
train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

plt.plot(train_loss_list_converted, 'g', label='train_loss')
plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
plt.xlabel("Steps - Every 500 epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Load the best model and generate text
model = Qwen3Model(**QWEN3_CONFIG)
device = "cuda" if torch.cuda.is_available() else "cpu"
best_model_params_path = "best_model_params.pt"
model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))

sentence = "Once upon a time there was a pumpkin."
context = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(dim=0)
y = model.generate(context, 200)
print(tokenizer.decode(y.squeeze().tolist()))
