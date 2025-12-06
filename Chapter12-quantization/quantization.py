import bitsandbytes as bnb
import torch

print(f"BitsAndBytes version: {bnb.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Model identifier from Hugging Face
qwen_model_name = "Qwen/Qwen3-0.6B"

# Load full precision model (FP32/FP16)
full_precision_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure 8-bit quantization
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load model with 8-bit quantization
quantized_8bit_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    quantization_config=quantization_config_8bit,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Configure 4-bit quantization with optimizations
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization type
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
    bnb_4bit_use_double_quant=True,  # Use double quantization for better accuracy
)

# Load model with 4-bit quantization
quantized_4bit_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    quantization_config=quantization_config_4bit,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Load tokenizer (same for all model variants)
text_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

# Display memory usage for each model
print("Memory Usage Comparison:")
print(f"Full Precision Model: {full_precision_model.get_memory_footprint():,} bytes")
print(f"8-bit Quantized Model: {quantized_8bit_model.get_memory_footprint():,} bytes")
print(f"4-bit Quantized Model: {quantized_4bit_model.get_memory_footprint():,} bytes")

# Extract weights from the first layer's query projection
first_layer_query_weights = full_precision_model.model.layers[0].self_attn.q_proj.weight.data

print("Full Precision Weights (FP16/FP32):")
print(first_layer_query_weights)
print(f"Shape: {first_layer_query_weights.shape}")
print(f"Data type: {first_layer_query_weights.dtype}")
print("\n" + "-" * 60 + "\n")

# Extract weights from the 8-bit quantized model
quantized_8bit_query_weights = quantized_8bit_model.model.layers[0].self_attn.q_proj.weight.data

print("8-bit Quantized Weights:")
print(quantized_8bit_query_weights)
print(f"Shape: {quantized_8bit_query_weights.shape}")
print(f"Data type: {quantized_8bit_query_weights.dtype}")
print("\n" + "-" * 60 + "\n")

# Extract weights from the 4-bit quantized model
quantized_4bit_query_weights = quantized_4bit_model.model.layers[0].self_attn.q_proj.weight.data

print("4-bit Quantized Weights:")
print(quantized_4bit_query_weights)
print(f"Shape: {quantized_4bit_query_weights.shape}")
print(f"Data type: {quantized_4bit_query_weights.dtype}")

def get_model_output(model, user_message, tokenizer):
    """
    Generate a text response from the model given a user message.

    Args:
        model: The language model (full precision or quantized)
        user_message: The input text prompt
        tokenizer: The tokenizer for encoding/decoding text

    Returns:
        The generated text response
    """
    # Format the message in the chat template format
    conversation = [{"role": "user", "content": user_message}]

    # Apply the chat template (adds system prompts, formatting, etc.)
    formatted_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # Convert text to token IDs and move to GPU
    tokenized_input = text_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # Generate response
    generated_tokens = model.generate(
        **tokenized_input,
        max_new_tokens=256,  # Maximum number of new tokens to generate
        do_sample=True,      # Enable sampling (non-deterministic)
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_k=50,           # Consider top 50 most likely tokens
        top_p=0.95          # Nucleus sampling: consider tokens until cumulative prob reaches 0.95
    )

    # Decode token IDs back to text
    generated_text = text_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_text

# Define a test prompt that requires reasoning
test_prompt = "Explain how neural networks learn from data, including the role of backpropagation and gradient descent."

print("=" * 70)
print("Testing Full Precision Model")
print("=" * 70)
full_precision_output = get_model_output(
    full_precision_model,
    test_prompt,
    text_tokenizer
)
print(full_precision_output)
print("\n" + "-" * 70 + "\n")

print("=" * 70)
print("Testing 8-bit Quantized Model")
print("=" * 70)
quantized_8bit_output = get_model_output(
    quantized_8bit_model,
    test_prompt,
    text_tokenizer
)
print(quantized_8bit_output)
print("\n" + "-" * 70 + "\n")

print("=" * 70)
print("Testing 4-bit Quantized Model")
print("=" * 70)
quantized_4bit_output = get_model_output(
    quantized_4bit_model,
    test_prompt,
    text_tokenizer
)
print(quantized_4bit_output)

def compute_perplexity(model, text_sample, tokenizer):
    """
    Calculate perplexity for a given text sample.

    Perplexity measures how well the model predicts the next token.
    Lower values indicate better predictive performance.

    Args:
        model: The language model to evaluate
        text_sample: The text string to evaluate perplexity on
        tokenizer: The tokenizer for encoding text

    Returns:
        float: The perplexity score
    """
    # Convert text to token IDs
    encoded_text = text_tokenizer(text_sample, return_tensors='pt').to("cuda")

    # Extract input token IDs
    token_ids = encoded_text.input_ids

    # Create target labels (same as input for next-token prediction)
    target_labels = token_ids.clone()

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Forward pass: model predicts next tokens
        model_output = model(token_ids, labels=target_labels)

    # Extract the cross-entropy loss
    cross_entropy_loss = model_output.loss

    # Convert loss to perplexity: perplexity = exp(loss)
    perplexity_score = torch.exp(cross_entropy_loss)

    return perplexity_score.item()

# Define a test text sample for perplexity evaluation
test_text = """
Neural networks are computational models inspired by biological neural networks.
They consist of interconnected nodes organized in layers. Each connection has a weight
that is adjusted during training. The learning process involves forward propagation,
where input data flows through the network, and backpropagation, where errors are
propagated backward to update weights using gradient descent.
"""

print("Calculating Perplexity for All Model Variants")
print("=" * 70)

# Calculate perplexity for full precision model
full_precision_ppl = compute_perplexity(
    full_precision_model,
    test_text,
    text_tokenizer
)
print(f"Full Precision Model Perplexity: {full_precision_ppl:.2f}")

# Calculate perplexity for 8-bit quantized model
quantized_8bit_ppl = compute_perplexity(
    quantized_8bit_model,
    test_text,
    text_tokenizer
)
print(f"8-bit Quantized Model Perplexity: {quantized_8bit_ppl:.2f}")

# Calculate perplexity for 4-bit quantized model
quantized_4bit_ppl = compute_perplexity(
    quantized_4bit_model,
    test_text,
    text_tokenizer
)
print(f"4-bit Quantized Model Perplexity: {quantized_4bit_ppl:.2f}")

# Calculate percentage difference
ppl_increase_8bit = ((quantized_8bit_ppl - full_precision_ppl) / full_precision_ppl) * 100
ppl_increase_4bit = ((quantized_4bit_ppl - full_precision_ppl) / full_precision_ppl) * 100

print("\n" + "-" * 70)
print("Perplexity Comparison:")
print(f"8-bit increase: {ppl_increase_8bit:.2f}%")
print(f"4-bit increase: {ppl_4bit_increase:.2f}%")
