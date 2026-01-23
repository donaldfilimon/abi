#!/usr/bin/env python3
"""
ABI Framework LLM Inference Example

Demonstrates local LLM inference capabilities including
model loading, text generation, and streaming.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi
from abi.llm import LlmEngine, InferenceConfig


def main():
    print("=" * 60)
    print("ABI Framework LLM Inference Example")
    print("=" * 60)

    # Initialize
    abi.init()
    print(f"\nABI version: {abi.version()}")

    # Create LLM engine with custom configuration
    print("\n1. Creating LLM Engine")
    print("-" * 40)

    config = InferenceConfig(
        max_context_length=2048,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        use_gpu=True,
    )

    engine = LlmEngine(config=config)
    print(f"   Configuration:")
    print(f"      Max context: {config.max_context_length}")
    print(f"      Temperature: {config.temperature}")
    print(f"      Top-p: {config.top_p}")
    print(f"      Top-k: {config.top_k}")

    # Note: In development mode without a real model,
    # we'll simulate model loading
    print("\n2. Model Loading (Simulated)")
    print("-" * 40)

    # In production, you would use a real model path:
    # engine.load_model("./models/llama-7b-q4.gguf")

    # For this example, we'll create a mock model path
    mock_model_path = "mock_model.gguf"

    # Create a temporary mock file for demonstration
    with open(mock_model_path, "w") as f:
        f.write("mock")

    try:
        model_info = engine.load_model(mock_model_path)
        print(f"   Model loaded: {model_info.name}")
        print(f"   Architecture: {model_info.architecture}")
        print(f"   Context length: {model_info.context_length}")
        print(f"   Quantization: {model_info.quantization}")
    finally:
        # Clean up mock file
        if os.path.exists(mock_model_path):
            os.remove(mock_model_path)

    # Text generation
    print("\n3. Text Generation")
    print("-" * 40)

    prompt = "The quick brown fox"
    print(f"   Prompt: {prompt}")

    response = engine.generate(prompt, max_tokens=50)
    print(f"   Response: {response}")

    # Streaming generation
    print("\n4. Streaming Generation")
    print("-" * 40)

    prompt = "Once upon a time"
    print(f"   Prompt: {prompt}")
    print("   Streaming: ", end="")

    for token in engine.generate_streaming(prompt, max_tokens=20):
        print(token, end="", flush=True)
    print()

    # Tokenization
    print("\n5. Tokenization")
    print("-" * 40)

    text = "Hello, how are you today?"
    tokens = engine.tokenize(text)
    print(f"   Text: {text}")
    print(f"   Tokens: {tokens}")
    print(f"   Token count: {len(tokens)}")

    # Approximate token counting
    count = engine.count_tokens(text)
    print(f"   Approximate count: {count}")

    # Chat interface
    print("\n6. Chat Interface")
    print("-" * 40)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = engine.chat(messages)
    print(f"   Messages: {len(messages)}")
    print(f"   Response: {response}")

    # Inference statistics
    print("\n7. Inference Statistics")
    print("-" * 40)

    stats = engine.stats
    print(f"   {stats}")
    print(f"   Total time: {stats.total_time_seconds:.4f}s")

    # Using different sampling configs
    print("\n8. Different Sampling Configurations")
    print("-" * 40)

    # Greedy sampling
    greedy_config = InferenceConfig.greedy()
    print(f"   Greedy: temp={greedy_config.temperature}, top_k={greedy_config.top_k}")

    # Creative sampling
    creative_config = InferenceConfig.creative()
    print(f"   Creative: temp={creative_config.temperature}, top_p={creative_config.top_p}")

    # Using LlmContext for framework integration
    print("\n9. Using LlmContext")
    print("-" * 40)

    from abi.config import LlmConfig

    llm_config = LlmConfig(
        context_size=4096,
        use_gpu=True,
        batch_size=512,
    )

    ctx = abi.LlmContext(llm_config)
    print(f"   LlmContext created with config:")
    print(f"      Context size: {llm_config.context_size}")
    print(f"      Use GPU: {llm_config.use_gpu}")

    # Cleanup
    print("\n10. Cleanup")
    print("-" * 40)
    engine.unload_model()
    ctx.close()
    abi.shutdown()
    print("   Resources released")

    print("\n" + "=" * 60)
    print("LLM Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
