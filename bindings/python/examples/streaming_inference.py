"""
ABI Framework - Streaming Inference Example

Demonstrates streaming LLM inference with real-time token generation.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi
from abi.llm import LlmEngine, InferenceConfig


def main():
    """Run streaming inference example."""
    print("ABI Streaming Inference Example")
    print("=" * 40)

    # Initialize framework
    abi.init()

    # Create LLM engine with configuration
    config = InferenceConfig(
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        streaming=True,
    )
    engine = LlmEngine(config)

    # For demo purposes, we simulate model loading
    # In production, you would load a real GGUF model:
    # engine.load_model("./models/llama-7b-q4.gguf")
    print("\nNote: Using mock model for demonstration")
    engine._model = {"loaded": True}  # Simulate loaded model
    engine._model_info = abi.llm.ModelInfo(name="demo-model")

    # Example prompt
    prompt = "Explain the benefits of GPU acceleration for machine learning"
    print(f"\nPrompt: {prompt}")
    print("\nStreaming response:")
    print("-" * 40)

    # Stream tokens one at a time
    full_response = []
    for token in engine.generate_streaming(prompt, max_tokens=30):
        print(token, end="", flush=True)
        full_response.append(token)

    print("\n" + "-" * 40)
    print(f"\nTotal tokens: {len(full_response)}")

    # Stream with callback
    print("\n\nWith callback:")
    print("-" * 40)

    token_count = [0]

    def on_token(token: str):
        token_count[0] += 1

    for token in engine.generate_streaming("What is Python?", max_tokens=20, callback=on_token):
        print(token, end="", flush=True)

    print(f"\n\nCallback received {token_count[0]} tokens")

    # Cleanup
    engine.unload_model()
    abi.shutdown()

    print("\nExample complete!")


if __name__ == "__main__":
    main()
