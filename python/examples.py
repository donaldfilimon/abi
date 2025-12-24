"""
Comprehensive examples for the ABI Framework Python bindings.
"""

import abi
import numpy as np
import time


def transformer_example():
    """Example: Text encoding with transformer models"""
    print("🚀 Transformer Example")
    print("=" * 50)

    # Create transformer model
    model = abi.Transformer(
        {
            "vocab_size": 30000,
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "max_seq_len": 512,
        }
    )

    # Sample texts
    texts = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "I love programming with Zig and Python!",
        "Machine learning is fascinating.",
        "The ABI framework is very fast.",
    ]

    # Encode texts
    start_time = time.time()
    embeddings = model.encode(texts)
    encode_time = time.time() - start_time

    print(f"Encoded {len(texts)} texts in {encode_time:.4f} seconds")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")
    print()


def vector_database_example():
    """Example: Vector similarity search"""
    print("🔍 Vector Database Example")
    print("=" * 50)

    # Create vector database
    db = abi.VectorDatabase(dimensions=512, distance_metric="cosine")

    # Generate sample vectors (representing documents)
    np.random.seed(42)  # For reproducible results
    num_vectors = 1000
    vectors = np.random.randn(num_vectors, 512).astype(np.float32)

    # Add vectors to database
    ids = [f"doc_{i}" for i in range(num_vectors)]
    start_time = time.time()
    db.add(vectors, ids)
    add_time = time.time() - start_time

    print(f"Added {num_vectors} vectors in {add_time:.4f} seconds")

    # Search for similar vectors
    query_vector = vectors[0]  # Use first vector as query
    start_time = time.time()
    results = db.search(query_vector, top_k=5)
    search_time = time.time() - start_time

    print(f"Search completed in {search_time:.6f} seconds")
    print("Top 5 results:")
    for i, result in enumerate(results):
        print(".4f")
    print()


def reinforcement_learning_example():
    """Example: Reinforcement learning with Q-learning"""
    print("🧠 Reinforcement Learning Example")
    print("=" * 50)

    # Create Q-learning agent
    agent = abi.ReinforcementLearning(
        "q_learning",
        state_size=16,  # 4x4 grid world
        action_count=4,  # up, down, left, right
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
    )

    print("Q-Learning agent created")
    print("Training on a simple grid world...")

    # Simple grid world simulation
    # States: 0-15 (4x4 grid)
    # Actions: 0=up, 1=down, 2=left, 3=right
    # Goal: Reach state 15 from state 0

    episodes = 1000
    max_steps = 50

    for episode in range(episodes):
        state = 0  # Start at top-left
        total_reward = 0

        for step in range(max_steps):
            # Choose action
            action = agent.choose_action([float(state)])  # Convert to list

            # Simulate environment (very simple grid world)
            if action == 0:  # up
                next_state = max(0, state - 4)
            elif action == 1:  # down
                next_state = min(15, state + 4)
            elif action == 2:  # left
                next_state = state if state % 4 == 0 else state - 1
            else:  # right
                next_state = state if state % 4 == 3 else state + 1

            # Calculate reward
            reward = 1.0 if next_state == 15 else -0.1  # Goal reward vs step penalty
            done = next_state == 15

            # Learn from experience
            agent.learn([float(state)], action, reward, [float(next_state)])

            state = next_state
            total_reward += reward

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1:4d}: Total Reward = {total_reward:+.2f}")

    print("\nTraining completed!")
    print("Testing learned policy...")

    # Test the learned policy
    state = 0
    path = [state]
    steps = 0

    while state != 15 and steps < 20:
        action = agent.choose_action([float(state)])
        # Apply action (same logic as above)
        if action == 0:  # up
            next_state = max(0, state - 4)
        elif action == 1:  # down
            next_state = min(15, state + 4)
        elif action == 2:  # left
            next_state = state if state % 4 == 0 else state - 1
        else:  # right
            next_state = state if state % 4 == 3 else state + 1

        state = next_state
        path.append(state)
        steps += 1

    print(f"Path to goal: {path}")
    print(f"Steps taken: {len(path) - 1}")
    print()


def federated_learning_example():
    """Example: Federated learning coordination"""
    print("🌐 Federated Learning Example")
    print("=" * 50)

    # Create federated learning coordinator
    coordinator = abi.FederatedLearningCoordinator(
        {
            "model_size": 10000,  # 10K parameter model
            "rounds": 5,
            "clients_per_round": 3,
        }
    )

    print("Federated learning coordinator created")
    print("Simulating distributed training with 5 clients...")

    # Simulate 5 clients
    client_ids = ["client_1", "client_2", "client_3", "client_4", "client_5"]

    # Register clients
    for client_id in client_ids:
        coordinator.register_client(client_id)

    print(f"Registered {len(client_ids)} clients")

    # Simulate training rounds
    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")

        # Select clients for this round (simulate random selection)
        selected_clients = client_ids[:3]  # First 3 clients

        print(f"Selected clients: {selected_clients}")

        # Simulate clients training and sending updates
        updates = []
        for client_id in selected_clients:
            # Simulate model update (random noise around global model)
            update = {
                "client_id": client_id,
                "model_delta": np.random.randn(10000).astype(np.float32) * 0.01,
                "sample_count": np.random.randint(100, 1000),
            }
            updates.append(update)

        # Aggregate updates
        global_model = coordinator.aggregate_updates(updates)
        print(f"Aggregated updates from {len(updates)} clients")
        print(".2f")

        # Distribute updated model to clients
        coordinator.distribute_model(global_model)

    print("\nFederated learning simulation completed!")
    print()


def gpu_acceleration_example():
    """Example: GPU-accelerated operations"""
    print("🎮 GPU Acceleration Example")
    print("=" * 50)

    # Initialize framework with GPU support
    framework = abi.Framework({"enable_gpu": True, "gpu_backend": "vulkan"})

    print("Framework initialized with GPU support")

    # GPU-accelerated transformer
    model = abi.Transformer(
        {"vocab_size": 30000, "d_model": 512, "n_heads": 8, "n_layers": 6}
    )

    # Large batch processing (will use GPU if available)
    batch_texts = [f"This is test document number {i}." for i in range(100)]

    start_time = time.time()
    embeddings = model.encode(batch_texts)
    gpu_time = time.time() - start_time

    print(
        f"GPU-accelerated encoding of {len(batch_texts)} texts: {gpu_time:.4f} seconds"
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(".2f")
    print()


def performance_comparison():
    """Compare performance of different operations"""
    print("⚡ Performance Comparison")
    print("=" * 50)

    # Initialize framework
    framework = abi.Framework()

    # Test 1: Transformer encoding
    model = abi.Transformer(
        {"vocab_size": 10000, "d_model": 256, "n_heads": 4, "n_layers": 3}
    )

    test_texts = ["Hello world"] * 10

    start_time = time.time()
    for _ in range(100):
        _ = model.encode(test_texts)
    transformer_time = (time.time() - start_time) / 100

    print(".6f")

    # Test 2: Vector database operations
    db = abi.VectorDatabase(dimensions=256)
    vectors = np.random.randn(1000, 256).astype(np.float32)
    db.add(vectors)

    start_time = time.time()
    for _ in range(1000):
        query = np.random.randn(256).astype(np.float32)
        _ = db.search(query, top_k=5)
    search_time = (time.time() - start_time) / 1000

    print(".6f")
    print(".1f")
    print()


def main():
    """Run all examples"""
    print("ABI Framework Python Bindings - Examples")
    print("==========================================")
    print()

    try:
        transformer_example()
        vector_database_example()
        reinforcement_learning_example()
        federated_learning_example()
        gpu_acceleration_example()
        performance_comparison()

        print("🎉 All examples completed successfully!")
        print("\nFor more information, visit: https://abi-framework.org/docs")

    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
