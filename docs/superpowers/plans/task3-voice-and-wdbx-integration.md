Task: Task 3 - Voice Gateway and Memory-Augmented Inference
Files:
- Modify: abi/src/features/ai/abbey/discord.zig
- Modify: abi/src/features/ai/abbey/engine.zig
- Test: abi/src/features/ai/abbey/test_inference_pipeline.zig

- [ ] **Step 1: Integrate Voice Gateway support**
  Implement basic voice channel connection/disconnection handlers in the Discord connector for voice interactions.

- [ ] **Step 2: Link Inference Router to WDBX**
  Modify `AbbeyEngine.processMessage` to perform a WDBX vector search for conversation history/context using the user's current message embedding.

- [ ] **Step 3: Prompt augmentation**
  Inject retrieved vector context as "Memory" into the prompt before the inference model backend (Ollama, etc.) processes it.

- [ ] **Step 4: Verify integration**
  Create `test_inference_pipeline.zig` to verify the full flow: Discord Message -> Embedding -> WDBX Search -> Prompt Injection -> Model Inference.
