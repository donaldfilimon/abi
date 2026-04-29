# ABI Development Todo

- [ ] Task 2: Implement Slash Command Handlers
  - [ ] Define interaction dispatcher.
  - [ ] Implement /mood, /stats, /clear, /chat.
  - [ ] Add integration tests in 'test_commands.zig'.
- [ ] Task 3: Finalize Voice Gateway & Memory Integration
  - [ ] Connect/Disconnect voice gateway handlers.
  - [ ] Implement WDBX context retrieval in 'AbbeyEngine'.
  - [ ] Inject retrieved context into LLM inference prompt.
  - [ ] Run and pass 'test_inference_pipeline.zig'.
- [ ] Task 4: Final Verification
  - [ ] Run full build check ('./build.sh check').
  - [ ] Verify test parity ('zig build check-parity').
