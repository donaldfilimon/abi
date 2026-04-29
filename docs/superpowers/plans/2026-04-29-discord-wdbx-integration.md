# Discord-WDBX Integration Plan

**Goal:** Integrate WDBX vector memory with the Abbey Discord bot for adaptive conversational memory.

**Architecture:** 
1. Initialize a persistent WDBX database handle within the `AbbeyDiscordBot`.
2. Map each Discord `channel_id` or `user_id` to a vector context in WDBX.
3. Inject relevant vector-retrieved context into the inference prompt before sending to the selected local backend.

**Tech Stack:** Zig 0.17-dev, WDBX, local model backends (Ollama/LM Studio).

---

### Task 1: WDBX Integration in Discord Bot

- [ ] **Step 1: Modify `AbbeyDiscordBot` to hold a WDBX handle**
  Modify `abi/src/features/ai/abbey/discord.zig` to include `wdbx_handle: wdbx.DatabaseHandle`.

- [ ] **Step 2: Update `AbbeyDiscordBot.init`**
  Ensure it creates or connects to a persistent database file.

- [ ] **Step 3: Integrate memory in `AbbeyDiscordBot.processMessage`**
  Retrieve relevant context from WDBX using the user's last message vector.

- [ ] **Step 4: Update inference prompt with retrieved context**
  Prepend retrieved context to the prompt before dispatching to local model backend.

---
