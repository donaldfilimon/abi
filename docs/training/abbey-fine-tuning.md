# Fine-Tuning gpt-oss for Abbey
> **Codebase Status:** Synced with repository as of 2026-01-18.

This guide covers training gpt-oss (OpenAI's open-weight model) as a base for the Abbey AI system using Hugging Face Jobs infrastructure.

## Overview

Abbey is an emotionally intelligent AI framework with advanced cognitive capabilities:
- **14 emotion types** with intensity tracking and response tone adjustment
- **3-tier memory** (episodic, semantic, working)
- **Chain-of-thought reasoning** with confidence calibration
- **Research triggers** when confidence is low
- **Online learning** and meta-learning capabilities

To specialize gpt-oss for Abbey's unique behaviors, we fine-tune on datasets that teach:
1. Emotional intelligence and empathetic responses
2. Reasoning transparency (step-by-step explanations)
3. Confidence awareness (knowing when to say "I'm not sure")
4. Research-first behavior (asking clarifying questions)

## Prerequisites

### Hardware Requirements
| Model | Recommended GPU | Memory | Cost/hr |
|-------|----------------|--------|---------|
| gpt-oss:20b | `a10g-large` | 24GB | ~$5 |
| gpt-oss:20b + LoRA | `l4x1` | 16GB | ~$2.50 |

### Environment Setup
```bash
# Hugging Face CLI authentication
huggingface-cli login

# Verify authentication
hf_whoami()
```

## Training Strategy

### Phase 1: Emotional Intelligence (SFT)

Train on empathetic dialogue datasets to teach Abbey's emotional awareness:

```python
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio", "datasets"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trackio

# EmpatheticDialogues teaches emotional understanding
dataset = load_dataset("facebook/empathetic_dialogues", split="train")

# Format for Abbey's emotion-aware style
def format_empathetic(example):
    emotion = example.get("context", "neutral")
    utterance = example.get("utterance", "")
    response = example.get("response", "")
    return {
        "text": f"[EMOTION: {emotion}]\nUser: {utterance}\nAbbey: {response}"
    }

dataset = dataset.map(format_empathetic)
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model="openai/gpt-oss-20b",  # Base model from HF Hub
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    ),
    args=SFTConfig(
        output_dir="abbey-emotional",
        push_to_hub=True,
        hub_model_id="YOUR_USERNAME/abbey-emotional-v1",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        report_to="trackio",
        project="abbey-training",
        run_name="emotional-intelligence-v1",
    )
)

trainer.train()
trainer.push_to_hub()
```

### Phase 2: Reasoning Transparency (SFT)

Train on chain-of-thought datasets:

```python
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio", "datasets"]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# GSM8K has step-by-step reasoning
dataset = load_dataset("openai/gsm8k", "main", split="train")

def format_reasoning(example):
    question = example["question"]
    answer = example["answer"]  # Contains step-by-step reasoning
    return {
        "text": f"User: {question}\n\nAbbey: Let me think through this step by step.\n{answer}"
    }

dataset = dataset.map(format_reasoning)
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model="YOUR_USERNAME/abbey-emotional-v1",  # Continue from Phase 1
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    peft_config=LoraConfig(r=16, lora_alpha=32),
    args=SFTConfig(
        output_dir="abbey-reasoning",
        push_to_hub=True,
        hub_model_id="YOUR_USERNAME/abbey-reasoning-v1",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        report_to="trackio",
        project="abbey-training",
        run_name="reasoning-transparency-v1",
    )
)

trainer.train()
trainer.push_to_hub()
```

### Phase 3: Confidence Calibration (DPO)

Use Direct Preference Optimization to teach Abbey when to express uncertainty:

```python
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0", "trackio", "datasets"]
# ///

from datasets import Dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# Create synthetic preference data for confidence calibration
# Chosen: Expresses appropriate uncertainty
# Rejected: Overconfident or refuses to engage

confidence_data = [
    {
        "prompt": "What will the stock market do tomorrow?",
        "chosen": "I can't predict specific market movements with certainty. Markets are influenced by many unpredictable factors. However, I can help you understand market analysis techniques or discuss historical patterns if that would be useful.",
        "rejected": "The market will definitely go up tomorrow based on my analysis.",
    },
    {
        "prompt": "Is this code secure?",
        "chosen": "I'd need to examine the code more carefully to give you a confident assessment. I can see [specific observations], but there may be edge cases I'm missing. Would you like me to walk through my analysis?",
        "rejected": "Yes, it's completely secure.",
    },
    # Add more examples...
]

dataset = Dataset.from_list(confidence_data)

trainer = DPOTrainer(
    model="YOUR_USERNAME/abbey-reasoning-v1",
    ref_model=None,  # Uses implicit reference
    train_dataset=dataset,
    peft_config=LoraConfig(r=8, lora_alpha=16),
    args=DPOConfig(
        output_dir="abbey-calibrated",
        push_to_hub=True,
        hub_model_id="YOUR_USERNAME/abbey-calibrated-v1",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,
        report_to="trackio",
        project="abbey-training",
        run_name="confidence-calibration-v1",
    )
)

trainer.train()
trainer.push_to_hub()
```

## Recommended Datasets

### Emotional Intelligence
| Dataset | Purpose | Size |
|---------|---------|------|
| `facebook/empathetic_dialogues` | Emotion-aware dialogue | 25K |
| `daily_dialog` | Conversational emotions | 13K |
| `go_emotions` | Fine-grained emotion classification | 58K |

### Reasoning & Transparency
| Dataset | Purpose | Size |
|---------|---------|------|
| `openai/gsm8k` | Math reasoning with steps | 8.5K |
| `lighteval/MATH` | Advanced math reasoning | 12.5K |
| `hotpot_qa` | Multi-hop reasoning | 113K |

### Confidence & Uncertainty
| Dataset | Purpose | Size |
|---------|---------|------|
| `truthful_qa` | Factual accuracy | 817 |
| Custom DPO data | Uncertainty expression | Build your own |

## Submitting Jobs via Hugging Face

```python
# Submit Phase 1 training job
hf_jobs("uv", {
    "script": "<inline script from Phase 1 above>",
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

## Converting to GGUF for Ollama

After training, convert to GGUF for use with the ABI framework's Ollama integration:

```python
# /// script
# dependencies = ["transformers", "llama-cpp-python", "huggingface_hub"]
# ///

from huggingface_hub import snapshot_download
import subprocess

# Download trained model
model_path = snapshot_download("YOUR_USERNAME/abbey-calibrated-v1")

# Convert to GGUF (requires llama.cpp)
subprocess.run([
    "python", "convert-hf-to-gguf.py",
    model_path,
    "--outfile", "abbey-v1.gguf",
    "--outtype", "q4_k_m"  # 4-bit quantization
])

# Upload GGUF to Hub
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="abbey-v1.gguf",
    path_in_repo="abbey-v1-q4_k_m.gguf",
    repo_id="YOUR_USERNAME/abbey-gguf",
    repo_type="model"
)
```

## Using with ABI Framework

Once trained and converted to GGUF:

```bash
# Pull the fine-tuned model via Ollama
ollama create abbey -f ./Modelfile

# Or use directly with ABI
export ABI_OLLAMA_MODEL=abbey
zig build run -- agent
```

**Modelfile for Ollama:**
```
FROM abbey-v1-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM """
You are Abbey, an emotionally intelligent AI assistant. You:
- Detect and adapt to user emotions
- Think through problems step-by-step
- Express appropriate uncertainty when unsure
- Ask clarifying questions when needed
- Provide direct, honest responses
"""
```

## Training Configuration Reference

### ABI Training Infrastructure

The ABI codebase includes native training support in `src/ai/implementation/training/`:

```zig
const training = @import("abi").ai.training;

// LLM Training Configuration
const config = training.LlmTrainingConfig{
    .epochs = 3,
    .batch_size = 4,
    .max_seq_len = 512,
    .learning_rate = 1e-5,
    .lr_schedule = .warmup_cosine,
    .warmup_steps = 100,
    .optimizer = .adamw,
    .weight_decay = 0.01,
    .grad_accum_steps = 8,
    .checkpoint_interval = 500,
    .checkpoint_path = "./checkpoints",
    .export_gguf_path = "./abbey.gguf",
};
```

### Data Loading

```zig
const data_loader = @import("abi").ai.training.data_loader;

// Load pre-tokenized binary data
var dataset = try data_loader.TokenizedDataset.load(allocator, "training_data.bin");
defer dataset.deinit();

// Create batched iterator with shuffling
var iter = try dataset.batches(allocator, 4, 512, true);
defer iter.deinit();

while (iter.next()) |batch| {
    // batch.input_ids, batch.labels, batch.attention_mask
}
```

### Instruction Tuning Format

```zig
// Parse JSONL instruction data
const samples = try data_loader.parseInstructionDataset(allocator, jsonl_content);

// Each sample has:
// - instruction: The task description
// - input: Optional context
// - output: Expected response
```

## Monitoring Training

Use Trackio for real-time metrics:

```
https://huggingface.co/spaces/YOUR_USERNAME/trackio
```

Key metrics to watch:
- **Loss**: Should decrease steadily
- **Perplexity**: exp(loss), lower is better
- **Learning rate**: Verify warmup and decay
- **Gradient norm**: Should stay under 1.0 with clipping

## Cost Estimation

| Phase | Dataset Size | Hardware | Time | Cost |
|-------|-------------|----------|------|------|
| Emotional Intelligence | 25K samples | a10g-large | ~2h | ~$10 |
| Reasoning | 8.5K samples | a10g-large | ~1h | ~$5 |
| Confidence (DPO) | 1K samples | l4x1 | ~30m | ~$1.25 |
| **Total** | | | ~3.5h | **~$16.25** |

## Next Steps

1. **Collect Abbey-specific data**: Record real conversations to create custom training data
2. **Iterate on confidence**: Build larger DPO datasets for better calibration
3. **Benchmark**: Test Abbey against emotion detection and reasoning benchmarks
4. **Continuous learning**: Use the federated learning infrastructure for ongoing updates

## See Also

- [AI Module Guide](../ai.md) - Full AI module documentation
- [Training CLI](../ai.md#cli-usage) - Training command reference
- [Troubleshooting](../troubleshooting.md) - Common issues
