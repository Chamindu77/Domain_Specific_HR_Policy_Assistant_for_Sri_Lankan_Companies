# 🤖 HR Policy Assistant — QLoRA Fine-Tuning
### Domain-Specific LLM for Sri Lankan HR Policies using TinyLlama-1.1B + QLoRA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-5.x-yellow.svg)](https://huggingface.co/transformers)
[![TRL](https://img.shields.io/badge/TRL-0.29+-green.svg)](https://github.com/huggingface/trl)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Model Comparison](#-model-comparison)
4. [Dataset](#-dataset)
5. [Installation](#-installation)
6. [Quick Start](#-quick-start)
7. [Training Configuration](#-training-configuration)
8. [All Fixes Applied](#-all-fixes-applied)
9. [Evaluation Results](#-evaluation-results)
10. [Project Structure](#-project-structure)
11. [References](#-references)

---

## 🎯 Project Overview

This project fine-tunes **TinyLlama-1.1B-Chat** using **QLoRA (Quantized Low-Rank Adaptation)** to create a domain-specific HR Policy Assistant for Sri Lankan companies.

The assistant answers HR queries based on:
- Sri Lankan Shop and Office Employees Act
- Payment of Gratuity Act No. 12 of 1983
- EPF / ETF contribution regulations
- Industrial Disputes Act
- Company-level HR policies

**Key capabilities:**
- Maternity / Paternity / Casual / Annual / Medical leave entitlements
- EPF & ETF contribution structures
- Gratuity calculations
- Probation and resignation notice periods
- Overtime calculation
- Grievance procedures
- Performance review processes

---

## 🏗 Architecture

```
Input Question
      │
      ▼
┌─────────────────────────────────────────┐
│         TinyLlama-1.1B-Chat             │
│  (Frozen 4-bit NF4 Quantized Weights)   │
│                                         │
│   ┌──────────────────────────────────┐  │
│   │     LoRA Adapters (r=16, α=32)   │  │
│   │   Injected into q_proj + v_proj  │  │
│   │   Trainable: 2.25M / 617M (0.36%)│  │
│   └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
      │
      ▼
  HR Policy Answer
```

**QLoRA Stack:**
```
Full Precision Weights  →  4-bit NF4 Quantization  (bitsandbytes)
                        →  LoRA Adapters injected   (peft)
                        →  SFTTrainer training      (trl)
                        →  Merge & unload           (peft)
                        →  Final merged model       (safetensors)
```

---

## ⚖️ Model Comparison

| | LLaMA-2-7B | **TinyLlama-1.1B** |
|---|---|---|
| **VRAM Required** | ~14 GB | ~4 GB |
| **Colab Free T4** | ❌ Often OOM | ✅ Works perfectly |
| **HF Access** | Meta approval needed | Fully open |
| **Fine-Tuning Time** | ~2–3 hours | ~20–30 minutes |
| **Architecture** | LLaMA-2 based | LLaMA-2 based |
| **Parameters** | 7B | 1.1B |
| **Trainable (LoRA)** | ~8.4M | ~2.25M |

> **Upgrade path**: Change `MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"` on Colab Pro+ A100. All other code stays identical. ✅

---

## 📂 Dataset

### Format
The dataset follows the **chat messages format** used by TinyLlama:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an HR Policy Assistant for Sri Lankan companies. Answer questions accurately based on Sri Lankan labor laws and company HR policies in a formal, professional tone."
    },
    {
      "role": "user",
      "content": "What is the maternity leave entitlement in Sri Lanka?"
    },
    {
      "role": "assistant",
      "content": "Under the Sri Lankan Shop and Office Employees Act, female employees are entitled to 84 days (12 weeks) of paid maternity leave for the first two confinements, and 42 days (6 weeks) for subsequent confinements..."
    }
  ]
}
```

### Topics Covered

| Category | Topics |
|---|---|
| **Leave** | Maternity, Paternity, Casual, Annual, Medical, Public Holidays |
| **Compensation** | EPF (8%/12%), ETF (3%), Gratuity, Overtime (1.5x / 2x) |
| **Employment** | Probation, Resignation Notice, Termination, PIP |
| **Workplace** | Grievance, Remote Work, Dress Code, Performance Review |
| **Benefits** | Salary Advances, Leave Carry-Forward, Leave Encashment |

### Split
```
Total Dataset  →  80% Train / 10% Validation / 10% Test
```

---

## ⚙️ Installation

### Google Colab (Recommended)
Open the notebook in Colab and run Step 2 (install cell):

```bash
pip install -U transformers>=4.41.0
pip install -U peft>=0.10.0
pip install -U trl>=0.8.6
pip install -U bitsandbytes>=0.41.3
pip install -U accelerate>=0.29.3
pip install -U datasets>=2.19.0
pip install evaluate rouge_score
```

> ⚠️ **Restart runtime** after installation before running any other cells.

### Local Installation
```bash
git clone https://github.com/Chamindu77/HR-Policy-Assistant-QLoRA-TinyLlama.git
cd HR-Policy-Assistant-QLoRA-TinyLlama
pip install -r requirements.txt
```

### requirements.txt
```
transformers>=4.41.0
peft>=0.10.0
trl>=0.8.6
bitsandbytes>=0.41.3
accelerate>=0.29.3
datasets>=2.19.0
evaluate>=0.4.0
rouge_score>=0.1.2
torch>=2.0.0
matplotlib>=3.7.0
```

---

## 🚀 Quick Start

### 1. Open Notebook in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chamindu77/HR-Policy-Assistant-QLoRA-TinyLlama/blob/main/notebooks/HR_Policy_Assistant_FineTuning_v2.ipynb)

### 2. Set Runtime
```
Runtime → Change runtime type → T4 GPU (Free)
```

### 3. Run All Cells in Order

| Step | Action |
|---|---|
| Step 1 | Check GPU with `nvidia-smi` |
| Step 2 | Install packages → **Restart runtime** |
| Step 3 | Verify all imports |
| Step 4 | Upload `hr_policy_dataset.json` |
| Step 5 | Validate + prepare dataset |
| Step 6 | Load clean base model (4-bit NF4) |
| Step 7 | Configure LoRA adapters |
| Step 8 | Configure SFTConfig |
| Step 9 | Train with SFTTrainer |
| Step 10 | Plot loss curves |
| Step 11 | Merge & save model |
| Step 12 | ROUGE evaluation on test set |
| Step 13 | Baseline vs fine-tuned comparison |
| Step 14 | Interactive inference test |
| Step 15 | Save to Google Drive (optional) |
| Step 16 | Upload to HuggingFace Hub (optional) |

### 4. Inference Example

```python
def generate_answer(question, max_new_tokens=250):
    prompt = (
        "<|system|>\n"
        "You are an HR Policy Assistant for Sri Lankan companies. "
        "Answer questions accurately based on Sri Lankan labor laws and company HR policies "
        "in a formal, professional tone.</s>\n"
        f"<|user|>\n{question}</s>\n"
        "<|assistant|>\n"
    )
    inputs    = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = merged_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Test
answer = generate_answer("What is the maternity leave entitlement in Sri Lanka?")
print(answer)
# → "Under the Sri Lankan Shop and Office Employees Act, female employees are
#    entitled to 84 days (12 weeks) of paid maternity leave..."
```

---

## 🔧 Training Configuration

### LoRA Config
```python
LoraConfig(
    r=16,                                 # LoRA rank
    lora_alpha=32,                        # Scaling factor (2x rank)
    target_modules=["q_proj", "v_proj"],  # Attention projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### SFTConfig
```python
SFTConfig(
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,        # Effective batch = 8
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    bf16=True,
    max_length=512,
    eval_strategy="epoch",
    load_best_model_at_end=True,
)
```

### Full Parameter Summary

| Config | Value |
|---|---|
| Base Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Quantization | 4-bit NF4 (double quant) |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Trainable params | ~2.25M (0.36% of total) |
| Epochs | 5 |
| Learning rate | 1e-4 |
| Effective batch size | 8 (2 × 4 grad accumulation) |
| Optimizer | paged_adamw_8bit |
| LR schedule | Cosine |
| Warmup steps | 20 |
| Max sequence length | 512 tokens |

---

## 🐛 All Fixes Applied

All API breaking changes from transformers / trl / peft upgrades are fully resolved:

| Error | Root Cause | Fix Applied |
|---|---|---|
| `triton.ops` ModuleNotFoundError | bitsandbytes 0.43+ / triton 2.x incompatibility | Use `bitsandbytes>=0.41.3` |
| `evaluation_strategy` unexpected kwarg | Renamed in transformers 4.45+ | → `eval_strategy` |
| `group_by_length` unexpected kwarg | Removed in transformers 4.46+ | Removed entirely |
| `warmup_ratio` deprecated warning | Deprecated in trl 0.9+ | → `warmup_steps=20` |
| `dataset_text_field` in SFTTrainer | Moved to SFTConfig in trl 0.9+ | Moved into `SFTConfig` |
| `max_seq_length` unexpected kwarg | Renamed in trl 0.9+ | → `max_length` |
| `tokenizer` unexpected kwarg | Renamed in trl 0.9+ | → `processing_class` |
| `PeftModel + peft_config` conflict | Double LoRA application | Load clean model, let SFTTrainer apply LoRA |
| `max_new_tokens + max_length` conflict | Pipeline uses both simultaneously | Use `.generate()` directly, decode new tokens only |

---

## 📊 Evaluation Results

Evaluation is run on the held-out **test set** (10% of dataset) using ROUGE metrics:

| Metric | Description |
|---|---|
| **ROUGE-1** | Unigram overlap between prediction and reference |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |
| **ROUGE-Lsum** | ROUGE-L at summary level |

Run Step 12 in the notebook after training to compute your scores.

---

## 📁 Project Structure

```
HR-Policy-Assistant-QLoRA-TinyLlama/
│
├── README.md                                        ← This file
├── requirements.txt                                 ← Python dependencies
├── LICENSE
│
├── data/
│   └── hr_policy_dataset.json                       ← HR Q&A pairs dataset
│
├── notebooks/
│   └── HR_Policy_Assistant_FineTuning_v2.ipynb      ← Main Colab notebook
│
├── src/
│   ├── train.py                                     ← Standalone training script
│   ├── inference.py                                 ← Inference script
│   └── evaluate.py                                  ← ROUGE evaluation script
│
└── outputs/
    └── training_loss.png                            ← Loss curve (generated after training)
```

---

## 📚 References

| Resource | Link |
|---|---|
| QLoRA Paper (Dettmers et al., 2023) | https://arxiv.org/abs/2305.14314 |
| TinyLlama Model | https://github.com/jzhang38/TinyLlama |
| TinyLlama on HuggingFace | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| PEFT Library | https://github.com/huggingface/peft |
| TRL Library | https://github.com/huggingface/trl |
| BitsAndBytes | https://github.com/bitsandbytes-foundation/bitsandbytes |
| HuggingFace Transformers | https://github.com/huggingface/transformers |
| Original QLoRA Repo | https://github.com/artidoro/qlora |
| LLaMA Factory | https://github.com/hiyouga/LLaMA-Factory |

---

## 👤 Author

**Chamindu Nipun**  
Machine Learning Engineer  
GitHub: [@Chamindu77](https://github.com/Chamindu77)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built for the ML Engineer Technical Assessment — Domain-Specific LLM Fine-Tuning*
