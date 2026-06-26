# TARLP: Task-Adaptive Recoverable Layer Pruning

A structured compression framework for BERT that identifies and removes 
redundant encoder layers using a combined gradient-based and probe-based 
importance criterion — without knowledge distillation.

## What it does
Standard BERT has 12 encoder layers, but not all layers contribute equally 
to every task. TARLP scores each layer using two signals:
- **Gradient importance** — how much each layer influences the task loss
- **Probe importance** — how much task-relevant information is encoded 
  in each layer's [CLS] representation

These are combined (α=0.6, β=0.4) to identify and remove the least 
important layers, followed by a short recovery fine-tuning phase.

## Key Results
| Model | Layers | SST-2 Accuracy | FLOPs |
|-------|--------|----------------|-------|
| BERT (full) | 12 | 92.43% | 100% |
| DistilBERT | 6 | 91.06% | 50% |
| TinyBERT | 4 | 85.09% | 6% |
| **TARLP-25% (ours)** | **9** | **91.09%** | **75%** |
| TARLP-50% (ours) | 6 | 89.56% | 50% |

TARLP-25% matches DistilBERT accuracy without any knowledge distillation.  
Outperforms TinyBERT by 6% (p = 0.001).

## Evaluated On
- SST-2 (sentiment classification)
- MNLI (textual entailment)
- QQP (paraphrase detection)
- MRPC (paraphrase detection)

## Key Findings
- Recovery fine-tuning finds a **new representational solution** rather 
  than restoring original BERT representations (CKA analysis)
- Different GLUE tasks rely on **structurally different encoder layers** 
  (Spearman ρ range: −0.148 to 0.835)
- Lower layers (L1–L5) are task-agnostic; upper layers (L9–L12) 
  specialise per task

## Tech Stack
PyTorch · HuggingFace Transformers · BERT · GLUE Benchmarks · Python

## Authors
Harshitha Pakati V, Dishan D, Nagarjun NH — PES University
