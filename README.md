# IoT Sensor Interpretation using LLM Fine-Tuning (Agentic AI Demo)

This project demonstrates how to fine-tune a pretrained Large Language Model (LLM)
(e.g. TinyLLaMA / LLaMA-style models) to **interpret IoT sensor data** and generate
**human-readable actions or explanations**.

The goal is to show how numeric sensor readings can be transformed into text,
tokenized, and used to adapt an LLM for **Agentic AI decision-making**.

---

## 1. Motivation

Traditional ML or RL systems operate on numeric vectors and output numeric actions.
While powerful, they often lack:
- Interpretability
- Explainability
- High-level reasoning

LLMs can complement these systems by:
- Translating low-level sensor data into **semantic understanding**
- Providing **explanations and recommendations**
- Acting as a **reasoning or planning component** in Agentic AI systems

---

## 2. High-Level Workflow



┌─────────────────────────────┐
│   Raw Text / Sensor Values  │
│                             │
│ "Node 1: CPU 95%, RAM 85%"  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          Tokenizer          │
│ (Text → Token IDs)          │
│                             │
│ ["Node","1",":","CPU","95"] │
│        ↓                    │
│ [1012, 3489, 12, 492, 95]   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        LLM (TinyLLaMA)      │
│ (Learns patterns from       │
│  token sequences)           │
│                             │
│ Input:  token IDs           │
│ Output: next token IDs      │
└─────────────────────────────┘


┌─────────────────────────────┐
│        Raw IoT Data         │
│                             │
│ CPU: 95%  RAM: 85%          │
│ Temp: 85C Latency: 300ms    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Textual Representation    │
│                             │
│ "Node 1: CPU 95%, RAM 85%,  │
│  Temp 85C, Latency 300ms"   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        Training Pair        │
│                             │
│ Input Text:                 │
│ "Node 1: CPU 95%, RAM 85%"  │
│                             │
│ Output Text:                │
│ "High load detected.        │
│  Shift workload to Node 2." │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Tokenized Input & Output  │
│ (Used for LLM Fine-Tuning)  │
└─────────────────────────────┘


---

## 3. Why Tokenization is Required

LLMs **do not understand raw text or numbers directly**.
They only operate on **tokens**, which are integers.

- The **tokenizer** converts text into token IDs
- The **model** learns patterns from these token IDs

The tokenizer and model must come from the **same pretrained checkpoint**
to ensure token–weight compatibility.

---

## 4. Dataset Format (IoT Example)

Each training example contains:
- `input`: Textual description of sensor readings
- `output`: Desired action or explanation

Example:

```json
{
  "input": "Node 1: CPU 95%, RAM 85%, Temp 85C, Latency 300ms",
  "output": "High load detected at Node 1. Shift workload to Node 2."
}
