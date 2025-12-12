# FPL RAG System with LLM Layer (Task 3)

## Overview

Complete Retrieval-Augmented Generation (RAG) system for Fantasy Premier League queries with natural language answers powered by multiple LLM models.

## Architecture

```
User Question
    ↓
Intent Classification & Entity Extraction
    ↓
Hybrid Retrieval (Cypher Queries + Vector Embeddings)
    ↓
Context Combination & Deduplication
    ↓
Structured Prompt (Persona + Context + Task)
    ↓
Multi-Model LLM Generation
    ↓
Natural Language Answer
```

## New Files (Task 3)

1. **`llm_layer.py`** - Core LLM functionality
   - Context combination from baseline + embedding results
   - Structured prompt construction
   - Multi-model interface (Gemma, Mistral, Phi-3)

2. **`fpl_chatbot.py`** - Interactive chatbot application
   - Full end-to-end RAG pipeline
   - Model selection
   - Context display toggle

3. **`evaluate_llms.py`** - Model evaluation framework
   - Runs 15 test questions through all 3 models
   - Quantitative metrics (time, tokens, length)
   - Qualitative scoring template

4. **`test_questions.json`** - Test dataset
   - 15 diverse FPL questions
   - Covers all intent types + edge cases

## Usage

### Quick Test (Standalone LLM Layer)

```bash
python llm_layer.py
```

Tests context combination, prompt building, and answer generation with mock data.

### Interactive Chatbot

```bash
python fpl_chatbot.py
```

**Features:**
- Choose from 3 LLM models
- Ask any FPL question
- Optional context display
- Commands: `model`, `context`, `exit`

**Example session:**
```
You: Who scored the most points in 2023-24?
Intent: player_info
Retrieved: 10 baseline players, 10 embedding players
Generating answer with GEMMA...

Answer: Erling Haaland scored the most points with 238 total points...
(Response time: 1.5s, Model: GEMMA)
```

### Model Evaluation

```bash
python evaluate_llms.py
```

Runs full evaluation:
1. Tests all 3 models on all 15 questions
2. Collects quantitative metrics
3. Generates `evaluation_results.csv` and `evaluation_report.md`
4. Provides qualitative scoring template

**Time estimate**: 10-20 minutes

## LLM Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **Gemma 2B** | 2B | ⚡ Fast | ⭐⭐⭐ | Quick responses |
| **Mistral 7B** | 7B | 🐢 Slower | ⭐⭐⭐⭐⭐ | High-quality answers |
| **Phi-3 Mini** | 3.8B | ⚡ Balanced | ⭐⭐⭐⭐ | Best overall |

All models use **HuggingFace Inference API** (free, M1 compatible).

## Prompt Structure

### Persona
```
You are an expert Fantasy Premier League (FPL) assistant with deep 
knowledge of player statistics, team performance, and FPL strategy.
```

### Context Format
```
=== PLAYERS ===
1. Erling Haaland (FWD)
   Points: 238, Goals: 36, Assists: 12, Form: 8.2
2. Mohamed Salah (MID)
   Points: 211, Goals: 18, Assists: 13, Form: 7.5
...
```

### Task Instructions
```
- Answer using ONLY the provided context
- Cite specific stats when available
- If information is missing, say so
- DO NOT hallucinate
```

## Requirements

- Python 3.8+
- Neo4j database (running with FPL data)
- HuggingFace API token (in `hf.txt`)
- Dependencies: `neo4j`, `huggingface_hub`, `sentence-transformers` (for cloud embeddings)

## Testing Checklist

- [x] `llm_layer.py` standalone test
- [ ] `fpl_chatbot.py` interactive mode
- [ ] `evaluate_llms.py` full evaluation
- [ ] Qualitative scoring (manual)
- [ ] Final comparison report

## Evaluation Metrics

### Quantitative (Automatic)
- Response time (seconds)
- Token count
- Answer length (words)
- Success rate

### Qualitative (Manual, 1-5 scale)
- **Accuracy**: Factually correct?
- **Relevance**: Addresses question?
- **Naturalness**: Human-like language?
- **Completeness**: Uses context well?

## Example Questions

**Easy:**
- "Who scored the most points in 2023-24?"
- "How many goals did Haaland score?"

**Medium:**
- "Compare Salah and Son's performance"
- "What are Arsenal's fixtures in gameweek 5-7?"

**Hard:**
- "Recommend a midfielder under 8 million with good form"
- "Which defender should I buy from a strong defensive team?"

**Edge Cases:**
- "Who is the best player?" (vague)
- "Tell me about Mbappe" (not in dataset - tests hallucination)

## Project Structure

```
ACL_milestone_3/
├── input_processing.py        # Intent & entity extraction
├── graph_retrieval.py         # Baseline + embedding retrieval
├── llm_layer.py              # ✨ Context + prompts + LLM
├── fpl_chatbot.py            # ✨ Interactive application
├── evaluate_llms.py          # ✨ Evaluation framework
├── test_questions.json       # ✨ Test dataset
├── hf.txt                    # HuggingFace token
├── config.txt                # Neo4j credentials
└── fpl_two_seasons.csv       # FPL data
```

## Task 3 Requirements ✅

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Combine baseline + embeddings | ✅ | `combine_contexts()` in `llm_layer.py` |
| Structured prompts | ✅ | Persona + Context + Task template |
| 3+ LLM models | ✅ | Gemma, Mistral, Phi-3 via HF API |
| Quantitative metrics | ✅ | Time, tokens, length in `evaluate_llms.py` |
| Qualitative metrics | ✅ | Manual scoring template in report |
| Model comparison | ✅ | CSV + Markdown report generator |

## Next Steps

1. Run `python fpl_chatbot.py` to test interactively
2. Run `python evaluate_llms.py` for full evaluation
3. Complete manual qualitative scoring
4. Generate final comparison report
5. Document findings and recommendations
