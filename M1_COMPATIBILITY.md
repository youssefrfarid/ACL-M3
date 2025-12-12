# M1 Pro Compatibility Fix - README

## Problem Solved

This update fixes OpenMP mutex deadlock issues (`[mutex.cc : 452] RAW: Lock blocking`) that occur when running the FPL codebase on M1 Pro MacBooks.

## Solution

The codebase now uses **cloud-based HuggingFace Inference API** instead of local SentenceTransformer models, eliminating the OpenMP threading conflicts.

## Setup Instructions

### 1. Get Your HuggingFace API Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "FPL Project")
4. Select "Read" access (sufficient for this project)
5. Click "Generate token"
6. Copy the token that starts with `hf_...`

### 2. Create Your Token File

Create a file named `hf.txt` in the project directory:

```bash
cd /Users/yousseframy/Documents/ACL_milestone_3
nano hf.txt
```

Paste your token (just the token, nothing else), save and exit.

**Example `hf.txt` content:**
```
hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456789
```

### 3. Verify Your Setup

Test that everything works:

```bash
python input_processing.py
```

You should see:
- No lock errors
- Connection to HuggingFace API
- A prompt asking for FPL questions

## Usage

### Basic Query Testing (Recommended First)

```bash
python input_processing.py
```

Try asking: "Who scored the most points in 2023-24?"

### Full Graph Retrieval Testing

```bash
python graph_retrieval.py
```

Options:
- **Mode 1 (Baseline)**: Uses Cypher queries only, no embeddings needed
- **Mode 2 (Embedding)**: Uses cloud embeddings (requires token)
- **Mode 3 (Hybrid)**: Uses both (requires token)

Start with Mode 1 for fastest testing!

### Building Embeddings (Optional)

If you want to use embedding-based retrieval (Mode 2 or 3):

```bash
python build_embeddings.py
```

**Note:** This will take longer than local mode due to API calls, but it's M1 compatible!

## Configuration Options

### Switch Between Cloud and Local Mode

In `input_processing.py`, line 82:
```python
USE_CLOUD_EMBEDDINGS = True  # Set to False to use local model (may cause M1 issues)
```

### Default Mode

- **Cloud mode (default)**: M1 compatible, requires internet, slower
- **Local mode**: Faster, but may cause lock errors on M1

## Troubleshooting

### Error: "hf.txt not found"

**Solution:** Create the `hf.txt` file with your token (see Setup step 2)

### Error: "401 Unauthorized"

**Solution:** Your token is invalid or expired. Generate a new one.

### Still getting lock errors?

**Solution:** Make sure `USE_CLOUD_EMBEDDINGS = True` in `input_processing.py`

### API calls are slow

**Expected:** Cloud API has network latency. For production, consider:
- Using local mode on non-M1 machines
- Caching embeddings in Neo4j (already implemented)

## Security

⚠️ **IMPORTANT:** Never commit `hf.txt` to version control!

Add to your `.gitignore`:
```
hf.txt
```

## What Changed

### Files Modified:
1. **input_processing.py**: Added cloud API support with automatic fallback
2. **graph_retrieval.py**: Updated embedding builder with cloud/local options
3. **build_embeddings.py**: Now uses cloud mode by default

### New Files:
1. **hf.txt.template**: Template for your token file
2. **M1_COMPATIBILITY.md**: This file

## Testing Checklist

- [ ] Created `hf.txt` with your token
- [ ] Ran `python input_processing.py` without lock errors
- [ ] Tested a basic query successfully
- [ ] Ran `python graph_retrieval.py` (Mode 1) successfully

## Questions?

The implementation follows the same approach used in Lab8 to solve identical M1 issues.
