# Streamlit UI for FPL Graph-RAG System (Task 4)

## Overview

Interactive web interface for the Fantasy Premier League Knowledge Graph RAG system built with Streamlit.

## Features

### Core Functionality
- ✅ **Model Selection**: Choose between Gemma 2B, Mistral 7B, or Phi-3 Mini
- ✅ **Retrieval Method Comparison**: Test Baseline Only, Embeddings Only, or Hybrid approaches
- ✅ **KG Context Display**: View retrieved knowledge graph data in structured tables
- ✅ **LLM Answers**: Get natural language responses powered by multiple LLMs

### Transparency Features
- 🔍 **Cypher Query Visualization**: See the actual queries executed
- 📊 **Separate Result Views**: Compare baseline vs embedding retrieval results
- 📈 **Retrieval Statistics**: Track number of results from each method

### Advanced Features
- 🕸️ **Interactive Graph Visualization**: Explore the knowledge graph with NetworkX + Plotly
  - Color-coded nodes by player position
  - Hover tooltips with player stats
  - Team nodes and PLAYS_FOR relationships
- 💡 **Example Questions**: Quick-start with pre-configured queries
- 📜 **Query History**: Review past questions in the sidebar
- ⚡ **Real-time Performance Metrics**: See response time and token counts

## Installation

### Prerequisites
- Python 3.8+
- Neo4j database running with FPL data loaded
- HuggingFace API token
- Neo4j credentials configured

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials**:
   - Create `config.txt` with Neo4j credentials:
     ```
     URI=neo4j+s://your-database.databases.neo4j.io
     USERNAME=neo4j
     PASSWORD=your_password
     ```
   - Create `hf.txt` with your HuggingFace token:
     ```
     hf_YourTokenHere
     ```

3. **Ensure Neo4j is running** with FPL data loaded (from previous tasks)

## Usage

### Launch the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Select Configuration** (Sidebar):
   - Choose an LLM model (Gemma for speed, Mistral for quality, Phi-3 for balance)
   - Select retrieval method (Hybrid recommended)

2. **Ask a Question**:
   - Type your question in the text area or click an example question
   - Click "🔍 Ask Question"

3. **Review Results**:
   - **Intent & Entities**: See how the system understood your question
   - **KG Context**: Browse retrieved data in three tabs (Combined, Baseline, Embeddings)
   - **Cypher Queries**: Expand to see executed queries
   - **Graph Visualization**: Explore the knowledge graph interactively
   - **LLM Answer**: Read the natural language response

### Example Workflow

**Query**: "Compare Salah and Haaland"

**Results**:
- Intent: `compare_players`
- Entities: `Salah, Haaland`
- Retrieval: 2 baseline players, 10 embedding players
- Graph: Shows both players connected to their teams
- Answer: "Mohamed Salah scored 211 points with 18 goals and 13 assists, while Erling Haaland scored 238 points with 36 goals..."

## Features Checklist (Task 4 Requirements)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| View KG-retrieved context | ✅ | Tabbed interface with Combined/Baseline/Embedding views |
| View final LLM answer | ✅ | Formatted answer display with metadata |
| Display Cypher queries | ✅ | Expandable section showing executed queries |
| Graph visualization | ✅ | Interactive NetworkX + Plotly graph |
| Recommendations with explanations | ✅ | Shown in context tables with similarity scores |
| Model selection dropdown | ✅ | Sidebar dropdown for 3 LLM models |
| Retrieval method selection | ✅ | Sidebar dropdown for baseline/embeddings/hybrid |

## Architecture

```
User Interface (Streamlit)
    ↓
Question Input
    ↓
Backend Integration:
  - input_processing.py (Intent + Entities)
  - graph_retrieval.py (Baseline/Embeddings/Hybrid)
  - llm_layer.py (Context + Prompt + LLM)
    ↓
Results Display:
  - KG Context Tables
  - Graph Visualization
  - Cypher Queries
  - LLM Answer
```

## Troubleshooting

### "HuggingFace token not found"
- Ensure `hf.txt` exists in the project directory
- Get a token from: https://huggingface.co/settings/tokens

### "Error loading data from Neo4j"
- Verify Neo4j is running
- Check `config.txt` credentials are correct
- Ensure FPL data was loaded in previous tasks

### Graph visualization not showing
- Check that players were successfully retrieved
- Some queries may return no results (e.g., players not in database)

### Slow response times
- Gemma 2B is fastest (~1-2s)
- Mistral 7B is slower but higher quality (~3-5s)
- Embeddings add ~0.5-1s overhead

## Comparison with CLI Chatbot

| Feature | CLI (`fpl_chatbot.py`) | Streamlit UI |
|---------|------------------------|--------------|
| Interface | Terminal | Web browser |
| Model switching | Interactive command | Dropdown |
| Context display | Optional text | Structured tables |
| Graph visualization | ❌ | ✅ |
| Retrieval comparison | ❌ | ✅ |
| Query history | ❌ | ✅ |
| Cypher queries | ❌ | ✅ |

## Next Steps

- Test with various query types
- Compare model performance
- Experiment with different retrieval methods
- Use for Task 3 model evaluation
