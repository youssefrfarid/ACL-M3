"""
FPL Graph-RAG Streamlit UI

Interactive web interface for the Fantasy Premier League Knowledge Graph RAG system.
Features:
- Model selection (Gemma, Mistral, Phi-3)
- Retrieval method comparison (Baseline, Embeddings, Hybrid)
- KG context transparency
- Cypher query visualization
- Graph visualization
- Recommendations with explanations

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Any, List, Optional
import time

# Import existing backend modules
from input_processing import (
    classify_intent,
    extract_entities,
    load_known_players_and_teams,
    get_query_embedding,
    load_hf_token
)
from graph_retrieval import retrieve_hybrid
from llm_layer import generate_fpl_answer


# =====================  CONFIGURATION  =====================

st.set_page_config(
    page_title="FPL Graph-RAG System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available models and retrieval methods
AVAILABLE_MODELS = {
    "Gemma 2B (Fast)": "gemma",
    "Mistral 7B (High Quality)": "mistral",
    "Phi-3 Mini (Balanced)": "phi3"
}

RETRIEVAL_METHODS = {
    "Baseline Only": "baseline",
    "Embeddings Only": "embeddings",
    "Hybrid (Both)": "hybrid"
}

# Example questions
EXAMPLE_QUESTIONS = [
    "Who scored the most points in 2023-24?",
    "Compare Salah and Haaland",
    "Recommend a midfielder under 8 million",
    "Which team has the best defense?",
    "What are Arsenal's fixtures in gameweek 5-7?",
    "Top 5 defenders by points",
]


# =====================  GRAPH VISUALIZATION  =====================

def create_knowledge_graph(hybrid_results: Dict[str, Any]) -> nx.Graph:
    """
    Build a NetworkX graph from retrieval results.
    
    Args:
        hybrid_results: Output from retrieve_hybrid()
        
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    
    # Get players from both baseline and embeddings
    baseline_players = hybrid_results.get("baseline_players", [])
    embedding_players = hybrid_results.get("embedding_players", [])
    
    # Combine and deduplicate players
    all_players = {}
    for player in baseline_players + embedding_players:
        player_id = player.get("player_name", "") + player.get("player_element", "")
        if player_id not in all_players:
            all_players[player_id] = player
    
    # Add player nodes
    for player_id, player in all_players.items():
        player_name = player.get("player_name", "Unknown")
        position = player.get("position", "UNK")
        team = player.get("team", "Unknown")
        total_points = player.get("total_points", 0)
        
        G.add_node(
            player_name,
            node_type="player",
            position=position,
            team=team,
            total_points=total_points
        )
        
        # Add team node if not exists
        if not G.has_node(team):
            G.add_node(team, node_type="team")
        
        # Add edge between player and team
        G.add_edge(player_name, team, relationship="PLAYS_FOR")
    
    return G


def visualize_graph(G: nx.Graph) -> go.Figure:
    """
    Create an interactive Plotly visualization of the knowledge graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Plotly Figure object
    """
    if len(G.nodes()) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare edge traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Prepare node traces (separate for players and teams)
    player_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'player']
    team_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'team']
    
    # Position colors based on position
    position_colors = {
        'GK': '#FFA500',  # Orange
        'DEF': '#4169E1',  # Blue
        'MID': '#32CD32',  # Green
        'FWD': '#FF4500',  # Red
        'UNK': '#808080'   # Gray
    }
    
    # Player node trace
    player_x = [pos[node][0] for node in player_nodes]
    player_y = [pos[node][1] for node in player_nodes]
    player_colors = [position_colors.get(G.nodes[node].get('position', 'UNK'), '#808080') for node in player_nodes]
    player_text = [
        f"{node}<br>Position: {G.nodes[node].get('position', 'N/A')}<br>"
        f"Team: {G.nodes[node].get('team', 'N/A')}<br>"
        f"Points: {G.nodes[node].get('total_points', 'N/A')}"
        for node in player_nodes
    ]
    
    player_trace = go.Scatter(
        x=player_x,
        y=player_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=player_text,
        text=[node.split()[-1] for node in player_nodes],  # Show last name
        textposition="top center",
        marker=dict(
            size=20,
            color=player_colors,
            line=dict(width=2, color='white')
        ),
        name='Players'
    )
    
    # Team node trace
    team_x = [pos[node][0] for node in team_nodes]
    team_y = [pos[node][1] for node in team_nodes]
    
    team_trace = go.Scatter(
        x=team_x,
        y=team_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=[f"Team: {node}" for node in team_nodes],
        text=team_nodes,
        textposition="bottom center",
        marker=dict(
            size=30,
            color='#FFD700',  # Gold
            symbol='square',
            line=dict(width=2, color='white')
        ),
        name='Teams'
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, player_trace, team_trace],
        layout=go.Layout(
            title='Knowledge Graph Visualization',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.9)',
            height=500
        )
    )
    
    return fig


# =====================  HELPER FUNCTIONS  =====================

def format_player_dataframe(players: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert player list to formatted DataFrame."""
    if not players:
        return pd.DataFrame()
    
    # Select relevant fields
    df_data = []
    for p in players:
        row = {
            "Player": p.get("player_name", "Unknown"),
            "Team": p.get("team", "N/A"),
            "Position": p.get("position", "N/A"),
            "Points": p.get("total_points", 0),
            "Goals": p.get("total_goals", 0),
            "Assists": p.get("total_assists", 0),
        }
        
        # Add similarity score if available (from embeddings)
        if "similarity_score" in p:
            row["Similarity"] = f"{p['similarity_score']:.3f}"
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def extract_cypher_queries(hybrid_results: Dict[str, Any]) -> List[str]:
    """Extract Cypher queries from retrieval results."""
    queries = []
    
    # Check if baseline results have cypher_query field
    baseline_meta = hybrid_results.get("baseline_metadata", {})
    if "cypher_query" in baseline_meta:
        queries.append(baseline_meta["cypher_query"])
    
    # Check for query in baseline_players metadata
    if hybrid_results.get("baseline_players"):
        # The actual query might be embedded in the retrieval logic
        # For now, we'll note that queries were executed
        queries.append("# Baseline Cypher query executed (see graph_retrieval.py for details)")
    
    return queries


# =====================  STREAMLIT APP  =====================

def main():
    """Main Streamlit application."""
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">⚽ FPL Graph-RAG System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Knowledge Graph-powered Fantasy Premier League Assistant</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'players' not in st.session_state:
        st.session_state.players = None
    if 'teams' not in st.session_state:
        st.session_state.teams = None
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_model_name = st.selectbox(
            "Select LLM Model",
            options=list(AVAILABLE_MODELS.keys()),
            help="Choose which language model to use for generating answers"
        )
        model_name = AVAILABLE_MODELS[selected_model_name]
        
        # Retrieval method
        selected_retrieval_name = st.selectbox(
            "Retrieval Method",
            options=list(RETRIEVAL_METHODS.keys()),
            help="Choose how to retrieve information from the knowledge graph"
        )
        retrieval_method = RETRIEVAL_METHODS[selected_retrieval_name]
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.current_question = example
        
        st.divider()
        
        # Query history
        if st.session_state.history:
            st.header("📜 History")
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"Q: {item['question'][:40]}..."):
                    st.write(f"**Model:** {item['model']}")
                    st.write(f"**Method:** {item['method']}")
                    st.write(f"**Time:** {item['time']:.2f}s")
    
    # Load data on first run
    if st.session_state.players is None:
        with st.spinner("Loading FPL data from Neo4j..."):
            try:
                st.session_state.hf_token = load_hf_token()
                if not st.session_state.hf_token:
                    st.error("❌ HuggingFace token not found. Please create hf.txt with your token.")
                    st.stop()
                
                players, teams = load_known_players_and_teams()
                st.session_state.players = players
                st.session_state.teams = teams
                st.success(f"✅ Loaded {len(players)} players and {len(teams)} teams")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                st.info("Make sure Neo4j is running and config.txt is configured.")
                st.stop()
    
    # Main question input
    st.header("🤔 Ask a Question")
    
    # Use session state for question if set by example button
    default_question = st.session_state.get('current_question', '')
    if default_question:
        question = st.text_area(
            "Enter your FPL question:",
            value=default_question,
            height=100,
            key="question_input"
        )
        # Clear the current_question after using it
        st.session_state.current_question = ''
    else:
        question = st.text_area(
            "Enter your FPL question:",
            height=100,
            placeholder="e.g., Who scored the most points in 2023-24?",
            key="question_input"
        )
    
    # Ask button
    if st.button("🔍 Ask Question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a question first")
        else:
            process_question(
                question=question,
                model_name=model_name,
                retrieval_method=retrieval_method,
                players=st.session_state.players,
                teams=st.session_state.teams,
                hf_token=st.session_state.hf_token
            )


def process_question(question: str, model_name: str, retrieval_method: str, 
                    players: List[str], teams: List[str], hf_token: str):
    """Process a user question through the RAG pipeline."""
    
    start_time = time.time()
    
    # Step 1: Intent classification
    with st.spinner("Classifying intent..."):
        intent = classify_intent(question)
    
    # Step 2: Entity extraction
    with st.spinner("Extracting entities..."):
        entities = extract_entities(question, players, teams)
    
    # Display intent and entities
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Intent:** {intent}")
    with col2:
        st.info(f"**Entities:** {', '.join(entities.player_names + entities.team_names) or 'None'}")
    
    # Step 3: Retrieval
    with st.spinner(f"Retrieving from Knowledge Graph ({retrieval_method})..."):
        query_embedding = get_query_embedding(question)
        
        # Modify retrieval based on method
        if retrieval_method == "baseline":
            # Only baseline
            from graph_retrieval import run_baseline_retrieval
            baseline_result = run_baseline_retrieval(intent, entities)
            hybrid_results = {
                "baseline_players": baseline_result.get("players", []),
                "embedding_players": [],
                "summary": {"baseline_player_count": len(baseline_result.get("players", [])), "embedding_player_count": 0}
            }
        elif retrieval_method == "embeddings":
            # Only embeddings
            from graph_retrieval import run_embedding_retrieval
            embedding_result = run_embedding_retrieval(query_embedding, entities)
            hybrid_results = {
                "baseline_players": [],
                "embedding_players": embedding_result.get("players", []),
                "summary": {"baseline_player_count": 0, "embedding_player_count": len(embedding_result.get("players", []))}
            }
        else:
            # Hybrid
            hybrid_results = retrieve_hybrid(
                intent=intent,
                entities=entities,
                query_embedding=query_embedding,
                index_name="player_embedding_index_minilm",
                top_k=10
            )
    
    # Display retrieval statistics
    summary = hybrid_results.get("summary", {})
    st.success(f"✅ Retrieved: {summary.get('baseline_player_count', 0)} baseline players, "
              f"{summary.get('embedding_player_count', 0)} embedding players")
    
    # Step 4: Display KG context
    st.header("📊 Knowledge Graph Context")
    
    baseline_players = hybrid_results.get("baseline_players", [])
    embedding_players = hybrid_results.get("embedding_players", [])
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Combined View", "Baseline Results", "Embedding Results"])
    
    with tab1:
        all_players = baseline_players + embedding_players
        if all_players:
            df = format_player_dataframe(all_players)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No players retrieved")
    
    with tab2:
        if baseline_players:
            df = format_player_dataframe(baseline_players)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No baseline results (may be using embeddings-only mode)")
    
    with tab3:
        if embedding_players:
            df = format_player_dataframe(embedding_players)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No embedding results (may be using baseline-only mode)")
    
    # Step 5: Show Cypher queries
    with st.expander("🔍 View Cypher Queries"):
        cypher_queries = extract_cypher_queries(hybrid_results)
        if cypher_queries:
            for i, query in enumerate(cypher_queries, 1):
                st.code(query, language="cypher")
        else:
            st.info("Cypher query details not available in current retrieval mode")
    
    # Step 6: Graph visualization
    st.header("🕸️ Knowledge Graph Visualization")
    try:
        G = create_knowledge_graph(hybrid_results)
        if len(G.nodes()) > 0:
            fig = visualize_graph(G)
            st.plotly_chart(fig, use_container_width=True)
            
            # Graph statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", len(G.nodes()))
            with col2:
                st.metric("Edges", len(G.edges()))
            with col3:
                player_count = sum(1 for _, data in G.nodes(data=True) if data.get('node_type') == 'player')
                st.metric("Players", player_count)
        else:
            st.info("No graph data available for visualization")
    except Exception as e:
        st.warning(f"Graph visualization unavailable: {e}")
    
    # Step 7: Generate LLM answer
    st.header("💬 LLM Answer")
    with st.spinner(f"Generating answer with {model_name.upper()}..."):
        try:
            result = generate_fpl_answer(
                question=question,
                hybrid_results=hybrid_results,
                model_name=model_name,
                hf_token=hf_token
            )
            
            if result.get('success'):
                st.success(result['answer'])
                
                # Response metadata
                elapsed_time = time.time() - start_time
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{elapsed_time:.2f}s")
                with col2:
                    st.metric("Model", model_name.upper())
                with col3:
                    st.metric("Tokens", result.get('token_count', 'N/A'))
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'model': model_name,
                    'method': retrieval_method,
                    'time': elapsed_time
                })
            else:
                st.error(f"❌ Error generating answer: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("View error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
