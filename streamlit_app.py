"""
FPL Graph-RAG Streamlit UI

Interactive web interface for the Fantasy Premier League Knowledge Graph RAG system.
Features:
- Model selection (Gemma, Mistral, Phi-3)
- Retrieval method comparison (Baseline, Embeddings, Hybrid)
- KG context transparency
- Cypher query visualization
- Graph visualization (Interactive, FPL Themed)
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
from llm_layer import (
    generate_fpl_answer, 
    generate_openrouter_answer,
    load_openrouter_key,
    OpenRouterLLMInterface
)


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

# OpenRouter models (free tier)
OPENROUTER_MODELS = {
    "Qwen3 Coder (Fast)": "qwen3-coder",
    "Llama 3.3 70B (Large)": "llama-3.3-70b",
    "Gemini 2.0 Flash (Balanced)": "gemini-flash"
}

RETRIEVAL_METHODS = {
    "Baseline Only": "baseline",
    "Embeddings Only": "embeddings",
    "Hybrid (Both)": "hybrid"
}

# Example questions
EXAMPLE_QUESTIONS = [
    "Who scored the most points in 2022-23?",
    "Compare Salah and Haaland",
    "Recommend a midfielder under 8 million",
    "Which team has the best defense?",
    "What are Arsenal's fixtures in gameweek 5-7?",
    "Top 5 defenders by points",
]


def set_question(q):
    """Callback to set the question input."""
    st.session_state.question_input = q



def extract_cypher_queries(results: Dict[str, Any]) -> List[str]:
    """Extract Cypher queries from retrieval results."""
    queries = []
    
    # Check baseline results (old metadata structure)
    baseline_meta = results.get("baseline_metadata", {})
    if "cypher_query" in baseline_meta:
        queries.append(baseline_meta["cypher_query"])
        
    # Check direct query result (new structure)
    if "cypher_query" in results:
        queries.append(results["cypher_query"])
    elif "baseline_result" in results and "cypher_query" in results["baseline_result"]:
        queries.append(results["baseline_result"]["cypher_query"])
    
    # Check for query in baseline_players metadata
    if not queries and results.get("baseline_players"):
        # The actual query might be embedded in the retrieval logic
        # For now, we'll note that queries were executed if we missed capturing it
        pass
    
    return queries


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
        # Retrieval returns 'name', not 'player_name'
        player_name = player.get("name", player.get("player_name", ""))
        if player_name and player_name not in all_players:
            all_players[player_name] = player
    
    # Add player nodes
    for player_name, player in all_players.items():
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
        if team != "Unknown" and not G.has_node(team):
            G.add_node(team, node_type="team")
        
        # Add edge between player and team
        if team != "Unknown":
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
    
    # Calculate layout - Use Kamada-Kawai for better separation (Neo4j-like)
    # Fallback to spring if kamada_kawai fails (requires scipy)
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, k=0.5, iterations=100)
    
    # Prepare edge traces with FPL styling
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color='rgba(200, 200, 200, 0.4)'), 
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces (separate for players and teams for legend)
    player_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'player']
    team_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'team']
    
    # Position colors based on FPL official theme
    position_colors = {
        'GK': '#FFD700',   # Gold (Keepers)
        'DEF': '#00FF87',  # Neon Green (Defenders)
        'MID': '#04F5FF',  # Cyan Blue (Midfielders)
        'FWD': '#E90052',  # Magenta Pink (Forwards)
        'UNK': '#9B9B9B'   # Silver Gray
    }
    
    # PLAYER NODES
    player_x = [pos[node][0] for node in player_nodes]
    player_y = [pos[node][1] for node in player_nodes]
    player_colors = [position_colors.get(G.nodes[node].get('position', 'UNK'), '#808080') for node in player_nodes]
    player_text = [
        f"<b>{node}</b><br>Position: {G.nodes[node].get('position', 'N/A')}<br>"
        f"Team: {G.nodes[node].get('team', 'N/A')}<br>"
        f"Points: {G.nodes[node].get('total_points', 'N/A')}"
        for node in player_nodes
    ]
    
    player_trace = go.Scatter(
        x=player_x,
        y=player_y,
        mode='markers', # Text handled separately if needed, or on hover
        hoverinfo='text',
        hovertext=player_text,
        marker=dict(
            size=18,
            color=player_colors,
            line=dict(width=2, color='white'),
            opacity=1.0
        ),
        name='Players'
    )

    # TEAM NODES
    team_x = [pos[node][0] for node in team_nodes]
    team_y = [pos[node][1] for node in team_nodes]
    
    team_trace = go.Scatter(
        x=team_x,
        y=team_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=[f"Team: {node}" for node in team_nodes],
        text=team_nodes,
        textposition="top center",
        textfont=dict(size=12, color='#E90052', family='Arial Black'),
        marker=dict(
            size=25,
            color='#E0E0E0', 
            line=dict(width=3, color='#E90052'),
            symbol='circle'
        ),
        name='Teams'
    )
    
    # Create figure with Neo4j-like Dark Theme
    fig = go.Figure(
        data=[edge_trace, player_trace, team_trace],
        layout=go.Layout(
            title={
                'text': 'Knowledge Graph Context',
                'font': {'size': 20, 'color': '#FFFFFF'},
                'x': 0.05,
                'xanchor': 'left'
            },
            showlegend=True,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="sans-serif",
                font_color="black"
            ),
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#121212',  # Very dark grey (almost black)
            paper_bgcolor='#121212',
            font=dict(color='#FAFAFA'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#FAFAFA')
            ),
            height=600
        )
    )
    
    return fig


# =====================  HELPER FUNCTIONS  =====================

def format_player_dataframe(players: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert player list to formatted DataFrame with flexible field handling."""
    if not players:
        return pd.DataFrame()
    
    # Determine which columns are available from the data
    df_data = []
    for p in players:
        row = {}
        
        # Always try to get name (Player or Team)
        row["Name"] = p.get("name", p.get("player_name", "Unknown"))
        
        # Optional fields - only add if present in data
        if "position" in p and p["position"]:
            row["Position"] = p["position"]
        
        if "team" in p and p["team"]:
            row["Team"] = p["team"]
        
        if "season" in p and p["season"]:
            row["Season"] = p["season"]
        
        # Stats fields - check multiple possible keys
        # Points (could be total_points, points, or value depending on query)
        points = p.get("total_points") or p.get("points") or p.get("value")
        if points is not None:
            row["Points"] = int(points) if points else 0
        
        # Goals (could be goals or goals_scored)
        goals = p.get("goals") or p.get("goals_scored")
        if goals is not None:
            row["Goals"] = int(goals) if goals else 0
        
        if "assists" in p:
            row["Assists"] = int(p["assists"]) if p["assists"] else 0
        
        if "minutes" in p:
            row["Minutes"] = int(p["minutes"]) if p["minutes"] else 0
        
        # New fields for enhancements
        if "price" in p:
            row["Price"] = f"£{float(p['price']):.1f}m"

        if "clean_sheets" in p:
            row["Clean Sheets"] = int(p["clean_sheets"])
            
        if "goals_conceded" in p:
            row["Conceded"] = int(p["goals_conceded"])
        
        if "form" in p and p["form"] is not None:
            row["Form"] = f"{float(p['form']):.1f}"
            
        # Similarity score (from embeddings)
        if "score" in p:
            row["Similarity"] = f"{float(p['score']):.3f}"
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)





# =====================  STREAMLIT APP  =====================

def main():
    """Main Streamlit application."""
    
    # Custom CSS for FPL Theme with Football Pitch Background
    st.markdown("""
        <style>
        /* ==================== FPL COLORS (Toned Down) ==================== */
        /* Primary: #38003C (Dark Purple)
           Accent 1: #2ECC71 (Softer Green)
           Accent 2: #C0392B (Softer Red)
           Accent 3: #3498DB (Softer Blue) */
        
        /* ==================== FOOTBALL PITCH BACKGROUND ==================== */
        .stApp {
            background: 
                /* Center circle */
                radial-gradient(circle at 50% 50%, transparent 8%, rgba(255, 255, 255, 0.08) 8.5%, rgba(255, 255, 255, 0.08) 9%, transparent 9.5%),
                /* Penalty box top */
                linear-gradient(to bottom, transparent 10%, rgba(255, 255, 255, 0.06) 10.2%, rgba(255, 255, 255, 0.06) 10.4%, transparent 10.6%),
                /* Halfway line */
                linear-gradient(to bottom, transparent 49.8%, rgba(255, 255, 255, 0.1) 50%, rgba(255, 255, 255, 0.1) 50.2%, transparent 50.4%),
                /* Penalty box bottom */
                linear-gradient(to bottom, transparent 89.4%, rgba(255, 255, 255, 0.06) 89.6%, rgba(255, 255, 255, 0.06) 89.8%, transparent 90%),
                /* Grass stripe pattern */
                repeating-linear-gradient(
                    90deg,
                    #1a5c2e 0px,
                    #1a5c2e 80px,
                    #166b32 80px,
                    #166b32 160px
                );
            background-attachment: fixed;
        }
        
        /* Main content area */
        .main .block-container {
            background: rgba(20, 60, 30, 0.92);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(8px);
            border: 2px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        /* ==================== HEADER ==================== */
        .main-header {
            font-family: 'Arial Black', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #FFD700, #FFFFFF, #FFD700);
            background-size: 200% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s ease infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .sub-header {
            font-family: 'Arial', sans-serif;
            font-size: 1.2rem;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
            opacity: 0.9;
        }
        
        /* ==================== METRICS (Scoreboard Style) ==================== */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 900;
            color: #FFD700 !important;
            font-family: 'Arial Black', sans-serif;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: bold;
            color: #FFFFFF !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem;
        }
        
        div[data-testid="metric-container"] {
            background: rgba(56, 0, 60, 0.85);
            border: 2px solid rgba(255, 215, 0, 0.5);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-3px);
        }
        
        /* ==================== BUTTONS ==================== */
        .stButton > button {
            background: linear-gradient(135deg, #38003C, #5a0060) !important;
            color: #FFFFFF !important;
            border: 2px solid #FFD700 !important;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.6rem 1.2rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #FFD700, #FFA500) !important;
            color: #38003C !important;
            transform: translateY(-2px);
            border-color: #38003C !important;
        }
        
        /* ==================== TABS ==================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(56, 0, 60, 0.6);
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #FFFFFF;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 215, 0, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FFD700, #FFA500) !important;
            color: #38003C !important;
        }
        
        /* ==================== EXPANDERS ==================== */
        .streamlit-expanderHeader {
            background: rgba(56, 0, 60, 0.8);
            border-radius: 10px;
            font-weight: bold;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 215, 0, 0.4);
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(56, 0, 60, 0.95);
        }
        
        /* ==================== SIDEBAR ==================== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #38003C 0%, #2d0030 50%, #38003C 100%);
            border-right: 3px solid #FFD700;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
            font-family: 'Arial Black', sans-serif;
        }
        
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        /* ==================== DATAFRAMES (League Table Style) ==================== */
        [data-testid="stDataFrame"] {
            border: 2px solid #FFD700;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        
        [data-testid="stDataFrame"] table {
            background-color: rgba(56, 0, 60, 0.9) !important;
        }
        
        [data-testid="stDataFrame"] thead tr {
            background: linear-gradient(135deg, #38003C, #5a0060) !important;
        }
        
        [data-testid="stDataFrame"] thead th {
            color: #FFD700 !important;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 2px solid #FFD700 !important;
        }
        
        [data-testid="stDataFrame"] tbody tr {
            color: #FFFFFF !important;
        }
        
        [data-testid="stDataFrame"] tbody tr:hover {
            background-color: rgba(255, 215, 0, 0.1) !important;
        }
        
        /* ==================== TEXT INPUTS ==================== */
        textarea, input {
            background-color: rgba(56, 0, 60, 0.8) !important;
            color: #FFFFFF !important;
            border: 2px solid rgba(255, 215, 0, 0.4) !important;
            border-radius: 8px !important;
        }
        
        textarea:focus, input:focus {
            border-color: #FFD700 !important;
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.3) !important;
        }
        
        /* ==================== MESSAGES ==================== */
        .stSuccess {
            background-color: rgba(46, 204, 113, 0.2) !important;
            border-left: 4px solid #2ECC71 !important;
            color: #FFFFFF !important;
        }
        
        .stError {
            background-color: rgba(231, 76, 60, 0.2) !important;
            border-left: 4px solid #E74C3C !important;
            color: #FFFFFF !important;
        }
        
        .stInfo {
            background-color: rgba(52, 152, 219, 0.2) !important;
            border-left: 4px solid #3498DB !important;
            color: #FFFFFF !important;
        }
        
        .stWarning {
            background-color: rgba(241, 196, 15, 0.2) !important;
            border-left: 4px solid #F1C40F !important;
            color: #FFFFFF !important;
        }
        
        /* ==================== DIVIDERS ==================== */
        hr {
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, #FFD700, transparent) !important;
            margin: 1.5rem 0 !important;
        }
        
        /* ==================== CODE BLOCKS ==================== */
        code {
            background-color: rgba(56, 0, 60, 0.9) !important;
            color: #3498DB !important;
            border: 1px solid #38003C !important;
            border-radius: 4px;
            padding: 2px 6px;
        }
        
        pre {
            background-color: rgba(56, 0, 60, 0.9) !important;
            border: 2px solid #38003C !important;
            border-radius: 8px;
        }
        
        /* ==================== SCROLLBAR ==================== */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(56, 0, 60, 0.5);
            border-radius: 8px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #38003C, #FFD700);
            border-radius: 8px;
        }
        
        /* ==================== SELECT BOXES ==================== */
        .stSelectbox > div > div {
            background: rgba(56, 0, 60, 0.8) !important;
            border: 2px solid rgba(255, 215, 0, 0.4) !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
        }
        
        /* ==================== GENERAL TEXT ==================== */
        h1, h2, h3, h4, h5, h6 {
            color: #FFD700 !important;
        }
        
        p, span, label {
            color: #FFFFFF !important;
        }
        
        /* ==================== SPINNER ==================== */
        .stSpinner > div {
            border-top-color: #FFD700 !important;
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    
    # Header with football icons
    st.markdown('<p class="main-header">⚽ FPL Graph-RAG System 🏆</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🥅 Your AI-Powered Fantasy Premier League Assistant | Powered by Knowledge Graphs ⚡</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'players' not in st.session_state:
        st.session_state.players = None
    if 'teams' not in st.session_state:
        st.session_state.teams = None
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
    if 'openrouter_key' not in st.session_state:
        st.session_state.openrouter_key = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎮 Match Settings")
        
        # Model provider selection
        model_provider = st.radio(
            "Model Provider",
            options=["OpenRouter (Recommended)", "HuggingFace"],
            help="OpenRouter provides free access to powerful models"
        )
        
        # Model selection based on provider
        if model_provider == "OpenRouter (Recommended)":
            selected_model_name = st.selectbox(
                "Select LLM Model",
                options=list(OPENROUTER_MODELS.keys()),
                help="Choose which OpenRouter model to use"
            )
            model_name = OPENROUTER_MODELS[selected_model_name]
            use_openrouter = True
        else:
            selected_model_name = st.selectbox(
                "Select LLM Model",
                options=list(AVAILABLE_MODELS.keys()),
                help="Choose which HuggingFace model to use"
            )
            model_name = AVAILABLE_MODELS[selected_model_name]
            use_openrouter = False
        
        # Retrieval method
        selected_retrieval_name = st.selectbox(
            "Retrieval Method",
            options=list(RETRIEVAL_METHODS.keys()),
            help="Choose how to retrieve information from the knowledge graph"
        )
        retrieval_method = RETRIEVAL_METHODS[selected_retrieval_name]
        
        # Embedding model selection (for embedding/hybrid modes)
        EMBEDDING_MODELS = {
            "MiniLM-L6 (Default)": "player_embedding_index_minilm",
            "Paraphrase-MiniLM": "player_embedding_index_para"
        }
        if retrieval_method in ["embeddings", "hybrid"]:
            selected_embedding_model = st.selectbox(
                "Embedding Model",
                options=list(EMBEDDING_MODELS.keys()),
                help="Choose which embedding model to use for semantic search"
            )
            embedding_index = EMBEDDING_MODELS[selected_embedding_model]
        else:
            embedding_index = "player_embedding_index_minilm"  # Default
        
        # Store in session state for use in process_question
        st.session_state["embedding_index"] = embedding_index
        
        st.divider()
        
        # Model comparison mode
        st.header("🏟️ Head-to-Head")
        compare_models = st.checkbox(
            "Compare All OpenRouter Models",
            help="Run the same question through all 3 OpenRouter models and compare results"
        )
        
        st.divider()
        
        # Example questions
        st.header("⚽ Quick Picks")
        for example in EXAMPLE_QUESTIONS:
            st.button(
                example, 
                key=f"example_{example}", 
                use_container_width=True,
                on_click=set_question,
                args=(example,)
            )
        
        st.divider()
        
        # Query history
        if st.session_state.history:
            st.header("🏅 Match History")
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
                st.session_state.openrouter_key = load_openrouter_key()
                
                if not st.session_state.openrouter_key:
                    st.warning("⚠️ OpenRouter key not found. Create openrouter_config.txt for OpenRouter models.")
                
                players, teams = load_known_players_and_teams()
                st.session_state.players = players
                st.session_state.teams = teams
                st.success(f"✅ Loaded {len(players)} players and {len(teams)} teams")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                st.info("Make sure Neo4j is running and config.txt is configured.")
                st.stop()
    
    # Batch Evaluation Section
    with st.expander("🏆 Tournament Mode (Multiple Prompts)", expanded=False):
        st.markdown("Run multiple test questions through all OpenRouter models for comprehensive comparison.")
        
        # Predefined test questions (using 2022-23 season - the latest in the data)
        test_questions = [
            "Who scored the most points in 2022-23?",
            "Compare Salah and Haaland in 2022-23",
            "Recommend a midfielder under 8 million",
            "Which team has the best defense in 2022-23?",
            "Top 5 defenders by points in 2022-23"
        ]
        
        st.caption(f"Test Questions: {len(test_questions)}")
        with st.expander("View Test Questions"):
            for i, q in enumerate(test_questions, 1):
                st.markdown(f"{i}. {q}")
        
        if st.button("🚀 Run Batch Evaluation", use_container_width=True):
            if not st.session_state.openrouter_key:
                st.error("OpenRouter API key required for batch evaluation")
            else:
                run_batch_evaluation(
                    test_questions, 
                    st.session_state.players, 
                    st.session_state.teams, 
                    st.session_state.openrouter_key
                )
    
    st.divider()
    
    # Main question input
    st.header("🎙️ Ask Your Manager")
    
    question = st.text_area(
        "Enter your FPL question:",
        height=100,
        placeholder="e.g., Who scored the most points in 2022-23?",
        key="question_input"
    )
    
    # Ask button
    if st.button("⚽ Kick Off!", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a question first")
        else:
            process_question(
                question=question,
                model_name=model_name,
                retrieval_method=retrieval_method,
                players=st.session_state.players,
                teams=st.session_state.teams,
                hf_token=st.session_state.hf_token,
                use_openrouter=use_openrouter,
                openrouter_key=st.session_state.openrouter_key,
                compare_models=compare_models
            )


def run_batch_evaluation(test_questions: List[str], players: List[str], 
                        teams: List[str], openrouter_key: str):
    """Run batch evaluation across all OpenRouter models and display results."""
    
    st.header("🧪 Batch Evaluation Results")
    
    all_results = []
    total_evals = len(test_questions) * len(OPENROUTER_MODELS)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_eval = 0
    
    for q_idx, question in enumerate(test_questions):
        # Process question once for retrieval
        status_text.text(f"Processing Q{q_idx+1}: {question[:40]}...")
        
        intent = classify_intent(question)
        entities = extract_entities(question, players, teams)
        query_embedding = get_query_embedding(question)
        
        # Get selected embedding index from session state
        embedding_index = st.session_state.get("embedding_index", "player_embedding_index_minilm")
        hybrid_results = retrieve_hybrid(
            intent=intent,
            entities=entities,
            query_embedding=query_embedding,
            index_name=embedding_index,
            top_k=10
        )
        
        for model_key in OPENROUTER_MODELS.values():
            current_eval += 1
            progress_bar.progress(current_eval / total_evals)
            status_text.text(f"Q{q_idx+1} / Model: {model_key}...")
            
            try:
                result = generate_openrouter_answer(
                    question=question,
                    hybrid_results=hybrid_results,
                    model_name=model_key,
                    api_key=openrouter_key
                )
                
                all_results.append({
                    'Question ID': q_idx + 1,
                    'Question': question,
                    'Model': model_key,
                    'Success': result.get('success', False),
                    'Response Time (s)': round(result.get('response_time', 0), 2),
                    'Prompt Tokens': result.get('prompt_tokens', 0),
                    'Completion Tokens': result.get('completion_tokens', 0),
                    'Total Tokens': result.get('token_count', 0),
                    'Answer Length (words)': len(result.get('answer', '').split()),
                    'Answer': result.get('answer', ''),
                    'Error': result.get('error', '')
                })
            
            except Exception as e:
                all_results.append({
                    'Question ID': q_idx + 1,
                    'Question': question,
                    'Model': model_key,
                    'Success': False,
                    'Response Time (s)': 0,
                    'Prompt Tokens': 0,
                    'Completion Tokens': 0,
                    'Total Tokens': 0,
                    'Answer Length (words)': 0,
                    'Answer': '',
                    'Error': str(e)
                })
    
    progress_bar.empty()
    status_text.empty()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Detailed Results", "📥 Export"])
    
    with tab1:
        st.subheader("Aggregate Metrics by Model")
        
        # Calculate aggregate metrics
        agg_data = []
        for model in OPENROUTER_MODELS.values():
            model_df = results_df[results_df['Model'] == model]
            success_df = model_df[model_df['Success'] == True]
            
            agg_data.append({
                'Model': model,
                'Success Rate': f"{(len(success_df) / len(model_df) * 100):.1f}%",
                'Avg Response Time (s)': f"{success_df['Response Time (s)'].mean():.2f}",
                'Avg Prompt Tokens': int(success_df['Prompt Tokens'].mean()) if len(success_df) > 0 else 0,
                'Avg Completion Tokens': int(success_df['Completion Tokens'].mean()) if len(success_df) > 0 else 0,
                'Avg Total Tokens': int(success_df['Total Tokens'].mean()) if len(success_df) > 0 else 0,
                'Avg Answer Length': int(success_df['Answer Length (words)'].mean()) if len(success_df) > 0 else 0
            })
        
        agg_df = pd.DataFrame(agg_data)
        st.dataframe(agg_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Response Time")
            models = [m.split('/')[-1].split(':')[0] for m in OPENROUTER_MODELS.values()]
            times = [results_df[results_df['Model'] == m]['Response Time (s)'].mean() 
                    for m in OPENROUTER_MODELS.values()]
            
            fig = go.Figure(data=[go.Bar(
                x=models, y=times,
                marker_color=['#00FF87', '#04F5FF', '#E90052'],
                text=[f"{t:.2f}s" for t in times],
                textposition='outside'
            )])
            fig.update_layout(
                yaxis_title="Seconds",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Token Usage")
            prompt_tokens = [results_df[results_df['Model'] == m]['Prompt Tokens'].mean() 
                           for m in OPENROUTER_MODELS.values()]
            completion_tokens = [results_df[results_df['Model'] == m]['Completion Tokens'].mean() 
                               for m in OPENROUTER_MODELS.values()]
            
            fig = go.Figure(data=[
                go.Bar(name='Prompt', x=models, y=prompt_tokens, marker_color='#00FF87'),
                go.Bar(name='Completion', x=models, y=completion_tokens, marker_color='#E90052')
            ])
            fig.update_layout(
                barmode='stack',
                yaxis_title="Tokens",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("All Evaluation Results")
        
        # Display results without the long answer column
        display_df = results_df[['Question ID', 'Model', 'Success', 'Response Time (s)', 
                                'Total Tokens', 'Answer Length (words)']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Show individual answers in expanders
        st.subheader("Individual Answers")
        for q_id in results_df['Question ID'].unique():
            q_results = results_df[results_df['Question ID'] == q_id]
            question = q_results.iloc[0]['Question']
            
            with st.expander(f"Q{q_id}: {question[:60]}..."):
                cols = st.columns(len(OPENROUTER_MODELS))
                for idx, (col, model) in enumerate(zip(cols, OPENROUTER_MODELS.values())):
                    with col:
                        model_result = q_results[q_results['Model'] == model].iloc[0]
                        st.markdown(f"**{model.split('/')[-1]}**")
                        if model_result['Success']:
                            st.success(model_result['Answer'][:500] + "..." if len(model_result['Answer']) > 500 else model_result['Answer'])
                        else:
                            st.error(f"Failed: {model_result['Error']}")
                        st.caption(f"⏱️ {model_result['Response Time (s)']}s | 🔢 {model_result['Total Tokens']} tokens")
    
    with tab3:
        st.subheader("Export Batch Evaluation Results")
        
        # CSV download
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv,
            file_name=f"batch_evaluation_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary CSV
        summary_csv = agg_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"batch_summary_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.info(f"✅ Batch evaluation complete: {len(test_questions)} questions × {len(OPENROUTER_MODELS)} models = {len(all_results)} evaluations")


def display_model_comparison(question: str, hybrid_results: Dict[str, Any], 
                            openrouter_key: str, retrieval_method: str, start_time: float):
    """Run all OpenRouter models and display comparison."""
    
    st.header("📊 Model Comparison")
    st.caption("Running the same query through all 3 OpenRouter models...")
    
    # Run all models
    comparison_results = []
    model_names = list(OPENROUTER_MODELS.values())
    model_display_names = list(OPENROUTER_MODELS.keys())
    
    progress_bar = st.progress(0)
    
    for idx, (display_name, model_key) in enumerate(OPENROUTER_MODELS.items()):
        progress_bar.progress((idx + 1) / len(OPENROUTER_MODELS), f"Testing {display_name}...")
        
        try:
            result = generate_openrouter_answer(
                question=question,
                hybrid_results=hybrid_results,
                model_name=model_key,
                api_key=openrouter_key
            )
            result['display_name'] = display_name
            result['model_key'] = model_key
            comparison_results.append(result)
        except Exception as e:
            comparison_results.append({
                'display_name': display_name,
                'model_key': model_key,
                'success': False,
                'error': str(e),
                'response_time': 0,
                'token_count': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'answer': f"Error: {e}"
            })
    
    progress_bar.empty()
    
    # Store results for export
    st.session_state.comparison_results = comparison_results
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Quantitative", "💬 Qualitative", "📥 Export"])
    
    with tab1:
        st.subheader("Quantitative Metrics")
        
        # Build metrics dataframe
        metrics_data = []
        for r in comparison_results:
            metrics_data.append({
                'Model': r['display_name'],
                'Response Time (s)': f"{r.get('response_time', 0):.2f}",
                'Prompt Tokens': r.get('prompt_tokens', 0),
                'Completion Tokens': r.get('completion_tokens', 0),
                'Total Tokens': r.get('token_count', 0),
                'Success': '✅' if r.get('success') else '❌'
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Response Time Chart
            st.subheader("Response Time Comparison")
            response_times = [r.get('response_time', 0) for r in comparison_results]
            models = [r['display_name'].split(' (')[0] for r in comparison_results]
            
            fig_time = go.Figure(data=[
                go.Bar(
                    x=models,
                    y=response_times,
                    marker_color=['#00FF87', '#04F5FF', '#E90052'],
                    text=[f"{t:.2f}s" for t in response_times],
                    textposition='outside'
                )
            ])
            fig_time.update_layout(
                yaxis_title="Seconds",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Token Usage Chart
            st.subheader("Token Usage Comparison")
            prompt_tokens = [r.get('prompt_tokens', 0) for r in comparison_results]
            completion_tokens = [r.get('completion_tokens', 0) for r in comparison_results]
            
            fig_tokens = go.Figure(data=[
                go.Bar(name='Prompt', x=models, y=prompt_tokens, marker_color='#00FF87'),
                go.Bar(name='Completion', x=models, y=completion_tokens, marker_color='#E90052')
            ])
            fig_tokens.update_layout(
                barmode='stack',
                yaxis_title="Tokens",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
    
    with tab2:
        st.subheader("Qualitative Comparison - Side by Side Answers")
        
        # Create columns for each model
        cols = st.columns(len(comparison_results))
        
        for idx, (col, result) in enumerate(zip(cols, comparison_results)):
            with col:
                st.markdown(f"**{result['display_name']}**")
                if result.get('success'):
                    st.success(result['answer'])
                else:
                    st.error(f"Failed: {result.get('error', 'Unknown')}")
                st.caption(f"⏱️ {result.get('response_time', 0):.2f}s | 🔢 {result.get('token_count', 0)} tokens")
    
    with tab3:
        st.subheader("Export Comparison Results")
        
        # Prepare export data
        export_data = []
        for r in comparison_results:
            export_data.append({
                'Question': question,
                'Model': r['display_name'],
                'Model Key': r['model_key'],
                'Success': r.get('success', False),
                'Response Time (s)': r.get('response_time', 0),
                'Prompt Tokens': r.get('prompt_tokens', 0),
                'Completion Tokens': r.get('completion_tokens', 0),
                'Total Tokens': r.get('token_count', 0),
                'Answer': r.get('answer', ''),
                'Error': r.get('error', '')
            })
        
        export_df = pd.DataFrame(export_data)
        
        # CSV download button
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"model_comparison_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Show preview
        with st.expander("Preview Export Data"):
            st.dataframe(export_df, use_container_width=True)
    
    # Add to history
    elapsed_time = time.time() - start_time
    st.session_state.history.append({
        'question': question,
        'answer': f"Comparison: {len([r for r in comparison_results if r.get('success')])} models succeeded",
        'model': 'COMPARISON',
        'method': retrieval_method,
        'time': elapsed_time,
        'provider': 'openrouter'
    })


def process_question(question: str, model_name: str, retrieval_method: str, 
                    players: List[str], teams: List[str], hf_token: str,
                    use_openrouter: bool = False, openrouter_key: str = None,
                    compare_models: bool = False):
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
        # Format all entities for display
        ent_dict = entities.to_dict()
        readable_entities = []
        
        if ent_dict.get('player_names'):
            readable_entities.append(f"Players: {', '.join(ent_dict['player_names'])}")
        if ent_dict.get('team_names'):
            readable_entities.append(f"Teams: {', '.join(ent_dict['team_names'])}")
        if ent_dict.get('position'):
            readable_entities.append(f"Position: {ent_dict['position']}")
        if ent_dict.get('season'):
            readable_entities.append(f"Season: {ent_dict['season']}")
        if ent_dict.get('stat_name'):
            readable_entities.append(f"Stat: {ent_dict['stat_name']}")
        if ent_dict.get('gameweek'):
            readable_entities.append(f"GW: {ent_dict['gameweek']}")
        if ent_dict.get('budget'):
            readable_entities.append(f"Budget: {ent_dict['budget']}")
            
        st.info(f"**Entities:** {' | '.join(readable_entities) or 'None'}")
    
    # Step 3: Retrieval
    with st.spinner(f"Retrieving from Knowledge Graph ({retrieval_method})..."):
        query_embedding = get_query_embedding(question)
        
        # Modify retrieval based on method
        if retrieval_method == "baseline":
            # Only baseline
            from graph_retrieval import run_baseline_retrieval
            baseline_result = run_baseline_retrieval(intent, entities)
            baseline_players = baseline_result.get("players", [])
            baseline_fixtures = baseline_result.get("fixtures", [])
            baseline_teams = baseline_result.get("teams", [])
            hybrid_results = {
                "baseline_players": baseline_players,
                "baseline_fixtures": baseline_fixtures,
                "baseline_teams": baseline_teams,
                "embedding_players": [],
                "summary": {
                    "baseline_player_count": len(baseline_players), 
                    "baseline_fixture_count": len(baseline_fixtures),
                    "baseline_team_count": len(baseline_teams),
                    "embedding_player_count": 0
                },
                "baseline_result": baseline_result,
                "baseline_context": baseline_result,
                "cypher_query": baseline_result.get("cypher_query")
            }
        elif retrieval_method == "embeddings":
            # Only embeddings
            from graph_retrieval import run_embedding_retrieval
            embedding_result = run_embedding_retrieval(query_embedding, entities)
            hybrid_results = {
                "baseline_players": [],
                "baseline_fixtures": [],
                "baseline_teams": [],
                "embedding_players": embedding_result.get("players", []),
                "summary": {
                    "baseline_player_count": 0, 
                    "baseline_fixture_count": 0,
                    "baseline_team_count": 0,
                    "embedding_player_count": len(embedding_result.get("players", []))
                },
                "cypher_query": "No Cypher query used (Embeddings Mode)"
            }
        else:
            # Hybrid
            # Get selected embedding index from session state
            embedding_index = st.session_state.get("embedding_index", "player_embedding_index_minilm")
            hybrid_results = retrieve_hybrid(
                intent=intent,
                entities=entities,
                query_embedding=query_embedding,
                index_name=embedding_index,
                top_k=10
            )
    
    # Check if comparison mode is enabled
    if compare_models and openrouter_key:
        # Run comparison across all OpenRouter models
        display_model_comparison(question, hybrid_results, openrouter_key, retrieval_method, start_time)
    else:
        # Single model answer generation
        st.header("📺 Match Analysis")
        with st.spinner(f"Generating answer with {model_name.upper()}..."):
            try:
                # Display retrieval statistics - include all types
                summary = hybrid_results.get("summary", {})
                player_count = summary.get('baseline_player_count', 0) + summary.get('embedding_player_count', 0)
                fixture_count = summary.get('baseline_fixture_count', 0)
                team_count = summary.get('baseline_team_count', 0)
                
                context_parts = []
                if player_count > 0:
                    context_parts.append(f"{player_count} players")
                if fixture_count > 0:
                    context_parts.append(f"{fixture_count} fixtures")
                if team_count > 0:
                    context_parts.append(f"{team_count} teams")
                
                context_str = ", ".join(context_parts) if context_parts else "0 items"
                st.caption(f"Context: {context_str}")
                
                if use_openrouter:
                    if not openrouter_key:
                        st.error("❌ OpenRouter API key not found. Please create openrouter_config.txt")
                        return
                    result = generate_openrouter_answer(
                        question=question,
                        hybrid_results=hybrid_results,
                        model_name=model_name,
                        api_key=openrouter_key
                    )
                else:
                    if not hf_token:
                        st.error("❌ HuggingFace token not found. Please create hf.txt")
                        return
                    result = generate_fpl_answer(
                        question=question,
                        hybrid_results=hybrid_results,
                        model_name=model_name,
                        hf_token=hf_token
                    )
                
                if result.get('success'):
                    # Big answer box
                    st.success(result['answer'])
                    
                    # Response metadata
                    elapsed_time = time.time() - start_time
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Response Time", f"{result.get('response_time', elapsed_time):.2f}s")
                    with col2:
                        st.metric("Model", model_name.upper())
                    with col3:
                        st.metric("Total Tokens", result.get('token_count', 'N/A'))
                    with col4:
                        provider = "OpenRouter" if use_openrouter else "HuggingFace"
                        st.metric("Provider", provider)
                    
                    # Show detailed token breakdown for OpenRouter
                    if use_openrouter and result.get('prompt_tokens'):
                        with st.expander("📊 Token Details"):
                            token_col1, token_col2, token_col3 = st.columns(3)
                            with token_col1:
                                st.metric("Prompt Tokens", result.get('prompt_tokens', 0))
                            with token_col2:
                                st.metric("Completion Tokens", result.get('completion_tokens', 0))
                            with token_col3:
                                st.metric("Total Tokens", result.get('token_count', 0))
                    
                    # Add to history
                    st.session_state.history.append({
                        'question': question,
                        'answer': result['answer'],
                        'model': model_name,
                        'method': retrieval_method,
                        'time': elapsed_time,
                        'provider': 'openrouter' if use_openrouter else 'huggingface'
                    })
                else:
                    st.error(f"❌ Error generating answer: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("View error details"):
                    st.code(traceback.format_exc())

    # Step 5: Knowledge Graph Context (MOVED DOWN)
    st.header("🎯 Tactical Breakdown")
    
    baseline_players = hybrid_results.get("baseline_players", [])
    embedding_players = hybrid_results.get("embedding_players", [])
    baseline_fixtures = hybrid_results.get("baseline_fixtures", [])
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Combined View", "Baseline Results", "Embedding Results"])
    
    # ... (Dataframe Logic - Keeping existing helper)
    
    # Helper for cleaner columns
    column_config = {
        "Name": st.column_config.TextColumn("Name", width="medium", required=True),
        "Team": st.column_config.TextColumn("Team", width="small"),
        "Position": st.column_config.TextColumn("Pos", width="small"),
        "Points": st.column_config.NumberColumn(
            "Pts",
            help="Total FPL Points",
            format="%d ⭐"
        ),
        "Goals": st.column_config.ProgressColumn(
            "Goals",
            help="Goals Scored",
            format="%d",
            min_value=0,
            max_value=30, # Approx max goals
        ),
        "Assists": st.column_config.ProgressColumn(
            "Assists",
            help="Assists",
            format="%d",
            min_value=0,
            max_value=20, 
        ),
        "Form": st.column_config.NumberColumn(
            "Form",
            help="Recent Form",
            format="%.1f 🔥"
        ),
        "Price": st.column_config.TextColumn(
            "Price",
            help="Average Price",
        ),
        "Clean Sheets": st.column_config.NumberColumn(
            "Clean Sheets",
            help="Clean Sheets",
        ),
        "Conceded": st.column_config.NumberColumn(
            "Conceded",
            help="Goals Conceded",
        ),
        "Similarity": st.column_config.ProgressColumn(
            "Match",
            help="Similarity Score",
            format="%.2f",
            min_value=0, 
            max_value=1
        )
    }

    with tab1:
        all_players = baseline_players + embedding_players
        if all_players:
            df = format_player_dataframe(all_players)
            st.dataframe(
                df, 
                use_container_width=True,
                column_config=column_config,
                hide_index=True
            )
        elif baseline_fixtures:
            # Display fixtures if no players but fixtures exist
            st.subheader("📅 Fixtures")
            fixture_data = []
            for f in baseline_fixtures:
                fixture_data.append({
                    "GW": f.get("gw", "N/A"),
                    "Home": f.get("home_team", "N/A"),
                    "Away": f.get("away_team", "N/A"),
                    "Kickoff": f.get("kickoff", "N/A")
                })
            st.dataframe(pd.DataFrame(fixture_data), use_container_width=True, hide_index=True)
        else:
            st.info("No players retrieved")
    
    with tab2:
        if baseline_players:
            df = format_player_dataframe(baseline_players)
            st.dataframe(
                df, 
                use_container_width=True,
                column_config=column_config,
                hide_index=True
            )
        else:
            st.info("No baseline results (may be using embeddings-only mode)")
    
    with tab3:
        if embedding_players:
            df = format_player_dataframe(embedding_players)
            st.dataframe(
                df, 
                use_container_width=True,
                column_config=column_config,
                hide_index=True
            )
        else:
            st.info("No embedding results (may be using baseline-only mode)")
    
    # Step 6: Show Cypher queries
    with st.expander("🔍 View Cypher Queries"):
        cypher_queries = extract_cypher_queries(hybrid_results)
        if cypher_queries:
            for i, query in enumerate(cypher_queries, 1):
                st.code(query, language="cypher")
        else:
            st.info("Cypher query details not available in current retrieval mode")
    
    # Step 7: Graph visualization
    st.header("⚽ Formation View")
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


if __name__ == "__main__":
    main()
