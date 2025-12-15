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
        
        # Stats fields
        if "total_points" in p:
            row["Points"] = int(p["total_points"]) if p["total_points"] else 0
        
        if "goals" in p:
            row["Goals"] = int(p["goals"]) if p["goals"] else 0
        elif "goals_scored" in p:
            row["Goals"] = int(p["goals_scored"]) if p["goals_scored"] else 0
        
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
    
    # Custom CSS for FPL Dark Theme with Animations
    st.markdown("""
        <style>
        /* ==================== FPL OFFICIAL COLORS ==================== */
        /* Primary: #38003C (Dark Purple)
           Accent 1: #00FF87 (Neon Green)
           Accent 2: #E90052 (Magenta Pink)
           Accent 3: #04F5FF (Cyan Blue) */
        
        /* ==================== GLOBAL DARK THEME ==================== */
        .stApp {
            background: linear-gradient(135deg, #1a0020 0%, #2d1b3d 50%, #1a0020 100%);
            background-attachment: fixed;
        }
        
        /* Main content area */
        .main .block-container {
            background-color: rgba(26, 0, 32, 0.6);
            border-radius: 15px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* ==================== ANIMATED HEADER ==================== */
        .main-header {
            font-family: 'Arial Black', sans-serif;
            font-size: 3.5rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #00FF87, #04F5FF, #E90052, #00FF87);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientFlow 3s ease infinite;
            text-shadow: 0 0 30px rgba(0, 255, 135, 0.3);
        }
        
        @keyframes gradientFlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .sub-header {
            font-family: 'Arial', sans-serif;
            font-size: 1.4rem;
            color: #00FF87;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* ==================== METRIC CARDS ==================== */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(135deg, #00FF87, #04F5FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: scaleIn 0.4s ease-out;
        }
        
        @keyframes scaleIn {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: bold;
            color: #E90052;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem;
        }
        
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(56, 0, 60, 0.8), rgba(56, 0, 60, 0.4));
            border: 2px solid #00FF87;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0, 255, 135, 0.2);
            transition: all 0.3s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 255, 135, 0.4);
            border-color: #04F5FF;
        }
        
        /* ==================== BUTTONS ==================== */
        .stButton > button {
            background: linear-gradient(135deg, #38003C, #5a0060) !important;
            color: #00FF87 !important;
            border: 2px solid #00FF87 !important;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(0, 255, 135, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover:before {
            width: 300px;
            height: 300px;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #00FF87, #04F5FF) !important;
            color: #38003C !important;
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 25px rgba(0, 255, 135, 0.5);
            border-color: #38003C !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(0.98);
        }
        
        /* ==================== TABS ==================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(56, 0, 60, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #00FF87;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(0, 255, 135, 0.1);
            border-color: #00FF87;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00FF87, #04F5FF) !important;
            color: #38003C !important;
            box-shadow: 0 4px 15px rgba(0, 255, 135, 0.4);
        }
        
        /* ==================== EXPANDERS ==================== */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(56, 0, 60, 0.8), rgba(56, 0, 60, 0.4));
            border-radius: 10px;
            font-weight: bold;
            color: #00FF87 !important;
            border: 1px solid #00FF87;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, rgba(0, 255, 135, 0.2), rgba(4, 245, 255, 0.2));
            box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
            transform: translateX(5px);
        }
        
        /* ==================== SIDEBAR DARK THEME ==================== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #38003C 0%, #1a0020 50%, #38003C 100%);
            border-right: 2px solid #00FF87;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #00FF87 !important;
            text-shadow: 0 0 10px rgba(0, 255, 135, 0.5);
        }
        
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #00FF87 !important;
        }
        
        /* ==================== DATAFRAMES & TABLES ==================== */
        [data-testid="stDataFrame"] {
            border: 2px solid #00FF87;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 255, 135, 0.3);
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from { transform: translateY(30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        [data-testid="stDataFrame"] table {
            background-color: rgba(26, 0, 32, 0.8) !important;
        }
        
        [data-testid="stDataFrame"] thead tr {
            background: linear-gradient(135deg, #38003C, #5a0060) !important;
        }
        
        [data-testid="stDataFrame"] thead th {
            color: #00FF87 !important;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 2px solid #00FF87 !important;
        }
        
        [data-testid="stDataFrame"] tbody tr {
            transition: all 0.2s ease;
        }
        
        [data-testid="stDataFrame"] tbody tr:hover {
            background-color: rgba(0, 255, 135, 0.1) !important;
            transform: scale(1.01);
        }
        
        /* ==================== TEXT AREAS & INPUTS ==================== */
        textarea, input {
            background-color: rgba(26, 0, 32, 0.8) !important;
            color: #00FF87 !important;
            border: 2px solid #38003C !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }
        
        textarea:focus, input:focus {
            border-color: #00FF87 !important;
            box-shadow: 0 0 15px rgba(0, 255, 135, 0.4) !important;
        }
        
        /* ==================== SPINNER/LOADING ==================== */
        .stSpinner > div {
            border-top-color: #00FF87 !important;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* ==================== SUCCESS/ERROR/INFO MESSAGES ==================== */
        .stSuccess {
            background-color: rgba(0, 255, 135, 0.2) !important;
            border-left: 4px solid #00FF87 !important;
            color: #00FF87 !important;
            animation: slideInRight 0.4s ease-out;
        }
        
        .stError {
            background-color: rgba(233, 0, 82, 0.2) !important;
            border-left: 4px solid #E90052 !important;
            color: #E90052 !important;
            animation: shake 0.5s ease-out;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }
        
        .stInfo {
            background-color: rgba(4, 245, 255, 0.2) !important;
            border-left: 4px solid #04F5FF !important;
            color: #04F5FF !important;
            animation: slideInRight 0.4s ease-out;
        }
        
        @keyframes slideInRight {
            from { transform: translateX(-30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .stWarning {
            background-color: rgba(255, 165, 0, 0.2) !important;
            border-left: 4px solid #FFA500 !important;
            animation: slideInRight 0.4s ease-out;
        }
        
        /* ==================== DIVIDERS ==================== */
        hr {
            border-color: #00FF87 !important;
            opacity: 0.3;
        }
        
        /* ==================== CODE BLOCKS ==================== */
        code {
            background-color: rgba(26, 0, 32, 0.9) !important;
            color: #04F5FF !important;
            border: 1px solid #38003C !important;
            border-radius: 4px;
            padding: 2px 6px;
        }
        
        pre {
            background-color: rgba(26, 0, 32, 0.9) !important;
            border: 2px solid #38003C !important;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 255, 135, 0.2);
        }
        
        /* ==================== SCROLLBAR ==================== */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(26, 0, 32, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #38003C, #00FF87);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #00FF87, #04F5FF);
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
    
    question = st.text_area(
        "Enter your FPL question:",
        height=100,
        placeholder="e.g., Who scored the most points in 2022-23?",
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
            hybrid_results = {
                "baseline_players": baseline_result.get("players", []),
                "embedding_players": [],
                "summary": {"baseline_player_count": len(baseline_result.get("players", [])), "embedding_player_count": 0},
                "baseline_result": baseline_result, # Store full result for metadata
                "cypher_query": baseline_result.get("cypher_query") # Direct access
            }
        elif retrieval_method == "embeddings":
            # Only embeddings
            from graph_retrieval import run_embedding_retrieval
            embedding_result = run_embedding_retrieval(query_embedding, entities)
            hybrid_results = {
                "baseline_players": [],
                "embedding_players": embedding_result.get("players", []),
                "summary": {"baseline_player_count": 0, "embedding_player_count": len(embedding_result.get("players", []))},
                "cypher_query": "No Cypher query used (Embeddings Mode)"
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
    
    # Step 4: Generate LLM Answer (MOVED TO TOP)
    st.header("💬 LLM Answer")
    with st.spinner(f"Generating answer with {model_name.upper()}..."):
        try:
            # Display retrieval statistics small
            summary = hybrid_results.get("summary", {})
            st.caption(f"Context: {summary.get('baseline_player_count', 0)} baseline items + {summary.get('embedding_player_count', 0)} embedding items")
            
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

    # Step 5: Knowledge Graph Context (MOVED DOWN)
    st.header("📊 Knowledge Graph Context")
    
    baseline_players = hybrid_results.get("baseline_players", [])
    embedding_players = hybrid_results.get("embedding_players", [])
    
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


if __name__ == "__main__":
    main()
