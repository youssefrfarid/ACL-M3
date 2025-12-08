from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import re

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# =====================  CONFIG / NEO4J CONNECTION  =====================

def read_config(config_file: str = "config.txt") -> dict:
    """
    Load Neo4j credentials from config.txt.
    Expected format:
        URI=neo4j+s://...databases.neo4j.io  (for Aura)
        USERNAME=neo4j
        PASSWORD=your_password
    """
    config = {}
    try:
        with open(config_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "config.txt not found. Make sure it exists in the same folder as this script."
        )
    return config


def get_driver():
    cfg = read_config()
    uri = cfg.get("URI", "neo4j://localhost:7687")
    user = cfg.get("USERNAME", "neo4j")
    password = cfg.get("PASSWORD", "neo4j")
    # For Aura, uri should be like neo4j+s://....databases.neo4j.io
    return GraphDatabase.driver(uri, auth=(user, password))


# =====================  INTENT LABELS  =====================

INTENT_PLAYER_INFO = "player_info"               # single player stats/info
INTENT_COMPARE_PLAYERS = "compare_players"       # compare 2+ players
INTENT_FIXTURE_INFO = "fixture_info"             # fixtures, schedule
INTENT_PLAYER_RECOMMEND = "player_recommendation"
INTENT_TEAM_RECOMMEND = "team_recommendation"
INTENT_GENERAL_QUESTION = "general_question"     # fallback / generic


# =====================  ENTITY DATA CLASS  =====================

@dataclass
class QueryEntities:
    player_names: List[str] = field(default_factory=list)
    team_names: List[str] = field(default_factory=list)
    position: Optional[str] = None          # "GK", "DEF", "MID", "FWD"
    gameweek: Optional[int] = None
    horizon_gw: Optional[int] = None        # "next 3 gameweeks"
    season: Optional[str] = None
    budget: Optional[float] = None          # e.g. 8.0
    stat_name: Optional[str] = None         # "points", "goals", etc.

    def to_dict(self) -> Dict:
        return {
            "player_names": self.player_names or [],
            "team_names": self.team_names or [],
            "position": self.position,
            "gameweek": self.gameweek,
            "horizon_gw": self.horizon_gw,
            "season": self.season,
            "budget": self.budget,
            "stat_name": self.stat_name,
        }


# =====================  EMBEDDING MODEL  =====================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformers model once and reuse it.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_query_embedding(question: str) -> List[float]:
    """
    Compute an embedding vector for the user question.
    This will be used later for embedding-based retrieval in Neo4j.
    """
    model = get_embedding_model()
    emb = model.encode(question, normalize_embeddings=True)  # numpy array
    return emb.tolist()  # convert to Python list (easier to store/send)


# =====================  INTENT CLASSIFIER  =====================

def classify_intent(question: str) -> str:
    """
    Very simple rule-based intent classifier for FPL queries.
    You can improve this later (LLM, more rules, etc.).
    """
    q = question.lower()

    # 1) Compare players
    if " vs " in q or " versus " in q or "compare" in q:
        return INTENT_COMPARE_PLAYERS

    # 2) Recommendations
    if "recommend" in q or "suggest" in q or "who should i buy" in q:
        if "team" in q or "squad" in q or "wildcard" in q:
            return INTENT_TEAM_RECOMMEND
        return INTENT_PLAYER_RECOMMEND

    # 3) Fixtures
    if "fixture" in q or "fixtures" in q or ("who do" in q and "play" in q):
        return INTENT_FIXTURE_INFO

    # 4) Player stats/info
    stats_keywords = ["points", "stats", "goals", "assists", "xg", "xa", "form"]
    if any(word in q for word in stats_keywords):
        return INTENT_PLAYER_INFO

    # 5) Fallback
    return INTENT_GENERAL_QUESTION


# =====================  ENTITY HELPERS  =====================

def extract_gameweek(question: str) -> Optional[int]:
    q = question.lower()
    # matches: "gw5", "gw 5", "gameweek 5"
    m = re.search(r"(gw|gameweek)\s*(\d+)", q)
    if m:
        return int(m.group(2))
    return None


def extract_horizon_gw(question: str) -> Optional[int]:
    q = question.lower()
    # matches: "next 3 gameweeks", "next 2 gws", "next 4 weeks"
    m = re.search(r"next\s+(\d+)\s+(gameweeks|gws|weeks)", q)
    if m:
        return int(m.group(1))
    return None


def extract_budget(question: str) -> Optional[float]:
    q = question.lower()
    # matches: "under 8.0", "below 7.5m", "for 6.5"
    m = re.search(r"(under|below|less than|for)\s+(\d+(\.\d+)?)", q)
    if m:
        return float(m.group(2))
    return None


def extract_position(question: str) -> Optional[str]:
    q = question.lower()
    if any(w in q for w in ["goalkeeper", "keeper", "gk"]):
        return "GK"
    if any(w in q for w in ["defender", "def"]):
        return "DEF"
    if any(w in q for w in ["midfielder", "mid"]):
        return "MID"
    if any(w in q for w in ["forward", "striker", "fwd"]):
        return "FWD"
    return None


def extract_stat_name(question: str) -> Optional[str]:
    q = question.lower()
    if "points" in q:
        return "points"
    if "goals" in q:
        return "goals"
    if "assists" in q:
        return "assists"
    if "xg" in q:
        return "xg"
    if "xa" in q:
        return "xa"
    if "clean sheet" in q or "clean sheets" in q:
        return "clean_sheets"
    return None


def extract_season(question: str) -> Optional[str]:
    """
    Extract season from question.
    Matches patterns like:
    - "2022-23", "2023-24"
    - "2022/23", "2023/24"
    - "2022/2023"
    - "season 2022-23"
    """
    q = question.lower()
    # Pattern 1: YYYY-YY (e.g., "2022-23")
    m = re.search(r"(\d{4})[-/](\d{2})", q)
    if m:
        year1 = m.group(1)
        year2 = m.group(2)
        return f"{year1}-{year2}"
    
    # Pattern 2: YYYY/YYYY (e.g., "2022/2023")
    m = re.search(r"(\d{4})/(\d{4})", q)
    if m:
        year1 = m.group(1)
        year2 = m.group(2)
        # Convert to short format: 2022/2023 -> 2022-23
        return f"{year1}-{year2[-2:]}"
    
    return None


# =====================  LOAD DATA FROM YOUR GRAPH  =====================

def load_known_players_and_teams() -> Tuple[List[str], List[str]]:
    """
    Reads distinct player and team names from YOUR Neo4j FPL graph.

    Assumes:
      - Player nodes: (:Player {player_name, player_element})
      - Team nodes:   (:Team {name})
    Adjust the Cypher if your labels/props differ.
    """
    driver = get_driver()
    players: List[str] = []
    teams: List[str] = []

    with driver.session() as session:
        # Players
        result_p = session.run("MATCH (p:Player) RETURN DISTINCT p.player_name AS name")
        players = [r["name"] for r in result_p if r["name"]]

        # Teams
        result_t = session.run("MATCH (t:Team) RETURN DISTINCT t.name AS name")
        teams = [r["name"] for r in result_t if r["name"]]

    driver.close()
    return players, teams


def extract_players_and_teams(
    question: str,
    known_players: List[str],
    known_teams: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Simple substring matching:
    If the full player/team name appears in the question text.
    """
    q_low = question.lower()
    players = [p for p in known_players if p and p.lower() in q_low]
    teams = [t for t in known_teams if t and t.lower() in q_low]
    return list(set(players)), list(set(teams))


def extract_entities(
    question: str,
    known_players: List[str],
    known_teams: List[str],
) -> QueryEntities:
    """Main entity extraction: calls all helper functions."""
    gw = extract_gameweek(question)
    horizon = extract_horizon_gw(question)
    budget = extract_budget(question)
    position = extract_position(question)
    stat_name = extract_stat_name(question)
    season = extract_season(question)  # Now extracts from question!

    player_names, team_names = extract_players_and_teams(
        question, known_players, known_teams
    )

    return QueryEntities(
        player_names=player_names,
        team_names=team_names,
        position=position,
        gameweek=gw,
        horizon_gw=horizon,
        season=season,
        budget=budget,
        stat_name=stat_name,
    )


# =====================  MANUAL TESTING (RUN THIS FILE)  =====================

if __name__ == "__main__":
    print("Connecting to Neo4j and loading players/teams from your FPL graph...")
    try:
        players, teams = load_known_players_and_teams()
    except Exception as e:
        print("Error loading from Neo4j:", e)
        print("Make sure:")
        print("- Neo4j is running (Aura instance online)")
        print("- config.txt exists and has URI, USERNAME, PASSWORD")
        raise SystemExit(1)

    print(f"Loaded {len(players)} players and {len(teams)} teams from the graph.")

    while True:
        q = input("\nType an FPL question (or 'exit'): ")
        if q.lower().strip() == "exit":
            break

        intent = classify_intent(q)
        entities = extract_entities(q, players, teams)
        embedding = get_query_embedding(q)

        print("\nIntent:", intent)
        print("Entities:", entities.to_dict())
        print(f"Embedding length: {len(embedding)}")
        print(f"Embedding (first 5 values): {embedding[:5]}")
