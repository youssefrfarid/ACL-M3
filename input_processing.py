from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import re

from neo4j import GraphDatabase


# =====================  CONFIG / NEO4J CONNECTION  =====================

def read_config(config_file: str = "config.txt") -> dict:
    config = {}
    with open(config_file, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                config[k.strip()] = v.strip()
    return config


def get_driver():
    cfg = read_config()
    return GraphDatabase.driver(
        cfg.get("URI", "neo4j://localhost:7687"),
        auth=(cfg.get("USERNAME", "neo4j"), cfg.get("PASSWORD", "neo4j")),
    )


# =====================  INTENT LABELS  =====================

INTENT_PLAYER_INFO = "player_info"
INTENT_COMPARE_PLAYERS = "compare_players"
INTENT_FIXTURE_INFO = "fixture_info"
INTENT_PLAYER_RECOMMEND = "player_recommendation"
INTENT_TEAM_RECOMMEND = "team_recommendation"
INTENT_GENERAL_QUESTION = "general_question"


# Semantic intents → NEED embeddings
SEMANTIC_INTENTS = {
    INTENT_COMPARE_PLAYERS,
    INTENT_PLAYER_RECOMMEND,
    INTENT_TEAM_RECOMMEND,
}


# =====================  ENTITY DATA CLASS  =====================

@dataclass
class QueryEntities:
    player_names: List[str] = field(default_factory=list)
    team_names: List[str] = field(default_factory=list)
    position: Optional[str] = None
    gameweek: Optional[int] = None
    horizon_gw: Optional[int] = None
    season: Optional[str] = None
    budget: Optional[float] = None
    stat_name: Optional[str] = None

    def to_dict(self):
        return {
            "player_names": self.player_names,
            "team_names": self.team_names,
            "position": self.position,
            "gameweek": self.gameweek,
            "horizon_gw": self.horizon_gw,
            "season": self.season,
            "budget": self.budget,
            "stat_name": self.stat_name,
        }


# =====================  INPUT EMBEDDING (QUERY EMBEDDING)  =====================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_CLOUD_EMBEDDINGS = True
_embedding_model = None


def load_hf_token(token_file="hf.txt"):
    try:
        with open(token_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def get_query_embedding_cloud(text: str, token: str):
    """
    Converts the USER QUESTION into a fixed 384-dimensional vector
    using HuggingFace cloud inference.
    """
    from huggingface_hub import InferenceClient
    import numpy as np

    client = InferenceClient(token=token)
    emb = client.feature_extraction(text, model=EMBEDDING_MODEL_NAME)
    emb = np.array(emb)

    # Average pooling if token-level output
    if emb.ndim == 2:
        emb = emb.mean(axis=0)

    # Normalize for cosine similarity
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def get_query_embedding(text: str):
    """
    Generates an embedding for the USER QUERY.
    Called ONLY for semantic intents.
    Safe: never crashes preprocessing.
    """
    try:
        if USE_CLOUD_EMBEDDINGS:
            token = load_hf_token()
            if token:
                return get_query_embedding_cloud(text, token)

        # Fallback: local model
        from sentence_transformers import SentenceTransformer
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        return _embedding_model.encode(text, normalize_embeddings=True).tolist()

    except Exception as e:
        print("⚠️ Embedding failed, continuing without embedding.")
        print("Reason:", e)
        return None


# =====================  INTENT CLASSIFIER  =====================

def classify_intent(question: str) -> str:
    q = question.lower()

    if " vs " in q or " versus " in q or "compare" in q:
        return INTENT_COMPARE_PLAYERS

    if "recommend" in q or "suggest" in q:
        if "team" in q or "squad" in q:
            return INTENT_TEAM_RECOMMEND
        return INTENT_PLAYER_RECOMMEND

    if "fixture" in q or ("who do" in q and "play" in q):
        return INTENT_FIXTURE_INFO

    if any(w in q for w in ["points", "goals", "assists", "xg", "xa", "form"]):
        return INTENT_PLAYER_INFO

    return INTENT_GENERAL_QUESTION


# =====================  ENTITY HELPERS  =====================

def extract_gameweek(q: str):
    m = re.search(r"(gw|gameweek)\s*(\d+)", q.lower())
    return int(m.group(2)) if m else None


def extract_horizon_gw(q: str):
    m = re.search(r"next\s+(\d+)\s+(gameweeks|gws|weeks)", q.lower())
    return int(m.group(1)) if m else None


def extract_budget(q: str):
    m = re.search(r"(under|below|less than|for)\s+£?(\d+(\.\d+)?)", q.lower())
    return float(m.group(2)) if m else None


def extract_position(q: str):
    q = q.lower()
    if "gk" in q or "goalkeeper" in q:
        return "GK"
    if "def" in q or "defender" in q:
        return "DEF"
    if "mid" in q or "midfielder" in q:
        return "MID"
    if "fwd" in q or "forward" in q or "striker" in q:
        return "FWD"
    return None


def extract_stat_name(q: str):
    q = q.lower()
    for stat in ["points", "goals", "assists", "xg", "xa", "clean sheets"]:
        if stat in q:
            return stat.replace(" ", "_")
    return None


def extract_season(q: str):
    m = re.search(r"(\d{4})/(\d{4})", q)
    if m:
        return f"{m.group(1)}-{m.group(2)[-2:]}"
    m = re.search(r"(\d{4})[-/](\d{2})", q)
    return f"{m.group(1)}-{m.group(2)}" if m else None


# =====================  ENTITY EXTRACTION (ROBUST)  =====================

def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())


def extract_players_and_teams(
    question: str,
    known_players: List[str],
    known_teams: List[str],
) -> Tuple[List[str], List[str]]:

    q = normalize_text(question)
    tokens = set(q.split())

    players = []
    for p in known_players:
        p_norm = normalize_text(p)
        last_name = p_norm.split()[-1]
        if p_norm in q or last_name in tokens:
            players.append(p)

    teams = []
    for t in known_teams:
        if normalize_text(t) in q:
            teams.append(t)

    return list(set(players)), list(set(teams))


def extract_entities(question: str, players: List[str], teams: List[str]) -> QueryEntities:
    p, t = extract_players_and_teams(question, players, teams)
    return QueryEntities(
        player_names=p,
        team_names=t,
        position=extract_position(question),
        gameweek=extract_gameweek(question),
        horizon_gw=extract_horizon_gw(question),
        season=extract_season(question),
        budget=extract_budget(question),
        stat_name=extract_stat_name(question),
    )


# =====================  LOAD GRAPH DATA  =====================

def load_known_players_and_teams():
    driver = get_driver()
    with driver.session() as s:
        players = [r["name"] for r in s.run("MATCH (p:Player) RETURN p.player_name AS name")]
        teams = [r["name"] for r in s.run("MATCH (t:Team) RETURN t.name AS name")]
    driver.close()
    return players, teams


# =====================  MANUAL TESTING  =====================

if __name__ == "__main__":
    players, teams = load_known_players_and_teams()
    print(f"Loaded {len(players)} players, {len(teams)} teams")

    while True:
        q = input("\nAsk FPL question (exit to quit): ")
        if q.lower() == "exit":
            break

        intent = classify_intent(q)
        entities = extract_entities(q, players, teams)

        print("Intent:", intent)
        print("Entities:", entities.to_dict())

        # ✅ Correct embedding logic
        embedding = None
        if intent in SEMANTIC_INTENTS:
            embedding = get_query_embedding(q)

        if embedding:
            print("Embedding generated (length =", len(embedding), ")")
        else:
            print("Embedding not used")
