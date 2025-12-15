

from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from input_processing import (
    QueryEntities,
    read_config,
    INTENT_PLAYER_INFO,
    INTENT_COMPARE_PLAYERS,
    INTENT_FIXTURE_INFO,
    INTENT_PLAYER_RECOMMEND,
    INTENT_TEAM_RECOMMEND,
    INTENT_GENERAL_QUESTION,
)


_config = read_config()
_driver = GraphDatabase.driver(
    _config["URI"],
    auth=(_config["USERNAME"], _config["PASSWORD"])
)


def get_driver():
    return _driver


def q_player_season_summary(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T1 – Player season summary
    Returns aggregated stats for a single player across a season.
    """
    player = entities.player_names[0] if entities.player_names else None
    season = entities.season or "2023-24"

    result = session.run(
        """
        MATCH (p:Player {player_name: $player_name})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p,
             SUM(r.minutes) AS minutes,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             SUM(r.total_points) AS total_points,
             SUM(r.clean_sheets) AS clean_sheets,
             SUM(r.bonus) AS bonus
        RETURN p.player_name AS name,
               $season AS season,
               minutes, goals, assists, total_points, clean_sheets, bonus
        ORDER BY total_points DESC
        """,
        player_name=player,
        season=season,
    )

    rows = [rec.data() for rec in result]
    return {"template": "player_season_summary", "players": rows}


def q_player_recent_form(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T2 – Player recent form (last N fixtures)
    Returns the last N fixtures for a player with aggregated form stats.
    """
    player = entities.player_names[0] if entities.player_names else None
    season = entities.season or "2023-24"
    n = entities.horizon_gw or 5

    result = session.run(
        """
        MATCH (p:Player {player_name: $player_name})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, r, f
        ORDER BY f.fixture_number DESC
        LIMIT $n
        WITH p, COLLECT({
          fixture: f.fixture_number,
          minutes: r.minutes,
          goals: r.goals_scored,
          assists: r.assists,
          points: r.total_points
        }) AS last_fixtures,
        AVG(r.total_points) AS avg_points
        RETURN p.player_name AS name,
               last_fixtures,
               avg_points
        """,
        player_name=player,
        season=season,
        n=n,
    )

    rows = [rec.data() for rec in result]
    return {"template": "player_recent_form", "players": rows}


def q_top_players_by_position(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T3 – Top players by position in a season
    Returns top players in a specific position ranked by total points.
    """
    position = entities.position or "MID"
    season = entities.season or "2023-24"
    limit = 20

    result = session.run(
        """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             AVG(r.form) AS form
        RETURN p.player_name AS name,
               $season AS season,
               pos.name AS position,
               total_points, goals, assists, form
        ORDER BY total_points DESC
        LIMIT $limit
        """,
        position=position,
        season=season,
        limit=limit,
    )
    rows = [rec.data() for rec in result]
    return {"template": "top_players_by_position", "players": rows}


def q_top_players_recent_form_position(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T4 – Top players by position in a GW window (recent form)
    Returns top players in a position based on recent gameweek performance.
    """
    position = entities.position or "MID"
    season = entities.season or "2023-24"
    gw_start = entities.gameweek or 1
    gw_end = entities.horizon_gw or gw_start + 3
    limit = 20

    result = session.run(
        """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (g:Gameweek {season: $season})-[:HAS_FIXTURE]->(f)
        WHERE g.GW_number >= $gw_start AND g.GW_number <= $gw_end
        WITH p, pos,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             AVG(r.form) AS avg_form
        RETURN p.player_name AS name,
               pos.name AS position,
               total_points,
               goals,
               avg_form
        ORDER BY avg_form DESC, total_points DESC
        LIMIT $limit
        """,
        position=position,
        season=season,
        gw_start=gw_start,
        gw_end=gw_end,
        limit=limit,
    )
    rows = [rec.data() for rec in result]
    return {"template": "top_recent_form_position", "players": rows}


def q_team_fixtures_range(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T5 – Fixtures for a team in a GW range
    Returns fixtures for a specific team within a gameweek range.
    """
    team = entities.team_names[0] if entities.team_names else None
    season = entities.season or "2023-24"
    gw_start = entities.gameweek or 1
    gw_end = entities.horizon_gw or gw_start + 3

    result = session.run(
        """
        MATCH (t:Team {name: $team_name})
        MATCH (g:Gameweek {season: $season})
        WHERE g.GW_number >= $gw_start AND g.GW_number <= $gw_end
        MATCH (g)-[:HAS_FIXTURE]->(f:Fixture)-[:HAS_HOME_TEAM|:HAS_AWAY_TEAM]->(t)
        WITH g, f
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        RETURN g.GW_number AS gw,
               f.fixture_number AS fixture,
               home.name AS home_team,
               away.name AS away_team,
               f.kickoff_time AS kickoff
        ORDER BY g.GW_number, fixture
        """,
        team_name=team,
        season=season,
        gw_start=gw_start,
        gw_end=gw_end,
    )

    rows = [rec.data() for rec in result]
    return {"template": "team_fixtures_range", "fixtures": rows}


def q_team_defensive_strength(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T6 – Team defensive strength (goals conceded)
    Returns teams ranked by defensive strength (lowest goals conceded).
    If no season specified, aggregates across all seasons.
    """
    season = entities.season
    limit = 20

    if season:
        # Filter by specific season
        result = session.run(
            """
            MATCH (t:Team)
            MATCH (f:Fixture {season: $season})
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            WITH t, SUM(r.goals_conceded) AS goals_conceded
            RETURN t.name AS team,
                   goals_conceded
            ORDER BY goals_conceded ASC
            LIMIT $limit
            """,
            season=season,
            limit=limit,
        )
    else:
        # Aggregate across all seasons
        result = session.run(
            """
            MATCH (t:Team)
            MATCH (f:Fixture)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
            WITH t, SUM(r.goals_conceded) AS goals_conceded
            RETURN t.name AS team,
                   goals_conceded
            ORDER BY goals_conceded ASC
            LIMIT $limit
            """,
            limit=limit,
        )

    rows = [rec.data() for rec in result]
    return {"template": "team_defensive_strength", "teams": rows}


def q_compare_two_players(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T7 – Compare two players (season totals)
    Returns season stats for multiple players for comparison.
    """
    season = entities.season or "2023-24"
    players = entities.player_names or []

    result = session.run(
        """
        MATCH (p:Player)
        WHERE p.player_name IN $player_names
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             SUM(r.minutes) AS minutes,
             AVG(r.form) AS form
        RETURN p.player_name AS name,
               total_points, goals, assists, minutes, form
        ORDER BY total_points DESC
        """,
        player_names=players,
        season=season,
    )
    rows = [rec.data() for rec in result]
    return {"template": "compare_two_players", "players": rows}


def q_top_scorers_overall(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T8 – Overall top points leaderboard
    Returns top players ranked by total points in a season.
    """
    season = entities.season or "2023-24"
    limit = 30

    result = session.run(
        """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists
        RETURN p.player_name AS name,
               total_points, goals, assists
        ORDER BY total_points DESC
        LIMIT $limit
        """,
        season=season,
        limit=limit,
    )
    rows = [rec.data() for rec in result]
    return {"template": "top_scorers_overall", "players": rows}


def q_cheap_defenders_from_strong_defences(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T9 – Cheap defenders from strong defences
    Returns defenders from teams with low goals conceded.
    (Budget filtering commented out if price property doesn't exist)
    """
    season = entities.season or "2023-24"
    limit = 20
    budget = entities.budget  # may be None

    cypher = """
    MATCH (team:Team)
    MATCH (f:Fixture {season: $season})
    MATCH (p:Player)-[r:PLAYED_IN]->(f)
    WHERE ((f)-[:HAS_HOME_TEAM]->(team) OR (f)-[:HAS_AWAY_TEAM]->(team))
      AND EXISTS { MATCH (p)-[:PLAYS_AS]->(:Position {name: 'DEF'}) }
    WITH team, p, SUM(r.goals_conceded) AS goals_conceded,
         SUM(r.total_points) AS total_points
    RETURN p.player_name AS name,
           team.name AS team,
           goals_conceded,
           total_points
    ORDER BY goals_conceded ASC, total_points DESC
    LIMIT $limit
    """

    result = session.run(cypher, season=season, limit=limit)
    rows = [rec.data() for rec in result]
    return {"template": "cheap_defenders_from_strong_defences", "players": rows}


def q_leaderboard_by_stat(session, entities: QueryEntities) -> Dict[str, Any]:
    """
    T10 – Generic fallback leaderboard by stat (e.g. goals or assists)
    Returns top players ranked by a specific stat.
    """
    season = entities.season or "2023-24"
    stat = entities.stat_name or "goals_scored"
    limit = 20

    # Map logical stat_name to relationship property
    stat_prop_map = {
        "goals": "goals_scored",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "points": "total_points",
        "total_points": "total_points",
    }
    prop = stat_prop_map.get(stat, "total_points")

    cypher = f"""
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {{season: $season}})
    RETURN p.player_name AS name,
           SUM(r.{prop}) AS value
    ORDER BY value DESC
    LIMIT $limit
    """

    result = session.run(cypher, season=season, limit=limit)
    rows = [rec.data() for rec in result]
    return {"template": f"leaderboard_by_{prop}", "players": rows}


# ROUTER FUNCTION

def run_baseline_retrieval(intent: str, entities: QueryEntities) -> Dict[str, Any]:
    """
    Router function that selects the appropriate Cypher query template
    based on the intent and QueryEntities.
    """
    with get_driver().session() as session:
        # Case 1: Player Info / Stats
        if intent == INTENT_PLAYER_INFO:
            if entities.stat_name:
                # Specific stat requested -> leaderboard
                result = q_leaderboard_by_stat(session, entities)
            elif entities.gameweek is not None or entities.horizon_gw is not None:
                # Specific gameweeks -> recent form
                result = q_player_recent_form(session, entities)
            elif entities.player_names:
                # Specific player -> season summary
                result = q_player_season_summary(session, entities)
            else:
                # Fallback -> top scorers
                result = q_top_scorers_overall(session, entities)

        # Case 2: Compare Players
        elif intent == INTENT_COMPARE_PLAYERS:
            result = q_compare_two_players(session, entities)

        # Case 3: Fixtures
        elif intent == INTENT_FIXTURE_INFO:
            result = q_team_fixtures_range(session, entities)

        # Case 4: Player Recommendation
        elif intent == INTENT_PLAYER_RECOMMEND:
            result = q_top_players_recent_form_position(session, entities)

        # Case 5: Team Recommendation
        elif intent == INTENT_TEAM_RECOMMEND:
            # If position specified, treat as player recommendation for that team context
            if entities.position:
                result = q_top_players_recent_form_position(session, entities)
            else:
                result = q_team_defensive_strength(session, entities)

        # Case 6: General Question / Fallback
        else:
            if entities.position:
                # Position specified -> position-based leaderboard
                result = q_top_players_by_position(session, entities)
            elif entities.stat_name:
                # Specific stat -> leaderboard
                result = q_leaderboard_by_stat(session, entities)
            else:
                # Fallback -> top scorers overall
                result = q_top_scorers_overall(session, entities)

    # Wrap with metadata for the LLM layer
    result["mode"] = "baseline"
    result["intent"] = intent
    result["entities"] = entities.to_dict()
    return result


# EMBEDDING-BASED RETRIEVAL


def build_player_feature_descriptions(session) -> List[Dict[str, Any]]:
    """
    Build textual feature descriptions for all players based on stats.
    This implements Feature Vector Embeddings approach.
    Enriched with more nuanced signals like attacking contribution and cards.
    """
    result = session.run(
        """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, pos,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             SUM(r.minutes) AS minutes,
             SUM(r.clean_sheets) AS clean_sheets,
             SUM(r.penalties_scored) AS penalties_scored,
             SUM(r.penalties_missed) AS penalties_missed,
             SUM(r.yellow_cards) AS yellow_cards,
             SUM(r.red_cards) AS red_cards,
             AVG(r.form) AS form,
             AVG(r.creativity) AS creativity,
             AVG(r.threat) AS threat,
             AVG(r.ict_index) AS ict_index
        RETURN p.player_name AS name,
               pos.name AS position,
               total_points,
               goals,
               assists,
               minutes,
               clean_sheets,
               penalties_scored,
               penalties_missed,
               yellow_cards,
               red_cards,
               form,
               creativity,
               threat,
               ict_index
        """
    )

    players = []
    for rec in result:
        # Calculate derived metrics
        minutes = rec['minutes'] if rec['minutes'] else 1
        goals = rec['goals'] if rec['goals'] else 0
        assists = rec['assists'] if rec['assists'] else 0
        
        # Attacking contribution (goals + assists per 90)
        per_90_factor = 90.0 / max(1.0, float(minutes))
        attacking_contrib_val = (goals + assists) * per_90_factor
        
        if attacking_contrib_val > 0.6:
            attacking_text = "High attacking contribution"
        elif attacking_contrib_val > 0.3:
            attacking_text = "Moderate attacking contribution"
        else:
            attacking_text = "Low attacking contribution"

        # Optional fields
        penalties_text = ""
        if rec['penalties_scored'] and rec['penalties_scored'] > 0:
            penalties_text = f", Penalties scored: {rec['penalties_scored']}"
            
        cards_text = ""
        if (rec['yellow_cards'] and rec['yellow_cards'] > 0) or (rec['red_cards'] and rec['red_cards'] > 0):
            yc = rec['yellow_cards'] if rec['yellow_cards'] else 0
            rc = rec['red_cards'] if rec['red_cards'] else 0
            cards_text = f", Cards: {yc}Y/{rc}R"

        form_val = rec['form'] if rec['form'] is not None else 0.0
        ict_val = rec['ict_index'] if rec['ict_index'] is not None else 0.0
        
        # Build rich textual description
        desc = (
            f"Player: {rec['name']}, "
            f"Position: {rec['position']}, "
            f"Total points: {rec['total_points']}, "
            f"Goals: {goals}, "
            f"Assists: {assists}, "
            f"Minutes: {minutes}, "
            f"Clean sheets: {rec['clean_sheets']}, "
            f"Form: {form_val:.2f}, "
            f"ICT: {ict_val:.2f}, "
            f"Attacking: {attacking_text}"
            f"{penalties_text}"
            f"{cards_text}"
        )
        players.append({
            "name": rec["name"],
            "position": rec["position"],
            "description": desc,
        })
    
    return players


def build_and_store_player_embeddings(
    model_name: str,
    index_name: str = "player_embedding_index",
    use_cloud: bool = True,
    hf_token: Optional[str] = None,
) -> None:
    """
    Build feature-based embeddings for all players and store them in Neo4j.
    Also creates a vector index on :Player(embedding) for similarity search.
    
    This implements the 'Feature Vector Embeddings' option for FPL.
    
    Args:
        model_name: SentenceTransformer model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        index_name: Name for the vector index
        use_cloud: If True, use HuggingFace API (M1 compatible). If False, use local model.
        hf_token: HuggingFace API token (required if use_cloud=True)
    
    Example:
        # Cloud mode (M1 compatible)
        build_and_store_player_embeddings(
            "sentence-transformers/all-MiniLM-L6-v2",
            "player_embedding_index_minilm",
            use_cloud=True,
            hf_token="your_token_here"
        )
        
        # Local mode (may cause M1 issues)
        build_and_store_player_embeddings(
            "sentence-transformers/all-MiniLM-L6-v2",
            "player_embedding_index_minilm",
            use_cloud=False
        )
    """
    print(f"\nBuilding embeddings with model: {model_name}")
    print(f"Index name: {index_name}")
    print(f"Mode: {'Cloud API (M1 compatible)' if use_cloud else 'Local model'}")
    
    if use_cloud:
        # Cloud mode - use HuggingFace API
        from huggingface_hub import InferenceClient
        import numpy as np
        
        if not hf_token:
            # Try loading from file
            from input_processing import load_hf_token
            hf_token = load_hf_token()
            if not hf_token:
                raise ValueError(
                    "Cloud mode requires HuggingFace token. "
                    "Either pass hf_token parameter or create hf.txt file."
                )
        
        client = InferenceClient(token=hf_token)
        dim = 384  # Standard dimension for all-MiniLM-L6-v2
        print(f"Embedding dimension: {dim}")
        
        def encode_text(text: str) -> List[float]:
            """Encode text using HuggingFace API"""
            response = client.feature_extraction(text, model=model_name)
            embedding = np.array(response)
            if len(embedding.shape) == 2:
                embedding = np.mean(embedding, axis=0)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
    else:
        # Local mode - use SentenceTransformer
        from sentence_transformers import SentenceTransformer
        
        print("WARNING: Local mode may cause OpenMP lock issues on M1 Macs")
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {dim}")
        
        def encode_text(text: str) -> List[float]:
            """Encode text using local model"""
            emb = model.encode(text, normalize_embeddings=True)
            return emb.tolist()

    with get_driver().session() as session:
        # Step 1: Build feature descriptions
        print("\nStep 1: Building player feature descriptions...")
        players = build_player_feature_descriptions(session)
        print(f"Built descriptions for {len(players)} players")

        # Step 2: Store embeddings on Player nodes
        print("\nStep 2: Computing and storing embeddings...")
        for i, p in enumerate(players):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(players)} players...")
            
            emb_list = encode_text(p["description"])
            
            session.run(
                """
                MATCH (pl:Player {player_name: $name})
                SET pl.embedding = $emb,
                    pl.embedding_model = $model_name
                """,
                name=p["name"],
                emb=emb_list,
                model_name=model_name,
            )
        
        print(f"  Stored embeddings for all {len(players)} players")

        # Step 3: Create vector index
        print("\nStep 3: Creating vector index...")
        try:
            # Drop existing index if it exists with different config
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            
            # Create vector index with correct dimensions
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name}
                FOR (p:Player) ON (p.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """
            )
            print(f"✓ Created vector index '{index_name}' with {dim} dimensions")
        except Exception as e:
            print(f"Note: {e}")
            print("Index may already exist or require manual creation")
    
    print(f"\n✓ Embedding build complete!")
    print(f"  Model: {model_name}")
    print(f"  Mode: {'Cloud API' if use_cloud else 'Local'}")
    print(f"  Index: {index_name}")
    print(f"  Players: {len(players)}")


def run_embedding_retrieval(
    query_embedding: List[float],
    entities: QueryEntities,
    index_name: str = "player_embedding_index",
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Use Neo4j vector index to find players semantically similar to the query.
    Performs post-filtering and deduplication.
    
    Args:
        query_embedding: Normalized embedding vector for the query
        entities: QueryEntities for optional filtering
        index_name: Name of the vector index to use
        top_k: Number of top results to return
    
    Returns:
        Dict with mode='embedding', players list, and metadata
    """
    with get_driver().session() as session:
        # Fetch more candidates to allow for post-filtering
        fetch_k = top_k * 3
        
        # Get player stats along with similarity scores
        result = session.run(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:PLAYS_AS]->(pos:Position)
            
            // Get season stats for the player
            OPTIONAL MATCH (node)-[r:PLAYED_IN]->(f:Fixture {season: $season})
            WITH node, score, pos,
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.minutes) AS minutes,
                 AVG(r.form) AS form
            
            RETURN node.player_name AS name,
                   score,
                   pos.name AS position,
                   node.embedding_model AS model,
                   total_points,
                   goals,
                   assists,
                   minutes,
                   form
            """,
            index_name=index_name,
            top_k=fetch_k,
            embedding=query_embedding,
            season=entities.season or "2022-23",  # Default to latest season in data
        )

        raw_players = []
        for rec in result:
            raw_players.append({
                "name": rec["name"],
                "score": float(rec["score"]),
                "position": rec["position"],
                "model": rec["model"],
                "total_points": rec["total_points"] or 0,
                "goals": rec["goals"] or 0,
                "assists": rec["assists"] or 0,
                "minutes": rec["minutes"] or 0,
                "form": float(rec["form"]) if rec["form"] else 0.0,
                "team": None,  # Could add if needed
            })

    # Post-processing: Deduplicate and Filter
    seen_names = set()
    cleaned_players = []
    
    target_pos = entities.position
    # If explicit team names provided, filter by them
    target_teams = set([t.lower() for t in entities.team_names]) if entities.team_names else None

    for p in raw_players:
        # Deduplicate
        if p["name"] in seen_names:
            continue
        
        # Filter by Position
        if target_pos and p["position"] != target_pos:
            continue
            
        # Filter by Team (simple case-insensitive check)
        if target_teams:
            if not p["team"] or p["team"].lower() not in target_teams:
                continue
            
        seen_names.add(p["name"])
        cleaned_players.append(p)
        
        if len(cleaned_players) >= top_k:
            break

    return {
        "mode": "embedding",
        "index_name": index_name,
        "intent": None,
        "entities": entities.to_dict(),
        "players": cleaned_players,
    }


def retrieve_hybrid(
    intent: str,
    entities: QueryEntities,
    query_embedding: List[float],
    index_name: str = "player_embedding_index",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Orchestrate both baseline and embedding retrieval.
    Returns a unified result with summary metadata.
    """
    # 1. Baseline
    baseline_ctx = run_baseline_retrieval(intent, entities)
    
    # 2. Embedding
    embedding_ctx = run_embedding_retrieval(query_embedding, entities, index_name, top_k)
    
    # 3. Combine
    baseline_players = baseline_ctx.get("players", [])
    baseline_fixtures = baseline_ctx.get("fixtures", [])
    baseline_teams = baseline_ctx.get("teams", [])
    embedding_players = embedding_ctx.get("players", [])
    
    return {
        "mode": "hybrid",
        "intent": intent,
        "entities": entities.to_dict(),
        
        "baseline_players": baseline_players,
        "baseline_fixtures": baseline_fixtures,
        "baseline_teams": baseline_teams,
        
        "embedding_players": embedding_players,
        
        "summary": {
            "baseline_player_count": len(baseline_players),
            "baseline_fixture_count": len(baseline_fixtures),
            "baseline_team_count": len(baseline_teams),
            "embedding_player_count": len(embedding_players),
            "templates_used": {
                "baseline": baseline_ctx.get("template"),
                "embedding_index": index_name,
            },
        },
        
        # Retain raw contexts if needed for deeper inspection
        "baseline_context": baseline_ctx,
        "embedding_context": embedding_ctx,
    }




# MANUAL TESTING

if __name__ == "__main__":
    from input_processing import (
        classify_intent, 
        extract_entities, 
        load_known_players_and_teams,
        get_query_embedding
    )

    # Load players and teams from the graph
    print("Loading players and teams from Neo4j...")
    try:
        players, teams = load_known_players_and_teams()
        print(f"Loaded {len(players)} players and {len(teams)} teams.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using empty lists for testing purposes...")
        players = []
        teams = []

    print("\n" + "="*60)
    print("FPL Graph Retrieval - Interactive Testing")
    print("="*60)
    print("\nModes available:")
    print("  1. Baseline only (default)")
    print("  2. Embedding only")
    print("  3. Hybrid (both)")
    print("\nCommands:")
    print("  'mode [1|2|3]' - Change retrieval mode")
    print("  'exit' - Exit program")
    print("="*60)

    # Default settings
    retrieval_mode = int(input("Enter retrieval mode (1, 2, or 3): "))  # 1=baseline, 2=embedding, 3=hybrid
    embedding_index = "player_embedding_index_minilm"  # Matches build_embeddings.py output
    
    while True:
        q = input("\nAsk an FPL question: ")
        if q.strip().lower() == "exit":
            break

        # Process query
        intent = classify_intent(q)
        entities = extract_entities(q, players, teams)
        
        print("\n" + "="*60)
        print(f"Mode: {['', 'Baseline', 'Embedding', 'Hybrid'][retrieval_mode]}")
        print(f"Intent: {intent}")
        print(f"Entities: {entities.to_dict()}")
        
        # Mode 1: Baseline only
        if retrieval_mode == 1:
            ctx = run_baseline_retrieval(intent, entities)
            print(f"Template: {ctx.get('template')}")
            
            players_result = ctx.get("players", [])
            fixtures_result = ctx.get("fixtures", [])
            teams_result = ctx.get("teams", [])
            
            if players_result:
                print(f"\nBaseline Players (showing first 3 of {len(players_result)}):")
                for p in players_result[:3]:
                    print(f"  {p}")
            
            if fixtures_result:
                print(f"\nFixtures (showing first 3 of {len(fixtures_result)}):")
                for f in fixtures_result[:3]:
                    print(f"  {f}")
            
            if teams_result:
                print(f"\nTeams (showing first 3 of {len(teams_result)}):")
                for t in teams_result[:3]:
                    print(f"  {t}")
        
        # Mode 2: Embedding only
        elif retrieval_mode == 2:
            query_emb = get_query_embedding(q)
            ctx = run_embedding_retrieval(
                query_emb, 
                entities, 
                index_name=embedding_index, 
                top_k=10
            )
            print(f"Index: {ctx.get('index_name')}")
            
            emb_players = ctx.get("players", [])
            if emb_players:
                print(f"\nEmbedding Players (deduplicated, top {len(emb_players)}):")
                for p in emb_players:
                    print(f"  {p['name']:30s} | Pos: {p['position']:3s} | Score: {p['score']:.4f}")
            else:
                print("\nNo embedding results found")
                print("Hint: Run build_and_store_player_embeddings() first")
        
        # Mode 3: Hybrid
        elif retrieval_mode == 3:
            query_emb = get_query_embedding(q)
            ctx = retrieve_hybrid(
                intent, 
                entities, 
                query_emb, 
                index_name=embedding_index, 
                top_k=10
            )
            
            summary = ctx.get("summary", {})
            print(f"\nSummary:")
            print(f"  Baseline: {summary.get('baseline_player_count', 0)} players, "
                  f"{summary.get('baseline_team_count', 0)} teams")
            print(f"  Embedding: {summary.get('embedding_player_count', 0)} players")
            print(f"  Templates: {summary.get('templates_used', {})}")
            
            # Show baseline results (players, fixtures, or teams)
            base_players = ctx.get("baseline_players", [])
            base_fixtures = ctx.get("baseline_fixtures", [])
            base_teams = ctx.get("baseline_teams", [])
            
            if base_players:
                print(f"\nBaseline Players (first 3):")
                for p in base_players[:3]:
                    print(f"  {p}")
            
            if base_fixtures:
                print(f"\nBaseline Fixtures (first 3):")
                for f in base_fixtures[:3]:
                    print(f"  {f}")
            
            if base_teams:
                print(f"\nBaseline Teams (first 3):")
                for t in base_teams[:3]:
                    print(f"  {t}")
            
            # Show embedding results
            emb_players = ctx.get("embedding_players", [])
            if emb_players:
                print(f"\nEmbedding Players (top {len(emb_players)}):")
                for p in emb_players:
                    print(f"  {p['name']:30s} | Pos: {p['position']:3s} | Score: {p['score']:.4f}")
            else:
                print("\nNo embedding results")
        
        print("="*60)
