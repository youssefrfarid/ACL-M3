

from typing import Dict, Any, List
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
    """
    season = entities.season or "2023-24"
    limit = 20

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
    
    Args:
        intent: Intent string from classify_intent()
        entities: QueryEntities object from extract_entities()
    
    Returns:
        Dict containing:
            - template: name of the template used
            - players/fixtures/teams: list of result dicts
            - mode: "baseline"
            - intent: original intent
            - entities: serialized QueryEntities
    """
    with get_driver().session() as session:
        if intent == INTENT_PLAYER_INFO:
            # Smart routing based on what info we have
            if entities.gameweek is not None or entities.horizon_gw is not None:
                # User asked about specific gameweeks - use recent form
                result = q_player_recent_form(session, entities)
            elif entities.player_names:
                # Specific player mentioned - use season summary
                result = q_player_season_summary(session, entities)
            else:
                # No specific player - fallback to top scorers
                result = q_top_scorers_overall(session, entities)

        elif intent == INTENT_COMPARE_PLAYERS:
            result = q_compare_two_players(session, entities)

        elif intent == INTENT_FIXTURE_INFO:
            result = q_team_fixtures_range(session, entities)

        elif intent == INTENT_PLAYER_RECOMMEND:
            # Use recent form for recommendations
            result = q_top_players_recent_form_position(session, entities)

        elif intent == INTENT_TEAM_RECOMMEND:
            result = q_team_defensive_strength(session, entities)

        else:  # INTENT_GENERAL_QUESTION or unsupported intents
            # Smart routing for general questions
            if entities.position:
                # Position specified - use position-based leaderboard
                result = q_top_players_by_position(session, entities)
            elif entities.stat_name:
                # Specific stat requested - use stat leaderboard
                result = q_leaderboard_by_stat(session, entities)
            else:
                # Generic question - top scorers overall
                result = q_top_scorers_overall(session, entities)

    # Wrap with metadata for the LLM layer
    result["mode"] = "baseline"
    result["intent"] = intent
    result["entities"] = entities.to_dict()
    return result




# MANUAL TESTING

if __name__ == "__main__":
    from input_processing import classify_intent, extract_entities, load_known_players_and_teams

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

    while True:
        q = input("\nAsk an FPL question (or 'exit'): ")
        if q.strip().lower() == "exit":
            break

        intent = classify_intent(q)
        entities = extract_entities(q, players, teams)

        ctx = run_baseline_retrieval(intent, entities)
        
        print("\n" + "="*60)
        print("Intent:", ctx["intent"])
        print("Entities:", ctx["entities"])
        print("Template:", ctx["template"])
        print("Mode:", ctx["mode"])
        
        # Display sample results
        players_result = ctx.get("players", [])
        fixtures_result = ctx.get("fixtures", [])
        teams_result = ctx.get("teams", [])
        
        if players_result:
            print(f"\nPlayers (showing first 3 of {len(players_result)}):")
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
        
        print("="*60)
