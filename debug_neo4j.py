from neo4j import GraphDatabase
import os

def get_driver():
    uri = "bolt://localhost:7687" # Assumed based on create_kg.py default log
    auth = ("neo4j", "fplpassword123")
    return GraphDatabase.driver(uri, auth=auth)

def check_counts():
    print("Checking Node Counts...")
    with get_driver().session() as session:
        for label in ["Player", "Team", "Fixture", "Gameweek", "Season"]:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
            print(f"{label}: {count}")

def check_relationships():
    print("\nChecking Relationships...")
    with get_driver().session() as session:
        rels = ["PLAYED_IN", "HAS_HOME_TEAM", "HAS_AWAY_TEAM", "HAS_FIXTURE", "PLAYS_AS"]
        for r in rels:
            count = session.run(f"MATCH ()-[r:{r}]->() RETURN count(r) as c").single()["c"]
            print(f"{r}: {count}")

def check_value_prop():
    print("\nChecking 'value' property on PLAYED_IN...")
    with get_driver().session() as session:
        q = "MATCH ()-[r:PLAYED_IN]->() WHERE r.value IS NOT NULL RETURN count(r) as c"
        c = session.run(q).single()["c"]
        print(f"Relationships with 'value': {c}")

def check_t6_logic():
    print("\nChecking T6 Query logic matches...")
    with get_driver().session() as session:
        # Check simple pattern
        q = """
        MATCH (t:Team)
        MATCH (f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        WHERE (f)-[:HAS_HOME_TEAM]->(t)
        RETURN count(r) as c LIMIT 1
        """
        c = session.run(q).single()["c"]
        print(f"Pattern (p)-[PLAYED_IN]->(f)-[HAS_HOME_TEAM]->(t) count: {c}")

if __name__ == "__main__":
    try:
        check_counts()
        check_relationships()
        check_value_prop()
        check_t6_logic()
    except Exception as e:
        print(e)
