from neo4j import GraphDatabase

def check_fixture_scores():
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "fplpassword123")
    driver = GraphDatabase.driver(uri, auth=auth)
    
    print("Checking Fixture Scores...")
    with driver.session() as session:
        # Check if any fixture has home_score
        q = "MATCH (f:Fixture) WHERE f.home_score IS NOT NULL RETURN count(f) as c"
        c = session.run(q).single()["c"]
        print(f"Fixtures with scores: {c}")
        
        # Check sample values
        q2 = "MATCH (f:Fixture) RETURN f.home_score, f.away_score LIMIT 5"
        res = session.run(q2)
        for rec in res:
            print(f"H: {rec['f.home_score']}, A: {rec['f.away_score']}")

if __name__ == "__main__":
    try:
        check_fixture_scores()
    except Exception as e:
        print(e)
