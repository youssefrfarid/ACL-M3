from neo4j import GraphDatabase

def check_value_prop():
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "fplpassword123")
    driver = GraphDatabase.driver(uri, auth=auth)
    
    print("Checking 'value' property on PLAYED_IN...")
    with driver.session() as session:
        # Check if any relationship has value
        q = "MATCH ()-[r:PLAYED_IN]->() WHERE r.value IS NOT NULL RETURN count(r) as c"
        c = session.run(q).single()["c"]
        print(f"Relationships with 'value': {c}")
        
        # Check sample values
        q2 = "MATCH ()-[r:PLAYED_IN]->() WHERE r.value IS NOT NULL RETURN r.value LIMIT 5"
        res = session.run(q2)
        for rec in res:
            print(f"Value: {rec['r.value']}")

if __name__ == "__main__":
    try:
        check_value_prop()
    except Exception as e:
        print(e)
