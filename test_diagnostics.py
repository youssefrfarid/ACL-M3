"""
Diagnostic test to check Neo4j connection and data
"""
from graph_retrieval import get_driver
from input_processing import QueryEntities
from graph_retrieval import run_baseline_retrieval

print("=" * 70)
print("DIAGNOSTIC TEST - Checking Neo4j Connection and Data")
print("=" * 70)

# Test 1: Check connection
print("\n1. Testing Neo4j connection...")
try:
    driver = get_driver()
    driver.verify_connectivity()
    print("   ✓ Connected to Neo4j successfully")
except Exception as e:
    print(f"   ✗ Connection failed: {e}")
    exit(1)

# Test 2: Check what seasons exist in the database
print("\n2. Checking available seasons in database...")
with driver.session() as session:
    result = session.run("MATCH (f:Fixture) RETURN DISTINCT f.season AS season ORDER BY season")
    seasons = [rec["season"] for rec in result]
    if seasons:
        print(f"   ✓ Found {len(seasons)} seasons: {seasons}")
    else:
        print("   ✗ No seasons found in database!")

# Test 3: Count total players
print("\n3. Counting players...")
with driver.session() as session:
    result = session.run("MATCH (p:Player) RETURN count(p) AS count")
    count = result.single()["count"]
    print(f"   ✓ Found {count} players in database")

# Test 4: Count total fixtures
print("\n4. Counting fixtures...")
with driver.session() as session:
    result = session.run("MATCH (f:Fixture) RETURN count(f) AS count")
    count = result.single()["count"]
    print(f"   ✓ Found {count} fixtures in database")

# Test 5: Sample a few players
print("\n5. Sampling 5 random players...")
with driver.session() as session:
    result = session.run("MATCH (p:Player) RETURN p.player_name AS name LIMIT 5")
    players = [rec["name"] for rec in result]
    for i, player in enumerate(players, 1):
        print(f"   {i}. {player}")

# Test 6: Check positions
print("\n6. Checking available positions...")
with driver.session() as session:
    result = session.run("MATCH (pos:Position) RETURN pos.name AS name ORDER BY name")
    positions = [rec["name"] for rec in result]
    print(f"   ✓ Positions: {positions}")

# Test 7: Test a simple query with actual season from DB
print("\n7. Testing query with actual season data...")
if seasons:
    test_season = seasons[0]  # Use first available season
    print(f"   Using season: {test_season}")
    
    entities = QueryEntities(season=test_season)
    try:
        result = run_baseline_retrieval("general_question", entities)
        print(f"   ✓ Template: {result['template']}")
        print(f"   ✓ Found {len(result.get('players', []))} players")
        if result.get('players'):
            print(f"   Sample result: {result['players'][0]}")
        else:
            print("   ⚠ Query ran but returned 0 players")
    except Exception as e:
        print(f"   ✗ Query failed: {e}")
        import traceback
        traceback.print_exc()

# Test 8: Test position query
print("\n8. Testing position-based query...")
if seasons and positions:
    test_season = seasons[0]
    test_position = positions[0] if positions else "MID"
    print(f"   Using season: {test_season}, position: {test_position}")
    
    entities = QueryEntities(position=test_position, season=test_season, gameweek=1, horizon_gw=5)
    try:
        result = run_baseline_retrieval("player_recommendation", entities)
        print(f"   ✓ Template: {result['template']}")
        print(f"   ✓ Found {len(result.get('players', []))} players")
        if result.get('players'):
            print(f"   Sample result: {result['players'][0]}")
        else:
            print("   ⚠ Query ran but returned 0 players")
    except Exception as e:
        print(f"   ✗ Query failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 70)
