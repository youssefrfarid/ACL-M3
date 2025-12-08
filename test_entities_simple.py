"""
Simple test to show what's happening with entities
"""
from input_processing import QueryEntities

print("=" * 60)
print("Testing QueryEntities directly")
print("=" * 60)

# Test 1: Create entities directly
print("\nTest 1: Creating QueryEntities directly")
entities1 = QueryEntities(season="2023-24")
print(f"  entities1.season = {entities1.season}")
print(f"  entities1.position = {entities1.position}")
print(f"  entities1.player_names = {entities1.player_names}")
print(f"  entities1.to_dict() = {entities1.to_dict()}")

# Test 2: Create entities with multiple fields
print("\nTest 2: Creating QueryEntities with multiple fields")
entities2 = QueryEntities(
    position="MID", 
    season="2023-24", 
    gameweek=1, 
    horizon_gw=5
)
print(f"  entities2.season = {entities2.season}")
print(f"  entities2.position = {entities2.position}")
print(f"  entities2.gameweek = {entities2.gameweek}")
print(f"  entities2.horizon_gw = {entities2.horizon_gw}")
print(f"  entities2.to_dict() = {entities2.to_dict()}")

# Test 3: Extract entities from a question
print("\nTest 3: Extracting entities from a question")
print("  (This requires loaded players/teams lists)")
from input_processing import extract_entities

# With empty lists
question = "Who are the top midfielders?"
entities3 = extract_entities(question, players=[], teams=[])
print(f"  Question: '{question}'")
print(f"  entities3.position = {entities3.position}")
print(f"  entities3.player_names = {entities3.player_names}")
print(f"  entities3.to_dict() = {entities3.to_dict()}")

print("\n" + "=" * 60)
print("If entities show values above, the issue is NOT with")
print("QueryEntities itself, but with:")
print("  1. Database connection")
print("  2. Season format mismatch (e.g., '2023-24' vs '2023/24')")
print("  3. Empty database")
print("=" * 60)
