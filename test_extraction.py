#!/usr/bin/env python3
"""Quick test for enhanced entity extraction"""

from input_processing import (
    classify_intent,
    extract_entities,
    load_known_players_and_teams
)

# Load data
print("Loading players and teams from Neo4j...")
players, teams = load_known_players_and_teams()
print(f"✓ Loaded {len(players)} players and {len(teams)} teams\n")

# Test query
test_query = "Compare between salah and halaand in terms of goals in the 2022/2023 season"

print("=" * 70)
print(f"Query: {test_query}")
print("=" * 70)

# Extract
intent = classify_intent(test_query)
entities = extract_entities(test_query, players, teams)

print(f"\nIntent: {intent}")
print(f"Entities: {entities.to_dict()}")
print("\nPlayer names extracted:")
for player in entities.player_names:
    print(f"  ✓ {player}")

print("\n✅ Test complete!")
