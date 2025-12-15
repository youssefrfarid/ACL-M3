#!/usr/bin/env python3
"""
FPL Knowledge Graph Creator
Creates a Neo4j knowledge graph from FPL CSV data
"""

import csv
from neo4j import GraphDatabase
import sys
from datetime import datetime


def read_config(config_file='config.txt'):
    """Read Neo4j configuration from config.txt file"""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found!")
        sys.exit(1)


def load_csv_data(csv_file='fpl_two_seasons.csv'):
    """Load and parse CSV data"""
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"Loaded {len(data)} rows from {csv_file}")
        return data
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        sys.exit(1)


class KnowledgeGraphBuilder:
    def __init__(self, uri, username, password):
        """Initialize Neo4j driver"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the driver connection"""
        self.driver.close()
        
    def verify_connection(self):
        """Verify Neo4j connection"""
        try:
            self.driver.verify_connectivity()
            print("✓ Successfully connected to Neo4j")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def clear_database(self):
        """Clear all existing data (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")
    
    def create_constraints(self):
        """Create uniqueness constraints for node identification"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT season_name IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE",
                "CREATE CONSTRAINT gameweek_id IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE",
                "CREATE CONSTRAINT fixture_id IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE",
                "CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT player_id IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE",
                "CREATE CONSTRAINT position_name IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
            print("✓ Constraints created")
    
    def create_positions(self, data):
        """Create Position nodes - dynamically extracted from data"""
        with self.driver.session() as session:
            # Extract unique positions from the CSV data
            positions = set(row['position'] for row in data)
            
            for pos in positions:
                session.run(
                    "MERGE (p:Position {name: $name})",
                    name=pos
                )
            print(f"✓ Created {len(positions)} Position nodes: {sorted(positions)}")
    
    def create_seasons(self, data):
        """Create Season nodes"""
        with self.driver.session() as session:
            seasons = set(row['season'] for row in data)
            for season in seasons:
                session.run(
                    "MERGE (s:Season {season_name: $season_name})",
                    season_name=season
                )
            print(f"✓ Created {len(seasons)} Season nodes")
    
    def create_teams(self, data):
        """Create Team nodes"""
        with self.driver.session() as session:
            teams = set()
            for row in data:
                teams.add(row['home_team'])
                teams.add(row['away_team'])
            
            for team in teams:
                session.run(
                    "MERGE (t:Team {name: $name})",
                    name=team
                )
            print(f"✓ Created {len(teams)} Team nodes")
    
    def create_gameweeks(self, data):
        """Create Gameweek nodes and HAS_GW relationships"""
        with self.driver.session() as session:
            gameweeks = set((row['season'], int(row['GW'])) for row in data)
            for season, gw_num in gameweeks:
                session.run("""
                    MATCH (s:Season {season_name: $season})
                    MERGE (g:Gameweek {season: $season, GW_number: $gw_num})
                    MERGE (s)-[:HAS_GW]->(g)
                    """,
                    season=season,
                    gw_num=gw_num
                )
            print(f"✓ Created {len(gameweeks)} Gameweek nodes")
    
    def create_fixtures(self, data):
        """Create Fixture nodes and relationships"""
        with self.driver.session() as session:
            fixtures = {}
            for row in data:
                fixture_key = (row['season'], int(row['fixture']))
                if fixture_key not in fixtures:
                    fixtures[fixture_key] = {
                        'season': row['season'],
                        'fixture_number': int(row['fixture']),
                        'kickoff_time': row['kickoff_time'],
                        'gw_number': int(row['GW']),
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'home_score': int(row['team_h_score']) if row['team_h_score'] else 0,
                        'away_score': int(row['team_a_score']) if row['team_a_score'] else 0
                    }
            
            for fixture_data in fixtures.values():
                session.run("""
                    MATCH (g:Gameweek {season: $season, GW_number: $gw_number})
                    MATCH (home:Team {name: $home_team})
                    MATCH (away:Team {name: $away_team})
                    MERGE (f:Fixture {season: $season, fixture_number: $fixture_number})
                    SET f.kickoff_time = $kickoff_time,
                        f.home_score = $home_score,
                        f.away_score = $away_score
                    MERGE (g)-[:HAS_FIXTURE]->(f)
                    MERGE (f)-[:HAS_HOME_TEAM]->(home)
                    MERGE (f)-[:HAS_AWAY_TEAM]->(away)
                    """,
                    **fixture_data
                )
            print(f"✓ Created {len(fixtures)} Fixture nodes")
    
    def create_players(self, data):
        """Create Player nodes and PLAYS_AS relationships"""
        with self.driver.session() as session:
            players = {}
            for row in data:
                player_key = (row['name'], int(row['element']))
                if player_key not in players:
                    players[player_key] = {
                        'player_name': row['name'],
                        'player_element': int(row['element']),
                        'position': row['position']
                    }
            
            for player_data in players.values():
                session.run("""
                    MATCH (pos:Position {name: $position})
                    MERGE (p:Player {player_name: $player_name, player_element: $player_element})
                    MERGE (p)-[:PLAYS_AS]->(pos)
                    """,
                    **player_data
                )
            print(f"✓ Created {len(players)} Player nodes")
    
    def create_player_fixture_relationships(self, data):
        """Create PLAYED_IN relationships with all properties"""
        with self.driver.session() as session:
            count = 0
            batch_size = 1000
            batch = []
            
            for row in data:
                # Convert numeric fields
                played_in_data = {
                    'player_name': row['name'],
                    'player_element': int(row['element']),
                    'season': row['season'],
                    'fixture_number': int(row['fixture']),
                    'minutes': int(row['minutes']) if row['minutes'] else 0,
                    'goals_scored': int(row['goals_scored']) if row['goals_scored'] else 0,
                    'assists': int(row['assists']) if row['assists'] else 0,
                    'total_points': int(row['total_points']) if row['total_points'] else 0,
                    'bonus': int(row['bonus']) if row['bonus'] else 0,
                    'clean_sheets': int(row['clean_sheets']) if row['clean_sheets'] else 0,
                    'goals_conceded': int(row['goals_conceded']) if row['goals_conceded'] else 0,
                    'own_goals': int(row['own_goals']) if row['own_goals'] else 0,
                    'penalties_saved': int(row['penalties_saved']) if row['penalties_saved'] else 0,
                    'penalties_missed': int(row['penalties_missed']) if row['penalties_missed'] else 0,
                    'yellow_cards': int(row['yellow_cards']) if row['yellow_cards'] else 0,
                    'red_cards': int(row['red_cards']) if row['red_cards'] else 0,
                    'saves': int(row['saves']) if row['saves'] else 0,
                    'bps': int(row['bps']) if row['bps'] else 0,
                    'influence': float(row['influence']) if row['influence'] else 0.0,
                    'creativity': float(row['creativity']) if row['creativity'] else 0.0,
                    'threat': float(row['threat']) if row['threat'] else 0.0,
                    'ict_index': float(row['ict_index']) if row['ict_index'] else 0.0,
                    'form': float(row['form']) if row['form'] else 0.0,
                    'value': int(row['value']) if row['value'] else 0,
                }
                
                batch.append(played_in_data)
                
                if len(batch) >= batch_size:
                    self._execute_played_in_batch(session, batch)
                    count += len(batch)
                    print(f"  Processed {count} PLAYED_IN relationships...")
                    batch = []
            
            # Process remaining batch
            if batch:
                self._execute_played_in_batch(session, batch)
                count += len(batch)
            
            print(f"✓ Created {count} PLAYED_IN relationships")
    
    def _execute_played_in_batch(self, session, batch):
        """Execute a batch of PLAYED_IN relationship creation"""
        session.run("""
            UNWIND $batch AS row
            MATCH (p:Player {player_name: row.player_name, player_element: row.player_element})
            MATCH (f:Fixture {season: row.season, fixture_number: row.fixture_number})
            MERGE (p)-[r:PLAYED_IN]->(f)
            SET r.minutes = row.minutes,
                r.goals_scored = row.goals_scored,
                r.assists = row.assists,
                r.total_points = row.total_points,
                r.bonus = row.bonus,
                r.clean_sheets = row.clean_sheets,
                r.goals_conceded = row.goals_conceded,
                r.own_goals = row.own_goals,
                r.penalties_saved = row.penalties_saved,
                r.penalties_missed = row.penalties_missed,
                r.yellow_cards = row.yellow_cards,
                r.red_cards = row.red_cards,
                r.saves = row.saves,
                r.bps = row.bps,
                r.influence = row.influence,
                r.creativity = row.creativity,
                r.threat = row.threat,
                r.ict_index = row.ict_index,
                r.form = row.form,
                r.value = row.value
            """,
            batch=batch
        )
    
    def build_graph(self, data):
        """Build the complete knowledge graph"""
        print("\n=== Building FPL Knowledge Graph ===\n")
        
        self.clear_database()
        self.create_constraints()
        
        # Create nodes in order
        self.create_positions(data)
        self.create_seasons(data)
        self.create_teams(data)
        self.create_gameweeks(data)
        self.create_fixtures(data)
        self.create_players(data)
        
        # Create relationships
        self.create_player_fixture_relationships(data)
        
        print("\n✓ Knowledge Graph successfully created!\n")


def main():
    """Main execution function"""
    print("FPL Knowledge Graph Builder")
    print("=" * 50)
    
    # Read configuration
    config = read_config()
    uri = config.get('URI')
    username = config.get('USERNAME')
    password = config.get('PASSWORD')
    
    if not all([uri, username, password]):
        print("Error: config.txt must contain URI, USERNAME, and PASSWORD")
        sys.exit(1)
    
    # Load CSV data
    data = load_csv_data()
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder(uri, username, password)
    
    try:
        if not builder.verify_connection():
            print("\nPlease ensure Neo4j is running and credentials are correct.")
            sys.exit(1)
        
        builder.build_graph(data)
        
        print("You can now access Neo4j Browser at: http://localhost:7474")
        print("Username: neo4j")
        print("Password: fplpassword123")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
