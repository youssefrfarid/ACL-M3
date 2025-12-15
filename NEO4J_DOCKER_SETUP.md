# Neo4j Docker Setup Guide

## Quick Setup (Already Running)

A Neo4j Docker container has been started with the following configuration:

**Container Name:** `neo4j-fpl`  
**Username:** `neo4j`  
**Password:** `fplpassword123`  
**Ports:**
- Browser UI: http://localhost:7474
- Bolt connection: bolt://localhost:7687

## Status

The container is currently pulling the Neo4j image and starting up. This may take a couple minutes.

### Check Container Status

```bash
docker ps --filter name=neo4j-fpl
```

You should see the container running with status "Up X seconds/minutes"

### View Container Logs

```bash
docker logs neo4j-fpl
```

Look for the message: "Remote interface available at http://localhost:7474/"

## Access Neo4j

### Browser Interface

1. Open your browser to: http://localhost:7474
2. Login with:
   - Username: `neo4j`
   - Password: `fplpassword123`

### Load FPL Data

Once Neo4j is running, load your FPL dataset:

```bash
# Make sure you're in the project directory
cd /Users/mariammaged/Documents/ACL-M3

# Activate virtual environment (if using one)
source venv/bin/activate

# Run the knowledge graph creation script
python3 create_kg.py
```

This will:
- Read `fpl_two_seasons.csv`
- Create Player, Team, and Fixture nodes
- Create relationships between them
- Build the knowledge graph structure

### Build Embeddings

After loading the data, build the vector embeddings index:

```bash
python3 build_embeddings.py
```

This creates the embedding index used for semantic search.

## Docker Commands Reference

### Stop Neo4j
```bash
docker stop neo4j-fpl
```

### Start Neo4j
```bash
docker start neo4j-fpl
```

### Restart Neo4j
```bash
docker restart neo4j-fpl
```

### Remove Container (Warning: deletes all data)
```bash
docker stop neo4j-fpl
docker rm neo4j-fpl
```

### View Resource Usage
```bash
docker stats neo4j-fpl
```

## Troubleshooting

### Container won't start
```bash
# Check logs for errors
docker logs neo4j-fpl

# Remove and recreate
docker stop neo4j-fpl
docker rm neo4j-fpl

# Run the docker run command again
docker run -d \
  --name neo4j-fpl \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/fplpassword123 \
  -v neo4j-data:/data \
  -v neo4j-logs:/logs \
  neo4j:latest
```

### Port already in use
If ports 7474 or 7687 are already used:
```bash
# Check what's using the port
lsof -i :7474
lsof -i :7687

# Use different ports
docker run -d \
  --name neo4j-fpl \
  -p 7475:7474 \
  -p 7688:7687 \
  -e NEO4J_AUTH=neo4j/fplpassword123 \
  -v neo4j-data:/data \
  neo4j:latest

# Update config.txt to use bolt://localhost:7688
```

### Connection refused in Streamlit
Make sure:
1. Neo4j container is running: `docker ps --filter name=neo4j-fpl`
2. Data is loaded: Run `create_kg.py`
3. Embeddings are built: Run `build_embeddings.py`
4. config.txt has correct credentials

## Next Steps

1. ✅ **Wait for container to finish starting** (check with `docker logs neo4j-fpl`)
2. ⏳ **Access browser UI** at http://localhost:7474
3. ⏳ **Load FPL data** with `python3 create_kg.py`
4. ⏳ **Build embeddings** with `python3 build_embeddings.py`
5. ⏳ **Test Streamlit app** - it's already running!

The Streamlit app should now be able to connect to Neo4j and run queries!
