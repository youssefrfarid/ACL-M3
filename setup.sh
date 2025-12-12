#!/bin/bash

# Quick Setup Script for Neo4j and FPL Data
# Run this script to set up the complete FPL Graph-RAG system

echo "========================================="
echo "FPL Graph-RAG System Setup"
echo "========================================="
echo ""

# Step 1: Check if Neo4j is running
echo "Step 1: Checking Neo4j status..."
if docker ps --filter name=neo4j-fpl | grep -q neo4j-fpl; then
    echo "✅ Neo4j is running"
else
    echo "❌ Neo4j is not running. Starting it now..."
    docker start neo4j-fpl || {
        echo "Creating new Neo4j container..."
        docker run -d --name neo4j-fpl \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/fplpassword123 \
            -v neo4j-data:/data \
            -v neo4j-logs:/logs \
            neo4j:latest
    }
    echo "Waiting for Neo4j to be ready..."
    sleep 10
fi

echo ""
echo "Step 2: Loading FPL data into Neo4j..."
python3 create_kg.py

echo ""
echo "Step 3: Building vector embeddings..."
python3 build_embeddings.py

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "You can now:"
echo "  1. Access Neo4j Browser: http://localhost:7474"
echo "     Username: neo4j"
echo "     Password: fplpassword123"
echo ""
echo "  2. Run Streamlit UI: streamlit run streamlit_app.py"
echo ""
echo "  3. Test CLI chatbot: python3 fpl_chatbot.py"
echo ""
