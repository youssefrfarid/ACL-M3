#!/bin/bash

# ============================================================
# FPL Graph-RAG Pipeline Setup Script
# ============================================================
# This script sets up and runs the complete FPL Graph-RAG pipeline:
# 1. Activates virtual environment
# 2. Starts Neo4j Docker container
# 3. Creates the Knowledge Graph
# 4. Builds player embeddings
# 5. Runs the Streamlit app
# ============================================================

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}⚽ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║           ⚽ FPL Graph-RAG Pipeline Setup ⚽              ║"
echo "  ║    Knowledge Graph-powered Fantasy Premier League        ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================
# Step 1: Check prerequisites
# ============================================================
print_step "Step 1: Checking Prerequisites"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker found"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi
print_success "Python3 found"

# Check if venv exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# ============================================================
# Step 2: Activate Virtual Environment
# ============================================================
print_step "Step 2: Activating Virtual Environment"

source venv/bin/activate
print_success "Virtual environment activated"

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip install -q -r requirements.txt
    print_success "Requirements installed"
fi

# ============================================================
# Step 3: Start Neo4j Docker Container
# ============================================================
print_step "Step 3: Starting Neo4j Docker Container"

CONTAINER_NAME="neo4j-fpl"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container exists, check if running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_success "Neo4j container is already running"
    else
        echo "Starting existing Neo4j container..."
        docker start $CONTAINER_NAME
        print_success "Neo4j container started"
    fi
else
    # Create new container
    echo "Creating new Neo4j container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/fplpassword123 \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
        neo4j:5.13.0
    print_success "Neo4j container created and started"
fi

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
while ! docker exec $CONTAINER_NAME wget -q --spider http://localhost:7474 2>/dev/null; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        print_error "Neo4j failed to start within expected time"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
print_success "Neo4j is ready at http://localhost:7474"

# ============================================================
# Step 4: Check Configuration Files
# ============================================================
print_step "Step 4: Checking Configuration Files"

# Check config.txt
if [ ! -f "config.txt" ]; then
    echo "Creating config.txt..."
    cat > config.txt << EOF
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=fplpassword123
EOF
    print_success "config.txt created"
else
    print_success "config.txt exists"
fi

# Check hf.txt
if [ ! -f "hf.txt" ]; then
    print_warning "hf.txt not found!"
    echo "Please create hf.txt with your HuggingFace API token."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    read -p "Enter your HuggingFace token (or press Enter to skip embeddings): " HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" > hf.txt
        print_success "hf.txt created"
    else
        print_warning "Skipping embeddings step (no HF token)"
        SKIP_EMBEDDINGS=true
    fi
else
    print_success "hf.txt exists"
fi

# Check openrouter_config.txt
if [ ! -f "openrouter_config.txt" ]; then
    print_warning "openrouter_config.txt not found!"
    echo "For OpenRouter LLM support, create openrouter_config.txt with your API key."
    echo "Get your key from: https://openrouter.ai/keys"
fi

# ============================================================
# Step 5: Create Knowledge Graph
# ============================================================
print_step "Step 5: Creating Knowledge Graph"

# Check if data file exists
if [ ! -f "fpl_two_seasons.csv" ]; then
    print_error "fpl_two_seasons.csv not found! Please add the data file."
    exit 1
fi

echo "Building Knowledge Graph from fpl_two_seasons.csv..."
python3 create_kg.py

print_success "Knowledge Graph created"

# ============================================================
# Step 6: Build Embeddings
# ============================================================
if [ "${SKIP_EMBEDDINGS:-false}" != "true" ]; then
    print_step "Step 6: Building Player Embeddings"
    
    echo "This may take 5-10 minutes for 1500+ players..."
    python3 build_embeddings.py
    
    print_success "Embeddings built"
else
    print_warning "Skipping embeddings (no HuggingFace token)"
fi

# ============================================================
# Step 7: Run Streamlit App
# ============================================================
print_step "Step 7: Launching Streamlit App"

echo -e "${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║              🏆 Pipeline Setup Complete! 🏆               ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "📊 Neo4j Browser:     http://localhost:7474"
echo "🚀 Streamlit App:     http://localhost:8501"
echo ""
echo "Starting Streamlit app..."
echo "Press Ctrl+C to stop the app."
echo ""

streamlit run streamlit_app.py
