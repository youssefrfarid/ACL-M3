"""
LLM Layer for FPL Knowledge Graph RAG System

This module provides:
1. Context combination (baseline + embedding results)
2. Structured prompt construction (context + persona + task)
3. Multi-model LLM interface for answer generation
"""

from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient
import time


# =====================  CONTEXT COMBINATION  =====================

def combine_contexts(hybrid_results: Dict[str, Any]) -> str:
    """
    Combine baseline and embedding retrieval results into unified context.
    
    Args:
        hybrid_results: Output from retrieve_hybrid() containing:
            - baseline_players: List of player dicts from Cypher
            - baseline_fixtures: List of fixture dicts
            - baseline_teams: List of team dicts
            - embedding_players: List of player dicts with similarity scores
    
    Returns:
        Formatted context string for LLM prompt
    """
    context_parts = []
    
    # Extract results
    baseline_players = hybrid_results.get("baseline_players", [])
    baseline_fixtures = hybrid_results.get("baseline_fixtures", [])
    baseline_teams = hybrid_results.get("baseline_teams", [])
    embedding_players = hybrid_results.get("embedding_players", [])
    
    # 1. Combine and deduplicate players
    all_players = {}
    
    # Add baseline players (priority)
    for p in baseline_players:
        player_name = p.get("name")
        if player_name:
            all_players[player_name] = {
                "source": "baseline",
                "data": p
            }
    
    # Add embedding players if not already present
    for p in embedding_players:
        player_name = p.get("name")
        if player_name and player_name not in all_players:
            all_players[player_name] = {
                "source": "embedding",
                "data": p
            }
    
    # Format players
    if all_players:
        context_parts.append("=== PLAYERS ===")
        for idx, (name, info) in enumerate(list(all_players.items())[:10], 1):
            data = info["data"]
            player_str = f"{idx}. {name}"
            
            # Add available stats
            if "position" in data:
                player_str += f" ({data['position']})"
            if "team" in data and data["team"]:
                player_str += f" - {data['team']}"
            if "total_points" in data:
                player_str += f"\n   Points: {data['total_points']}"
            if "goals" in data:
                player_str += f", Goals: {data['goals']}"
            # Reorder specific stats to front for LLM visibility
            parts = []
            if "total_points" in data:
                parts.append(f"Points: {data['total_points']}")
            if "price" in data:
                parts.append(f"Price: £{data['price']:.1f}m")
            if "goals_scored" in data:
                parts.append(f"Scored: {data['goals_scored']}")
            if "clean_sheets" in data:
                parts.append(f"Clean Sheets: {data['clean_sheets']}")
            if "goals_conceded" in data:
                parts.append(f"Conceded: {data['goals_conceded']}")
            if "form" in data and data["form"]:
                parts.append(f"Form: {data['form']:.1f}")
            
            # Combine
            stats_str = ", ".join(parts)
            player_str += f" - {stats_str}"
            
            if "score" in data:  # Embedding similarity
                player_str += f"\n   Similarity: {data['score']:.3f}"
            
            context_parts.append(player_str)
        context_parts.append("")
    
    # 2. Format fixtures
    if baseline_fixtures:
        context_parts.append("=== FIXTURES ===")
        for idx, fixture in enumerate(baseline_fixtures[:10], 1):
            fix_str = f"{idx}. GW {fixture.get('gw', 'N/A')}: {fixture.get('home_team', 'N/A')} vs {fixture.get('away_team', 'N/A')}"
            if "kickoff" in fixture and fixture["kickoff"]:
                fix_str += f" ({fixture['kickoff']})"
            context_parts.append(fix_str)
        context_parts.append("")
    
    # 3. Format teams
    if baseline_teams:
        context_parts.append("=== TEAMS ===")
        for idx, team in enumerate(baseline_teams[:10], 1):
            team_str = f"{idx}. {team.get('team', 'N/A')}"
            if "goals_conceded" in team:
                team_str += f" - Goals Conceded: {team['goals_conceded']}"
            if "total_points" in team:
                team_str += f", Points: {team['total_points']}"
            context_parts.append(team_str)
        context_parts.append("")
    
    # If no results
    if not context_parts:
        return "No relevant data found in the knowledge graph."
    
    return "\n".join(context_parts)


# =====================  PROMPT CONSTRUCTION  =====================

FPL_PERSONA = """You are an expert Fantasy Premier League (FPL) assistant with deep knowledge of player statistics, team performance, and FPL strategy. You provide accurate, data-driven advice based on real statistics."""

PROMPT_TEMPLATE = """{persona}

Context (Retrieved from FPL Knowledge Graph):
{context}

Instructions:
- Answer the user's question using ONLY the information provided in the context above
- Be specific and cite statistics when available (e.g., "Haaland scored 36 goals and earned 238 points")
- If the context doesn't contain the information needed, say "I don't have that information in the available data"
- Note: For 'Goals Conceded' and 'Defensive Strength', a LOWER number indicates a BETTER defense.
- The players are listed in descending order of relevance. The first player (1.) is typically the best recommendation.
- When recommending players, prioritize those with the highest Points or Form.
- If a budget is specified, choose the highest-scoring player that fits the budget.
- Explicitly mention the Price and Points to justify the choice.
- Keep answers concise but informative
- DO NOT make up or hallucinate information not present in the context

User Question: {question}

Answer:"""


def build_fpl_prompt(question: str, context: str, persona: str = FPL_PERSONA) -> str:
    """
    Build structured prompt for LLM with context, persona, and task.
    
    Args:
        question: User's question
        context: Retrieved KG context
        persona: System persona/role definition
    
    Returns:
        Complete formatted prompt
    """
    return PROMPT_TEMPLATE.format(
        persona=persona,
        context=context,
        question=question
    )


# =====================  LLM INTERFACE  =====================

class FPLLLMInterface:
    """
    Unified interface for multiple LLM models via HuggingFace Inference API.
    Supports multiple models for comparison.
    """
    
    # Supported models (all free via HF Inference API)
    MODELS = {
        "gemma": "google/gemma-2-2b-it",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
    }
    
    def __init__(self, model_name: str, hf_token: str):
        """
        Initialize LLM interface.
        
        Args:
            model_name: One of "gemma", "mistral", "phi3" or full model path
            hf_token: HuggingFace API token
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_name = self.MODELS[model_name]
            self.model_key = model_name
        else:
            self.model_name = model_name
            self.model_key = model_name
        
        self.client = InferenceClient(token=hf_token)
        print(f"Initialized LLM: {self.model_name}")
    
    def generate_answer(
        self, 
        prompt: str, 
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate answer from prompt.
        
        Args:
            prompt: Complete prompt with context
            max_tokens: Maximum response length
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Dict with:
                - answer: Generated text
                - response_time: Time taken (seconds)
                - token_count: Estimated token count
                - model: Model name
        """
        start_time = time.time()
        
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            answer = response.choices[0].message["content"]
            response_time = time.time() - start_time
            
            # Estimate token count (rough: ~4 chars per token)
            token_count = len(answer) // 4
            
            return {
                "answer": answer,
                "response_time": response_time,
                "token_count": token_count,
                "model": self.model_key,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "answer": f"Error generating response: {str(e)}",
                "response_time": response_time,
                "token_count": 0,
                "model": self.model_key,
                "success": False,
                "error": str(e)
            }


# =====================  END-TO-END PIPELINE  =====================

def generate_fpl_answer(
    question: str,
    hybrid_results: Dict[str, Any],
    model_name: str = "gemma",
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline: combine context, build prompt, generate answer.
    
    Args:
        question: User's question
        hybrid_results: Output from retrieve_hybrid()
        model_name: LLM model to use ("gemma", "mistral", "phi3")
        hf_token: HuggingFace token (if None, loads from file)
    
    Returns:
        Dict with answer, metadata, and context
    """
    # Load token if not provided
    if not hf_token:
        from input_processing import load_hf_token
        hf_token = load_hf_token()
        if not hf_token:
            raise ValueError("HuggingFace token required. Create hf.txt or pass token.")
    
    # Step 1: Combine contexts
    context = combine_contexts(hybrid_results)
    
    # Step 2: Build prompt
    prompt = build_fpl_prompt(question, context)
    
    # Step 3: Generate answer
    llm = FPLLLMInterface(model_name, hf_token)
    result = llm.generate_answer(prompt)
    
    # Add context for transparency
    result["context"] = context
    result["prompt_length"] = len(prompt)
    
    return result


# =====================  MANUAL TESTING  =====================

if __name__ == "__main__":
    print("Testing LLM Layer...")
    print("=" * 60)
    
    # Mock hybrid results for testing
    mock_results = {
        "baseline_players": [
            {"name": "Erling Haaland", "position": "FWD", "total_points": 238, 
             "goals": 36, "assists": 12, "form": 8.2},
            {"name": "Mohamed Salah", "position": "MID", "total_points": 211,
             "goals": 18, "assists": 13, "form": 7.5}
        ],
        "embedding_players": [
            {"name": "Bukayo Saka", "position": "MID", "score": 0.85},
        ],
        "baseline_fixtures": [],
        "baseline_teams": []
    }
    
    # Test 1: Context combination
    print("\n1. Testing context combination...")
    context = combine_contexts(mock_results)
    print(context)
    
    # Test 2: Prompt building
    print("\n2. Testing prompt building...")
    question = "Who scored the most points?"
    prompt = build_fpl_prompt(question, context)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"First 200 chars: {prompt[:200]}...")
    
    # Test 3: LLM generation (requires token)
    print("\n3. Testing LLM generation...")
    try:
        from input_processing import load_hf_token
        token = load_hf_token()
        if token:
            result = generate_fpl_answer(question, mock_results, model_name="gemma")
            print(f"\nAnswer: {result['answer']}")
            print(f"Response time: {result['response_time']:.2f}s")
            print(f"Model: {result['model']}")
        else:
            print("Skipping LLM test - no token found")
    except Exception as e:
        print(f"LLM test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ LLM Layer tests complete!")
