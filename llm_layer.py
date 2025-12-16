"""
LLM Layer for FPL Knowledge Graph RAG System

This module provides:
1. Context combination (baseline + embedding results)
2. Structured prompt construction (context + persona + task)
3. Multi-model LLM interface for answer generation
4. OpenRouter API integration for additional models
"""

from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient
import requests
import json
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
            
            # Build stats parts - check all possible field names
            parts = []
            
            # Points (check multiple possible keys)
            points = data.get('total_points') or data.get('points') or data.get('value')
            if points is not None:
                parts.append(f"Points: {points}")
            
            # Position
            position = data.get('position', '')
            
            # Team
            team = data.get('team', '')
            
            # Price
            if "price" in data and data["price"]:
                try:
                    parts.append(f"Price: £{float(data['price']):.1f}m")
                except:
                    parts.append(f"Price: {data['price']}")
            
            # Goals (check multiple keys)
            goals = data.get('goals_scored') or data.get('goals')
            if goals is not None:
                parts.append(f"Goals: {goals}")
            
            # Assists
            if "assists" in data and data["assists"]:
                parts.append(f"Assists: {data['assists']}")
            
            # Clean sheets
            if "clean_sheets" in data and data["clean_sheets"]:
                parts.append(f"Clean Sheets: {data['clean_sheets']}")
            
            # Goals conceded
            if "goals_conceded" in data and data["goals_conceded"]:
                parts.append(f"Conceded: {data['goals_conceded']}")
            
            # Form
            if "form" in data and data["form"]:
                try:
                    parts.append(f"Form: {float(data['form']):.1f}")
                except:
                    parts.append(f"Form: {data['form']}")
            
            # Build the player string
            player_str = f"{idx}. {name}"
            if position:
                player_str += f" ({position})"
            if team:
                player_str += f" - {team}"
            
            # Add stats
            stats_str = ", ".join(parts)
            if stats_str:
                player_str += f"\n   {stats_str}"
            
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


# =====================  OPENROUTER LLM INTERFACE  =====================

def load_openrouter_key(filepath: str = "openrouter_config.txt") -> Optional[str]:
    """Load OpenRouter API key from file."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


class OpenRouterLLMInterface:
    """
    Interface for LLM models via OpenRouter API.
    Supports multiple free models for comparison.
    """
    
    # OpenRouter models (free tier)
    MODELS = {
        "qwen3-coder": "qwen/qwen3-coder:free",
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
        "gemini-flash": "google/gemini-2.0-flash-exp:free",
    }
    
    # Display names for UI
    MODEL_DISPLAY_NAMES = {
        "qwen3-coder": "Qwen3 Coder (Fast)",
        "llama-3.3-70b": "Llama 3.3 70B (Large)",
        "gemini-flash": "Gemini 2.0 Flash (Balanced)",
    }
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize OpenRouter LLM interface.
        
        Args:
            model_name: One of "qwen3-coder", "gpt-oss-120b", "gemini-flash" or full model path
            api_key: OpenRouter API key
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_id = self.MODELS[model_name]
            self.model_key = model_name
        else:
            self.model_id = model_name
            self.model_key = model_name
        
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fpl-rag-system",
            "X-Title": "FPL RAG Chatbot"
        }
        print(f"Initialized OpenRouter LLM: {self.model_id}")
    
    def generate_answer(
        self, 
        prompt: str, 
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """
        Generate answer from prompt using OpenRouter API.
        
        Args:
            prompt: Complete prompt with context
            max_tokens: Maximum response length
            temperature: Sampling temperature (lower = more deterministic)
            max_retries: Maximum number of retries for rate limiting (default 5)
        
        Returns:
            Dict with:
                - answer: Generated text
                - response_time: Time taken (seconds)
                - token_count: Total tokens used
                - prompt_tokens: Tokens in prompt
                - completion_tokens: Tokens in response
                - model: Model name
        """
        start_time = time.time()
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.API_URL,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=120
                )
                
                response_time = time.time() - start_time
                
                # Handle rate limiting with retry
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff: 1s, 2s, 4s, 8s, 10s max
                    print(f"⏳ Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    last_error = f"Rate limited - exhausted {max_retries} retries"
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    return {
                        "answer": error_msg,
                        "response_time": response_time,
                        "token_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "model": self.model_key,
                        "success": False,
                        "error": error_msg
                    }
                
                result = response.json()
                
                # Extract answer
                answer = result["choices"][0]["message"]["content"]
                
                # Extract token usage (OpenRouter provides this)
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                return {
                    "answer": answer,
                    "response_time": response_time,
                    "token_count": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": self.model_key,
                    "success": True,
                    "error": None
                }
            
            except requests.exceptions.Timeout:
                last_error = "Request timed out after 120 seconds"
                continue
            
            except Exception as e:
                last_error = str(e)
                continue
        
        # If we exhausted all retries
        response_time = time.time() - start_time
        return {
            "answer": f"Failed after {max_retries} retries: {last_error}",
            "response_time": response_time,
            "token_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": self.model_key,
            "success": False,
            "error": last_error
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


def generate_openrouter_answer(
    question: str,
    hybrid_results: Dict[str, Any],
    model_name: str = "gemini-flash",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline using OpenRouter: combine context, build prompt, generate answer.
    
    Args:
        question: User's question
        hybrid_results: Output from retrieve_hybrid()
        model_name: OpenRouter model to use ("qwen3-coder", "gpt-oss-120b", "gemini-flash")
        api_key: OpenRouter API key (if None, loads from file)
    
    Returns:
        Dict with answer, metadata, and context
    """
    # Load API key if not provided
    if not api_key:
        api_key = load_openrouter_key()
        if not api_key:
            raise ValueError("OpenRouter API key required. Create openrouter_config.txt or pass key.")
    
    # Step 1: Combine contexts
    context = combine_contexts(hybrid_results)
    
    # Step 2: Build prompt
    prompt = build_fpl_prompt(question, context)
    
    # Step 3: Generate answer
    llm = OpenRouterLLMInterface(model_name, api_key)
    result = llm.generate_answer(prompt)
    
    # Add context for transparency
    result["context"] = context
    result["prompt_length"] = len(prompt)
    result["provider"] = "openrouter"
    
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
