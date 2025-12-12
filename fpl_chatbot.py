"""
FPL Chatbot - Interactive RAG Application

Complete end-to-end pipeline:
1. User question input
2. Intent classification & entity extraction
3. Hybrid retrieval (baseline + embeddings)
4. Context combination
5. LLM answer generation

Usage:
    python fpl_chatbot.py
"""

import sys
from typing import Optional

from input_processing import (
    classify_intent,
    extract_entities,
    load_known_players_and_teams,
    get_query_embedding,
    load_hf_token
)
from graph_retrieval import retrieve_hybrid
from llm_layer import generate_fpl_answer, FPLLLMInterface


# =====================  CONFIGURATION  =====================

AVAILABLE_MODELS = {
    "1": {"name": "gemma", "display": "Gemma 2B (Fast, Lightweight)"},
    "2": {"name": "mistral", "display": "Mistral 7B (High Quality)"},
    "3": {"name": "phi3", "display": "Phi-3 Mini (Balanced)"},
}

DEFAULT_MODEL = "gemma"


# =====================  CHATBOT CLASS  =====================

class FPLChatbot:
    """Interactive FPL chatbot with full RAG pipeline"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, show_context: bool = False):
        """
        Initialize chatbot.
        
        Args:
            model_name: LLM model to use ("gemma", "mistral", "phi3")
            show_context: Whether to display retrieved context
        """
        self.model_name = model_name
        self.show_context = show_context
        
        # Load HuggingFace token
        self.hf_token = load_hf_token()
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required. Create hf.txt with your token.\n"
                "Get token from: https://huggingface.co/settings/tokens"
            )
        
        # Load players and teams from Neo4j
        print("Loading FPL data from Neo4j...")
        try:
            self.players, self.teams = load_known_players_and_teams()
            print(f"✓ Loaded {len(self.players)} players and {len(self.teams)} teams\n")
        except Exception as e:
            print(f"Error loading data from Neo4j: {e}")
            print("Make sure Neo4j is running and config.txt is set up.")
            raise
        
        print(f"Using model: {model_name.upper()}")
        print(f"Show context: {show_context}")
        print()
    
    def answer_question(self, question: str) -> dict:
        """
        Process question through full RAG pipeline.
        
        Args:
            question: User's question
        
        Returns:
            Dict with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Step 1: Intent classification
        intent = classify_intent(question)
        print(f"Intent: {intent}")
        
        # Step 2: Entity extraction
        entities = extract_entities(question, self.players, self.teams)
        print(f"Entities: {entities.to_dict()}")
        
        # Step 3: Hybrid retrieval
        print("\nRetrieving from Knowledge Graph...")
        query_embedding = get_query_embedding(question)
        hybrid_results = retrieve_hybrid(
            intent=intent,
            entities=entities,
            query_embedding=query_embedding,
            index_name="player_embedding_index_minilm",
            top_k=10
        )
        
        # Show retrieval stats
        summary = hybrid_results.get("summary", {})
        print(f"Retrieved: {summary.get('baseline_player_count', 0)} baseline players, "
              f"{summary.get('embedding_player_count', 0)} embedding players")
        
        # Step 4: Generate answer
        print(f"\nGenerating answer with {self.model_name.upper()}...")
        result = generate_fpl_answer(
            question=question,
            hybrid_results=hybrid_results,
            model_name=self.model_name,
            hf_token=self.hf_token
        )
        
        # Display context if requested
        if self.show_context:
            print(f"\n--- Retrieved Context ---")
            print(result.get("context", "No context"))
            print(f"--- End Context ---\n")
        
        return result
    
    def run_interactive(self):
        """Run interactive chatbot loop"""
        print("\n" + "="*60)
        print("FPL CHATBOT - Interactive Mode")
        print("="*60)
        print("\nAsk me anything about Fantasy Premier League!")
        print("Examples:")
        print("  - Who scored the most points in 2023-24?")
        print("  - Compare Salah and Haaland")
        print("  - Recommend a midfielder under 8 million")
        print("  - Which team has the best defense?")
        print("\nCommands:")
        print("  'exit' or 'quit' - Exit chatbot")
        print("  'model' - Change LLM model")
        print("  'context' - Toggle context display")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'model':
                    self._change_model()
                    continue
                
                if user_input.lower() == 'context':
                    self.show_context = not self.show_context
                    print(f"\nContext display: {'ON' if self.show_context else 'OFF'}\n")
                    continue
                
                # Process question
                result = self.answer_question(user_input)
                
                # Display answer
                print(f"\n{'='*60}")
                if result['success']:
                    print(f"Answer: {result['answer']}")
                    print(f"\n(Response time: {result['response_time']:.2f}s, "
                          f"Model: {result['model'].upper()})")
                else:
                    print(f"Error: {result['error']}")
                print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                import traceback
                traceback.print_exc()
    
    def _change_model(self):
        """Interactive model selection"""
        print("\nAvailable models:")
        for key, info in AVAILABLE_MODELS.items():
            current = " (CURRENT)" if info["name"] == self.model_name else ""
            print(f"  {key}. {info['display']}{current}")
        
        choice = input("Select model (1-3): ").strip()
        if choice in AVAILABLE_MODELS:
            self.model_name = AVAILABLE_MODELS[choice]["name"]
            print(f"\nSwitched to: {AVAILABLE_MODELS[choice]['display']}\n")
        else:
            print("\nInvalid choice. Model unchanged.\n")


# =====================  MAIN  =====================

def main():
    """Main entry point"""
    print("="*60)
    print("FPL CHATBOT - RAG System with Multi-Model LLM")
    print("="*60)
    
    # Model selection
    print("\nSelect LLM model:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  {key}. {info['display']}")
    
    model_choice = input(f"\nEnter choice (1-3, default=1): ").strip()
    if model_choice not in AVAILABLE_MODELS:
        model_choice = "1"
    
    model_name = AVAILABLE_MODELS[model_choice]["name"]
    
    # Context display option
    show_context_input = input("Show retrieved context? (y/n, default=n): ").strip().lower()
    show_context = show_context_input == 'y'
    
    # Initialize and run chatbot
    try:
        chatbot = FPLChatbot(model_name=model_name, show_context=show_context)
        chatbot.run_interactive()
    except Exception as e:
        print(f"\nFailed to initialize chatbot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
