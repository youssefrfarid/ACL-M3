"""
LLM Model Evaluation Script

Evaluates and compares 3 LLM models on FPL question answering:
- Gemma 2B (fast, lightweight)
- Mistral 7B (high quality)
- Phi-3 Mini (balanced)

Metrics:
- Quantitative: Response time, token count, success rate
- Qualitative: Accuracy, relevance, naturalness, completeness (manual scoring)

Usage:
    python evaluate_llms.py
"""

import json
import time
import csv
from typing import List, Dict, Any
from datetime import datetime

from input_processing import (
    classify_intent,
    extract_entities,
    load_known_players_and_teams,
    get_query_embedding,
    load_hf_token
)
from graph_retrieval import retrieve_hybrid
from llm_layer import (
    generate_fpl_answer, 
    generate_openrouter_answer,
    load_openrouter_key,
    FPLLLMInterface,
    OpenRouterLLMInterface
)


# =====================  CONFIGURATION  =====================

# HuggingFace models
HF_MODELS = ["gemma", "mistral", "phi3"]

# OpenRouter models (free tier)
OPENROUTER_MODELS = ["qwen3-coder", "llama-3.3-70b", "gemini-flash"]

# Default: Test OpenRouter models
MODELS_TO_TEST = OPENROUTER_MODELS

TEST_QUESTIONS_FILE = "test_questions.json"
RESULTS_FILE = "evaluation_results.csv"
REPORT_FILE = "evaluation_report.md"


# =====================  EVALUATION FRAMEWORK  =====================

class LLMEvaluator:
    """Evaluate and compare multiple LLM models"""
    
    def __init__(self, use_openrouter: bool = True):
        """Initialize evaluator
        
        Args:
            use_openrouter: If True, use OpenRouter models; else use HuggingFace
        """
        self.use_openrouter = use_openrouter
        
        # Load appropriate credentials
        if use_openrouter:
            self.api_key = load_openrouter_key()
            if not self.api_key:
                raise ValueError("OpenRouter API key required. Create openrouter_config.txt")
            self.hf_token = None
        else:
            self.hf_token = load_hf_token()
            if not self.hf_token:
                raise ValueError("HuggingFace token required. Create hf.txt")
            self.api_key = None
        
        # Load players and teams
        print("Loading FPL data from Neo4j...")
        self.players, self.teams = load_known_players_and_teams()
        print(f"✓ Loaded {len(self.players)} players and {len(self.teams)} teams\n")
        
        # Load test questions
        with open(TEST_QUESTIONS_FILE, 'r') as f:
            self.test_questions = json.load(f)
        print(f"✓ Loaded {len(self.test_questions)} test questions\n")
        
        self.results = []
    
    def evaluate_question(
        self, 
        question_data: Dict[str, Any], 
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single question with a single model.
        
        Args:
            question_data: Question dict from test_questions.json
            model_name: Model to use
        
        Returns:
            Result dict with answer and metrics
        """
        question = question_data["question"]
        
        print(f"  Q{question_data['id']}: {question[:50]}...")
        
        try:
            # Step 1: Intent & entities
            intent = classify_intent(question)
            entities = extract_entities(question, self.players, self.teams)
            
            # Step 2: Retrieval
            query_embedding = get_query_embedding(question)
            hybrid_results = retrieve_hybrid(
                intent=intent,
                entities=entities,
                query_embedding=query_embedding,
                index_name="player_embedding_index_minilm",
                top_k=10
            )
            
            # Step 3: Generate answer (use appropriate provider)
            start_time = time.time()
            
            if self.use_openrouter:
                result = generate_openrouter_answer(
                    question=question,
                    hybrid_results=hybrid_results,
                    model_name=model_name,
                    api_key=self.api_key
                )
            else:
                result = generate_fpl_answer(
                    question=question,
                    hybrid_results=hybrid_results,
                    model_name=model_name,
                    hf_token=self.hf_token
                )
            
            total_time = time.time() - start_time
            
            # Compile results
            return {
                "question_id": question_data["id"],
                "question": question,
                "intent": intent,
                "model": model_name,
                "provider": "openrouter" if self.use_openrouter else "huggingface",
                "answer": result["answer"],
                "success": result["success"],
                "response_time": result["response_time"],
                "total_time": total_time,
                "token_count": result["token_count"],
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "answer_length": len(result["answer"].split()),
                "error": result.get("error"),
                "context_length": len(result.get("context", "")),
            }
        
        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                "question_id": question_data["id"],
                "question": question,
                "model": model_name,
                "provider": "openrouter" if self.use_openrouter else "huggingface",
                "success": False,
                "error": str(e),
                "answer": "",
                "response_time": 0,
                "total_time": 0,
                "token_count": 0,
                "answer_length": 0,
            }
    
    def run_evaluation(self):
        """Run full evaluation on all models and questions"""
        print("="*60)
        print("STARTING LLM EVALUATION")
        print("="*60)
        print(f"Models: {', '.join([m.upper() for m in MODELS_TO_TEST])}")
        print(f"Questions: {len(self.test_questions)}")
        print(f"Total evaluations: {len(MODELS_TO_TEST) * len(self.test_questions)}")
        print("="*60 + "\n")
        
        for model_name in MODELS_TO_TEST:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {model_name.upper()}")
            print(f"{'='*60}\n")
            
            for question_data in self.test_questions:
                result = self.evaluate_question(question_data, model_name)
                self.results.append(result)
                
                # Show brief result
                if result["success"]:
                    print(f"    ✓ {result['response_time']:.2f}s, "
                          f"{result['answer_length']} words")
                else:
                    print(f"    ✗ Failed: {result['error']}")
            
            print(f"\n✓ Completed {model_name.upper()}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}\n")
    
    def save_results(self):
        """Save results to CSV"""
        print(f"Saving results to {RESULTS_FILE}...")
        
        with open(RESULTS_FILE, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"✓ Saved {len(self.results)} results\n")
    
    def generate_quantitative_report(self) -> str:
        """Generate quantitative comparison report"""
        # Group results by model
        model_stats = {}
        
        for model in MODELS_TO_TEST:
            model_results = [r for r in self.results if r["model"] == model and r["success"]]
            
            if model_results:
                model_stats[model] = {
                    "count": len(model_results),
                    "success_rate": len(model_results) / len([r for r in self.results if r["model"] == model]),
                    "avg_response_time": sum(r["response_time"] for r in model_results) / len(model_results),
                    "avg_total_time": sum(r["total_time"] for r in model_results) / len(model_results),
                    "avg_tokens": sum(r["token_count"] for r in model_results) / len(model_results),
                    "avg_answer_length": sum(r["answer_length"] for r in model_results) / len(model_results),
                }
        
        # Build report
        report = []
        report.append("## Quantitative Metrics\n")
        report.append("| Model | Success Rate | Avg Response Time | Avg Answer Length | Avg Tokens |")
        report.append("|-------|--------------|-------------------|-------------------|------------|")
        
        for model in MODELS_TO_TEST:
            if model in model_stats:
                stats = model_stats[model]
                report.append(
                    f"| {model.upper()} | "
                    f"{stats['success_rate']*100:.1f}% | "
                    f"{stats['avg_response_time']:.2f}s | "
                    f"{stats['avg_answer_length']:.0f} words | "
                    f"{stats['avg_tokens']:.0f} |"
                )
        
        report.append("")
        return "\n".join(report)
    
    def generate_report(self):
        """Generate markdown evaluation report"""
        print(f"Generating report {REPORT_FILE}...")
        
        report = []
        report.append(f"# FPL LLM Model Evaluation Report\n")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Models Tested**: {', '.join([m.upper() for m in MODELS_TO_TEST])}")
        report.append(f"**Test Questions**: {len(self.test_questions)}")
        report.append(f"**Total Evaluations**: {len(self.results)}\n")
        
        # Quantitative metrics
        report.append(self.generate_quantitative_report())
        
        # Sample answers
        report.append("## Sample Answers\n")
        report.append("**Question**: Who scored the most points in the 2023-24 season?\n")
        
        q1_results = [r for r in self.results if r["question_id"] == 1]
        for result in q1_results:
            if result["success"]:
                report.append(f"### {result['model'].upper()}")
                report.append(f"```\n{result['answer']}\n```\n")
        
        # Qualitative evaluation template
        report.append("## Qualitative Evaluation (Manual Scoring)\n")
        report.append("*Instructions: Score each model 1-5 for each question on the following criteria:*\n")
        report.append("- **Accuracy**: Is the answer factually correct based on KG data?")
        report.append("- **Relevance**: Does it address the question?")
        report.append("- **Naturalness**: Is the language fluent and human-like?")
        report.append("- **Completeness**: Does it use the context well?\n")
        
        report.append("| Question | Model | Accuracy | Relevance | Naturalness | Completeness | Notes |")
        report.append("|----------|-------|----------|-----------|-------------|--------------|-------|")
        
        for q in self.test_questions[:5]:  # Show first 5 for manual scoring
            for model in MODELS_TO_TEST:
                report.append(f"| Q{q['id']} | {model.upper()} | | | | | |")
        
        report.append("\n*Add more rows as needed for all questions*\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        report.append("*Fill in after completing qualitative evaluation:*\n")
        report.append("- **Best for Speed**: ")
        report.append("- **Best for Quality**: ")
        report.append("- **Best Overall**: ")
        report.append("- **Reasoning**: \n")
        
        # Write report
        with open(REPORT_FILE, 'w') as f:
            f.write("\n".join(report))
        
        print(f"✓ Report saved to {REPORT_FILE}\n")


# =====================  MAIN  =====================

def main():
    """Main evaluation entry point"""
    print("\n" + "="*60)
    print("FPL LLM MODEL EVALUATION")
    print("="*60)
    print("\nThis script will:")
    print("1. Run all test questions through 3 LLM models")
    print("2. Collect quantitative metrics (time, tokens, etc.)")
    print("3. Generate evaluation report")
    print("4. Create template for manual qualitative scoring")
    print("\nNote: This may take 10-20 minutes depending on API speed.")
    print("="*60 + "\n")
    
    proceed = input("Proceed with evaluation? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Evaluation cancelled.")
        return
    
    try:
        evaluator = LLMEvaluator()
        evaluator.run_evaluation()
        evaluator.save_results()
        evaluator.generate_report()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"\nResults saved to:")
        print(f"  - {RESULTS_FILE} (CSV data)")
        print(f"  - {REPORT_FILE} (Markdown report)")
        print("\nNext steps:")
        print("1. Review the generated report")
        print("2. Manually score qualitative metrics (accuracy, relevance, etc.)")
        print("3. Add your recommendations to the report")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
