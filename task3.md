3. LLM Layer
a. Combine the KG results from both the baseline and the embeddings
Merge the results from Cypher queries (baseline) and embedding-based retrieval
into a unified context. This provides both structured and semantic information to
the LLM. Combine retrieved nodes, relationships, and data from both methods.
Remove duplicates and rank/prioritize results if needed.
b. Use structure prompt: context, persona, task
Structure your LLM prompt with three components:
- Context: The retrieved KG information (nodes, relationships, data)
- Persona: Define the assistant's role (e.g., "You are a helpful travel
assistant" for Hotel, "You are an FPL expert" for FPL, "You are a flight
information assistant" for Airline)
- Task: Clear instructions on what to do with the context (e.g., "Answer the
user's question using only the provided information")
This structured approach improves answer quality and reduces hallucinations by
explicitly grounding the LLM in the KG data.
c. You must compare at least three models (examples)
Test your system with at least three different LLMs to evaluate performance
differences. Examples include GPT-3.5, GPT-4, Claude, Gemini (However, their
APIs are paid, so be careful if you choose these models), or open-source models
like Llama, Mistral, Gemma. Compare their accuracy, response quality, and cost.
Use free models from HuggingFace or OpenRouter for prototype development.
d. The comparison must include qualitative and quantitative impressions
Evaluate models using both:
- Quantitative: Metrics like accuracy, response time, token usage, cost
- Qualitative: Human evaluation of answer quality, relevance, naturalness,
and correctness
Create test cases and measure how well each model answers questions using
the KG context. Document which model performs best for your use case.