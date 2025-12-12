Build a UI (e.g., Streamlit)
Create a user interface that demonstrates your Graph-RAG system in action.
Streamlit is recommended for its simplicity. The UI should be functional and
intuitive, but doesn't need to be complex. Focus on demonstrating the system's
capabilities.
Select one task or more (QA, booking assistant, recommender, etc.). The UI
should allow a user to:
a. View the KG-retrieved context
Display the raw information retrieved from the knowledge graph before it's
processed by the LLM. Show users what nodes, relationships, and data were
found. This increases transparency and trust.
b. View the final LLM answer
Display the LLM's final response to the user's query. The main output is the
answer generated using the KG context.
c. Optionally display:
-
Cypher Queries Executed:
Show the actual Cypher queries that were run to retrieve
information. This helps users understand how the system found the
information. Useful for debugging and transparency.
-
Graph Visualization Snippets:
Visualize the retrieved subgraph (nodes and relationships) in a
graph format.
Use libraries like NetworkX, Plotly, or Neo4j's built-in visualization to
show the graph structure.
-
Recommendations:
-
-
-
-
-
If implementing a recommender, display ranked recommendations
with explanations.
Show why certain entities were recommended based on user
preferences and KG data:
-
Hotel: hotel recommendations with explanations

-
-
-
-
-
FPL: player recommendations for fantasy teams
Model Selection Dropdown (to switch between LLMs):
Allow users to select which LLM to use for generating answers.
Enables real-time comparison of different models' responses to the
same query.
Retrieval Method Selection (baseline, embeddings, or both): Allow users
to choose which retrieval method(s) to use. Enables comparison of
baseline Cypher queries vs. embedding-based retrieval vs. hybrid
approaches.