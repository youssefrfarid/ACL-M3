1. Input Preprocessing
   a. Intent Classification
   Classify what the user wants to do (e.g., ask a question, get recommendations, search
   for entities). This helps route the query to the appropriate retrieval strategy. You can use
   rule-based methods (keyword matching) or LLM-based classification. The intent
   determines which Cypher queries or retrieval methods to use. Each theme should have
   its own intent classifier adapted to its domain (e.g., hotel search, player performance
   analysis, flight route queries).
   b. Entity Extractions
   Extract relevant entities from user input (e.g., entity names, locations, dates, attributes).
   These entities are used to fill in the chosen Cypher query with the parameters.
   Use Named Entity Recognition (NER) to identify theme-specific entities:
   ➢ FPL theme: players, teams, positions, seasons, gameweeks, statistics
   c. Input Embedding (depending on 2.b)
   Convert the user's text input into a vector representation for semantic similarity
   search in the embedding-based retrieval approach. Only needed when you
   implement embedding-based retrieval (section 2.b). Use the same embedding
   model that was used to create node or feature vector embeddings in your KG.

2. Graph retrieval layer
   You are required to implement TWO experiments. The first one is the baseline only,
   and the second adds the embeddings (section 2.b) to the baseline.
   a. Baseline
3. Use Cypher queries to retrieve relevant information.
   Write structured queries in Cypher (Neo4j's query language) to fetch nodes,
   relationships, and properties from the knowledge graph based on extracted
   entities. These are deterministic queries that use exact matches or filters.
   Examples:

- FPL: "Get top players by position in season 2023"

2. At least 10 queries that answer 10 questions, based on the user input.
   Create a library of at least 10 different Cypher query templates that can handle
   various question types. These queries should cover different intents and entity
   combinations. You'll select and parameterize the appropriate query based on
   intent classification and entity extraction.
   FPL: player performance, team analysis, fixture queries,
   recommendations, statistics
3. Pass the extracted entities from the input to query the KG and retrieve the
   answer.
   Use the entities extracted in (step 1.b) as parameters to fill in the Cypher query
   templates, then execute them to get relevant graph data.
   Examples:FPL: If the user asks "Top forwards in 2023", extract position="FWD" and
   season="2023", then query players by position and season

b. Embeddings:
Implement semantic similarity search using vector embeddings. Choose ONE of the
following approaches, and experiment with at least TWO different embedding models
for comparison:

1. Node Embeddings
   Create vector representations for each node in the graph. Similar nodes will have
   similar embeddings.

- For themes with numerical data (FPL, Airline): Use numerical feature
  vectors derived from node properties (e.g., player stats, journey metrics)
- For themes with textual data (Hotel): Use graph embedding techniques
  (e.g., Node2Vec, GraphSAGE) or text embeddings of node properties
- Store these in Neo4j's vector index for fast similarity search

2. Features Vector Embeddings
   Create vector representations for feature combinations. This captures the
   semantic meaning of feature sets.

- For themes without textual features (FPL, Airline): You may construct
  text descriptions from numerical properties, then embed them (e.g.,
  "Player: X, Position: Y, Points: Z" or "Journey: X, Class: Y, Food: Z, Delay:
  W"). Or use the numerical features as they are directly
