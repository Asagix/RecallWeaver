# Plan: Query-Type Biased Retrieval

Implement biasing of the initial memory search based on a quick LLM classification of the user's query.

**Steps:**

1.  [x] Create Query Classification Prompt: Add `prompts/query_type_prompt.txt`.
2.  [x] Add Helper Method: Implement `_classify_query_type` in `GraphMemoryClient` to call the LLM with the new prompt.
3.  [x] Modify `process_interaction`: Call `_classify_query_type` before calling `_search_similar_nodes`.
4.  [x] Modify `_search_similar_nodes`: Accept `query_type` ('episodic', 'semantic', 'other') as an argument. Adjust the search logic:
    *   If 'episodic', prioritize searching 'turn' nodes.
    *   If 'semantic', prioritize searching 'summary' and 'concept' nodes.
    *   If 'other', maintain current behavior.
5.  [x] Update Plan: Mark steps as complete.

**Status: Plan Complete**
