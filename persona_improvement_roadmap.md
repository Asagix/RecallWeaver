# Persona Improvement Roadmap

This document tracks the implementation status of recommendations made by the expert team (Dr. Reed, Prof. Tanaka, Dr. Sharma) to enhance the AI's human-like qualities and persona integration.

## Drive System Enhancements

*   [x] **LLM Analysis Granularity:** Modify `drive_analysis_prompt.txt` to output numerical scores (-1.0 to 1.0) instead of categories.
*   [x] **Event-Driven Updates:** Trigger short-term drive updates based on high emotional impact events (`emotional_impact_threshold`), in addition to interaction counts.
*   [x] **Drive Analysis Context:** Provide the previous drive state to the `drive_analysis_prompt.txt` for better assessment of change.
*   [x] **Long-Term Analysis Focus:** Update `long_term_drive_analysis_prompt.txt` to explicitly focus on identifying drive-related patterns within the ASM summary.
*   [x] **Inter-Drive Dynamics:** Model inhibitory or excitatory relationships *between* different drives during state updates (e.g., high Safety suppressing Novelty). (Basic implementation added)
*   [x] **Heuristic Refinement:** Add more sophisticated heuristic adjustments based on conversation patterns (e.g., user corrections affecting 'Control', collaboration affecting 'Connection'). (Basic keyword implementation added)
*   [x] **Baseline Dynamics:** Implement mechanisms for significant events (highly salient/emotional memories) to cause more abrupt shifts in the long-term drive state or baseline. (Basic implementation added)

## Mood & Emotional Resonance

*   [x] **Mood Influence on Generation:** Pass the calculated mood (Valence/Arousal) to the main chat prompt (`_construct_prompt`) and instruct the LLM to let it subtly influence response tone.
*   [x] **Expose Drive State Summary:** Include a qualitative summary of the current drive state (e.g., "Feeling: Secure, Curious") in the main chat prompt context.

## ASM (Autobiographical Self-Model)

*   [x] **Refined Structure:** Update `asm_generation_prompt.txt` to use the new structure: `core_traits`, `recurring_themes`, `goals_motivations`, `relational_stance`, `emotional_profile`, `summary_statement`.
*   [x] **ASM Usage in Planning:** Integrate ASM fields (`goals_motivations`, `relational_stance`) into the `workspace_planning_prompt.txt` to influence planning style and priorities.
*   [x] **ASM Usage in Generation:** Add system note in `_construct_prompt` instructing the LLM to consider the ASM when formulating responses.
*   [x] **Dynamic ASM Integration:** Implement logic for retrieved memories that strongly contradict the ASM to trigger a mini-consolidation or flag the ASM for review/update. (Basic check and flag added)

## Memory Retrieval & Graph Dynamics

*   [x] **Contextual Edge Weighting:** Implement basic dynamic edge weighting in `retrieve_memory_chain` where drive states (e.g., low 'Understanding') temporarily boost relevant edge types (e.g., `HIERARCHICAL`, `CAUSES`).
*   [x] **Memory Strength Budgeting:** Implement logic in `_construct_prompt` to allocate prompt context budget based on memory strength (stronger memories get more detail/tokens).
*   [x] **Second-Order Inference Refinement:** Enhance `_infer_second_order_relations` to infer *typed* relationships (e.g., A *enables* C via B) instead of just generic `INFERRED_RELATED_TO`. (Implemented using LLM)

## Prompting & Interaction Flow

*   [x] **Intention Handling:** Modify `_construct_prompt` to instruct the LLM to review retrieved intention nodes and incorporate them if the trigger condition seems met. Modify `process_interaction` to retrieve active intention nodes.
*   [ ] **Prompt Chaining for Reflection:** Implement meta-cognitive loops by chaining prompts (e.g., generate ASM, then feed ASM into a reflection prompt). (Placeholder added, full implementation deferred)

## Configuration & Tuning

*   [x] **Initial Tuning:** Adjustments made to `config.yaml` for drive factors, mood influence, activation parameters, etc., to support the implemented features. (Note: This is an ongoing process).
