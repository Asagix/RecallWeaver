# Implementation Plan: Persistent Graph-Based Memory System (v3)

This document tracks the implementation progress and outlines future enhancements, including detailed steps for Phase 1 features.

## Phase 1: Backend Core (`persistent_backend_graph.py`)

-   [x] **Basic Setup & Persistence:**
    -   [x] Create `persistent_backend_graph.py`.
    -   [x] Define `GraphMemoryClient` class structure.
    -   [x] Implement `__init__` method (initialize graph, embedder, paths).
    -   [x] Implement `_load_memory` (load graph, embeddings, index, mapping from files).
    -   [x] Implement `_save_memory` (save graph, embeddings, index, mapping to files).
    -   [x] Implement `_rebuild_index_from_graph_embeddings` for consistency checks.
    -   [x] Add basic logging.
    -   [x] Add `if __name__ == "__main__":` test block for initialization.
-   [x] **Memory Addition:**
    -   [x] Implement `_get_embedding` helper.
    -   [x] Implement `add_memory_node` (UUID, timestamp, embedding generation, add to graph, add to FAISS, update mappings).
    -   [x] Add temporal edge creation in `add_memory_node`.
-   [x] **Data Model Enhancements (Phase 1 - Strength Based):**
    -   [x] **Modify `add_memory_node`:** Initialize new node attributes: `memory_strength` (1.0), `access_count` (0), default `emotion_valence`/`arousal` (from config), initial `saliency_score` (calculated based on V1 formula - see Saliency section). Remove `status`.
    -   [x] **Verify Save/Load:** Test `_save_memory` and `_load_memory` to ensure new attributes are persisted correctly.
    -   [x] **Update `config.yaml`:** Add `features` section with enable flags. Add `emotion_analysis` section with `default_valence`/`arousal`. Add `memory_strength` section. Remove `status`-related forgetting params.
-   [x] **V1 Saliency Implementation (Phase 1):**
    -   [x] **Enable Flag:** Check `features.enable_saliency` in relevant code sections.
    * [x] **Modify `add_memory_node`:** Implement V1 initial `saliency_score` calculation (based on node type base score + default emotion arousal influence, clamped 0-1) as detailed in Data Model changes.
    * [x] **Modify `retrieve_memory_chain` (Access Count):** Increment `access_count` attribute for nodes in the final `relevant_nodes` list. Ensure `last_accessed_ts` is also updated.
    * [x] **Modify `retrieve_memory_chain` (Activation Influence):** Factor `saliency_score` into the `act_pass` calculation (e.g., `act_pass *= (1.0 + source_saliency * activation_influence)`) using configured `activation_influence` factor.
    * [x] **Modify `run_consolidation`:** Ensure new summary/concept nodes created via `add_memory_node` receive their initial saliency score automatically.
    * [x] **Add Placeholder Method:** Add stub `update_node_saliency(self, node_uuid, change_factor)` method to `GraphMemoryClient` for V2.
    * [x] **Update `config.yaml`:** Define `saliency` section with `initial_scores`, `emotion_influence_factor`, `activation_influence`. (Config already updated)
    * [ ] **Testing (Saliency):** Unit test calculation logic. Integration test attribute persistence, `access_count` increment, activation influence, and consolidation assignment.
-   [x] **LLM Interaction:**
    -   [x] Implement `_call_kobold_api` (Generate API).
    -   [x] Implement `_call_kobold_multimodal_api` (Chat Completions API).
    -   [x] Include basic error handling for API calls.
-   [x] **Basic Interaction Flow & Retrieval:**
    -   [x] Implement `_search_similar_nodes` using FAISS for initial node retrieval.
    -   [x] Implement `process_interaction` structure (call search, retrieval, construct prompt, call LLM, add turns to memory, handle image attachments).
    -   [x] Implement basic `_construct_prompt`.
-   [x] **Memory Retrieval (Activation Spreading):**
    -   [x] Define activation parameters (now in config).
    -   [x] Implement helper `_calculate_node_decay`.
    -   [x] Implement helper `_calculate_dynamic_edge_strength`.
    -   [x] Implement `retrieve_memory_chain` core logic (Initialization, Spreading Loop, Decay, Thresholding, Edge Factors).
    -   [x] Update `process_interaction` to use the results of `retrieve_memory_chain`.
-   [x] **Context Injection Refinement:**
    -   [x] Integrate `transformers.AutoTokenizer` for accurate token counting.
    -   [x] Update `_construct_prompt` (Budgeting, Formatting, Chronological Injection, Gemma Format, Time Info, Recall Instruction).
-   [x] **Memory Manipulation:**
    -   [x] Implement `analyze_memory_modification_request`.
    -   [x] Implement `delete_memory_entry`.
    -   [x] Implement `edit_memory_entry`.
    -   [x] Implement `forget_topic`.
    -   [x] Trigger `_save_memory` consistently after modifications.
-   [ ] **Consolidation (V1 Enhancements):**
    -   [x] Implement `run_consolidation` basic structure.
    -   [x] Define initial LLM prompts.
    -   [x] Add logic to select nodes.
    -   [x] Add logic to parse LLM response and add nodes/edges.
    -   [x] Implement concept deduplication.
    -   [x] Implement pruning of summarized 'turn' nodes.
    -   [x] **Refine LLM prompts** for summarization, concept extraction, and relation extraction.
    -   [x] **Implement Hybrid Association Extraction (V1.1):**
        -   [x] Integrate NLP library (e.g., spaCy) to extract basic entities/dependency relations first. (spaCy loaded, basic structure in place)
        -   [x] Store basic spaCy dependency relations as `SPACY_REL` edges.
        -   [x] *Optionally* (if `features.enable_rich_associations` is true) call LLM with text + extracted info, asking for specific *additional* typed relations (e.g., `CAUSES`) from a core set (`IS_A`, `PART_OF`, `CAUSES`, `HAS_PROPERTY`, `RELATED_TO`). (LLM call implemented)
        -   [x] Implement **robust parsing** for LLM's structured (JSON) output for relationships. (Basic JSON list parsing implemented)
    -   [x] **Implement V1 Emotion Analysis:**
        -   [x] Integrate local sentiment/emotion library analysis during `run_consolidation`. (Using text2emotion)
        -   [x] Call `_analyze_and_update_emotion` helper to store results on relevant nodes.
    -   [x] Implement mechanism for automatic/periodic consolidation trigger.
-   [x] **Memory Strength & Forgetting (V1 Implementation):**
    -   [x] **Enable Flag:** Check `features.enable_forgetting`.
    * [x] **Implement `run_memory_maintenance()` Method:**
        -   [x] Add trigger mechanism (interaction count based) in `Worker`.
        -   [x] Implement candidate node filtering (age, activation).
        -   [x] Implement normalization functions for score factors.
        -   [x] Implement forgettability score calculation (weighted sum formula).
        -   [x] Implement **strength reduction** (`memory_strength *= (1 - forget_score * decay_rate)`).
        -   [x] Add clear logging for strength reduction.
    * [x] **Modify Retrieval:** Update `retrieve_memory_chain` and `_search_similar_nodes` to remove `status` filtering and incorporate `memory_strength` into activation.
    * [x] **Update `config.yaml`:** Define `forgetting` section (weights, candidates). Define `memory_strength` section (initial, decay_rate, purge_threshold, purge_age). Remove status-based params.
    * [ ] **Testing (Forgetting/Strength):** Test candidate filtering, score calculation, strength reduction logic, and retrieval modulation. Verify tuning knobs work.
    * [x] **Implement `purge_weak_nodes()`:** Permanently delete nodes below strength threshold and above age threshold.
-   [x] **Action/Tool Handling (via Focused Intent Analysis):**
    -   [x] Backend: Basic structure for `analyze_action_request` and `execute_action` exists.
    -   [x] Backend: Basic file/calendar wrapper methods exist.
    -   [x] Backend: Refine LLM prompt for `analyze_action_request`.
    -   [x] Backend: Implement robust JSON parsing & argument validation for `analyze_action_request`.
    -   [x] Backend: Enhance error handling within `execute_action` and file/calendar wrappers.
    -   [x] GUI: Integrate action analysis call into `Worker.add_input`.
    -   [x] GUI: Implement logic in `Worker` to handle `clarify` responses (store state, process next input as clarification) and queue/execute specific action tasks.
    -   [x] GUI: Implement UI elements/flow for clarification requests (clearer prompt, persistent status bar message, placeholder text, visual indicator light).
    -   [ ] GUI: Implement UI flow for potential action confirmations (e.g., before overwriting a file).
-   [ ] **Context Dependency (Focus via Recent Concepts):**
    -   [x] Modify `process_interaction` to identify concept nodes mentioned in the last user/AI turn.
    -   [x] Modify `retrieve_memory_chain` to accept recent concept UUIDs.
    -   [x] Implement logic in `retrieve_memory_chain` to apply `context_focus_boost` to initial activation of relevant nodes.
    -   [x] Add `activation.context_focus_boost` parameter to `config.yaml`.
    -   [ ] Testing: Verify boost is applied correctly and influences retrieval as expected.
-   [ ] **Interference Simulation:**
    -   [x] Add `activation.interference` section to `config.yaml` (enable, thresholds, factor).
    -   [x] Modify `retrieve_memory_chain` to identify semantically similar, co-activated nodes.
    -   [x] Implement logic to apply `interference_penalty_factor` to non-dominant nodes within similar clusters.
    -   [ ] Testing: Verify interference is applied correctly and subtly reduces activation of competing similar memories.

## Phase 2: GUI (`gui_chat.py`)

-   [x] **Basic UI Structure & Features** (Input, Output, Buttons, Formatting, Styling, Status Bar, Menu, Image Handling).
-   [x] **Backend Worker Thread** (Setup, Signals, Task Handling, Personality Switching).
-   [x] **Connecting UI and Worker**.
-   [x] **Displaying Conversation & Memory Context**.
-   [x] **Handling Memory Modification Commands**.
-   [x] **Action/Tool Handling (GUI Layer):** (See Backend section for details)
    -   [x] Integrate action analysis call.
    -   [x] Add Worker logic for clarification/execution tasks (including handling user response to clarification).
    -   [x] Implement UI for clarification (Improved prompt/indicator implemented).
    -   [ ] Implement UI for potential action confirmations.
-   [ ] **(Future GUI Enhancements):**
    -   [ ] Add interface to view/manage 'archived' nodes.
    -   [ ] Add UI for user feedback on memory saliency.
    -   [ ] Add basic graph visualization capabilities.

## Phase 3: Configuration & Refinement

-   [x] **Configuration File (`config.yaml`)**.
-   [ ] **Parameter Tuning & Evaluation:**
    -   [ ] Develop systematic test scenarios.
    -   [ ] Systematically tune activation parameters.
    -   [ ] Systematically tune consolidation parameters (including association extraction).
    -   [ ] Systematically tune **forgetting parameters** (weights, thresholds, triggers, **recency_decay_constant**).
    -   [ ] Systematically tune **saliency parameters** (initial scores, influences, **recall_boost_factor**).
    -   [ ] Tune Faiss search parameter (`k`).
    -   [ ] Tune LLM generation parameters.
    -   [ ] Evaluate prompting strategies.
-   [ ] **Error Handling & Robustness:**
    -   [ ] Implement more robust parsing of LLM responses (consolidation, analysis).
    -   [ ] Enhance file I/O and FAISS error handling.
    -   [ ] Add more thorough checks for graph/index consistency.
-   [ ] **Testing & Documentation:**
    -   [x] Basic `if __name__ == "__main__":` tests exist.
    -   [x] Added basic docstrings.
    -   [x] Created initial README.
    -   [ ] Develop **unit tests** for new helpers (saliency calc, forgettability score, strength reduction logic, normalization, emotion analysis wrappers, parsers, saliency boost logic).
    -   [ ] Develop **integration tests** for Phase 1 features (saliency influence & recall boost, strength reduction cycle, retrieval modulation by strength, purging) and existing flows.
    -   [ ] Complete/Refine docstrings.
    -   [x] Update README as features evolve (Reflected strength change).
    -   [ ] Add clear documentation for all `config.yaml` parameters.
-   [ ] **Scalability Enhancements (Future):**
    -   [ ] Evaluate optimized FAISS index.
    -   [ ] Evaluate graph database alternatives.

