# Persistent Graph Memory System (MemSys)

## Overview

MemSys is a Python application designed to provide a persistent, long-term memory for conversational AI systems. Instead of relying solely on a limited context window, it stores conversational turns and derived knowledge (summaries, concepts, relationships) in a graph structure. This graph is persisted to disk along with vector embeddings, allowing for semantic retrieval of relevant past information to enrich the AI's responses.

The system uses a combination of:

* **NetworkX:** For graph representation and manipulation.
* **FAISS:** For efficient similarity search over text embeddings.
* **Sentence Transformers:** For generating text embeddings.
* **Hugging Face Transformers:** For tokenization (required for accurate prompt construction).
* **KoboldCpp (or compatible API):** As the backend Large Language Model (LLM) for generating responses, summarizing text, and analyzing requests.
* **PyQt6:** For the graphical user interface.

## Current Features

* **Graph-Based Memory:** Stores conversation turns, summaries, and concepts as nodes in a directed graph (`NetworkX`). Edges represent relationships like temporal sequence, summary-of, mentions-concept, associative, and hierarchical links.
* **Persistence:** Saves the memory graph, text embeddings (`numpy`), and FAISS index to disk, allowing memory to persist across sessions. Organizes memory by "personality".
* **Multi-Personality Support:** Memory is stored in separate directories per personality, selectable via the GUI menu.
* **Semantic Retrieval:**
    * Uses FAISS index for fast similarity search to find relevant starting points in memory based on user input.
    * Employs an activation spreading algorithm (`retrieve_memory_chain`) traversing the graph based on edge types, decay parameters, and **node saliency** to retrieve a chain of relevant memories.
* **Node Saliency (V1):**
    * Nodes are assigned an initial saliency score based on type and configuration.
    * Saliency influences activation spreading during retrieval.
    * Node access count is tracked.
* **LLM Integration:**
    * Interfaces with KoboldCpp-compatible APIs (Generate and Chat Completions).
    * Constructs prompts dynamically, injecting relevant retrieved memories and conversation history within the LLM's token limit.
    * Supports multimodal input (text + image) via the Chat Completions API.
* **Memory Consolidation:**
    * Periodically (manually triggered or after a configurable number of turns), processes recent turns to:
        * Generate summaries.
        * Extract key concepts (with deduplication against existing concepts).
        * Identify basic associative and hierarchical relationships between concepts.
        * Prune original turns that have been summarized (optional).
* **Memory Modification:** Allows users to:
    * `delete` specific memory nodes (by UUID).
    * `edit` the text of memory nodes (by UUID).
    * `forget` memories related to a specific topic (via similarity search).
* **Memory Strength & Gradual Forgetting:**
    * Nodes have a `memory_strength` attribute (0.0-1.0), initialized at 1.0.
    * Periodically (based on interaction count), the system identifies nodes eligible for potential strength reduction (based on age, activation level).
    * A "forgettability score" is calculated based on various factors (recency, activation, saliency, emotion, connectivity, type).
    * The `memory_strength` of candidate nodes is reduced based on their forgettability score and a configured decay rate.
    * Node strength influences activation spreading during retrieval (weaker nodes contribute less).
    * Nodes with strength below a configurable threshold and older than a certain age can be permanently purged.
* **Basic File/Calendar Actions (Experimental):**
    * Includes backend logic (`file_manager.py`) and analysis prompts (`analyze_action_request`) to potentially handle requests for creating/appending files or adding/reading calendar events within a personality-specific workspace. (GUI integration is incomplete).
* **Graphical User Interface (PyQt6):**
    * Chat-like interface displaying conversation bubbles.
    * Displays retrieved memory context in a collapsible section.
    * Allows attaching images (via button, paste, or drag-and-drop).
    * Provides buttons for sending messages, resetting memory, and triggering consolidation.
    * Personality selection menu.
    * Status bar indicating backend readiness.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-directory>
    ```
2.  **Create a Python environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `requirements.txt` is not provided, install manually:*
    ```bash
    pip install networkx numpy faiss-cpu requests pyyaml sentence-transformers transformers torch PyQt6 tzdata
    ```
    *(Ensure you have the appropriate version of `faiss-cpu` or `faiss-gpu` depending on your hardware and setup.)*
    *(`torch` is usually a dependency for `sentence-transformers` and `transformers`.)*

4.  **LLM Backend:** Ensure you have a running KoboldCpp instance (or a compatible OpenAI-style API endpoint) accessible.

## Configuration

Modify the `config.yaml` file to set up:

* `base_memory_path`: The root directory where personality memory folders will be stored.
* `default_personality`: The personality to load by default (optional).
* `embedding_model`, `tokenizer_name`: Models used for embedding and tokenization.
* `kobold_api_url`, `kobold_chat_api_url`: URLs for your LLM backend.
* Activation spreading parameters (`activation` section).
* Prompting budget parameters (`prompting` section).
* Consolidation parameters (`consolidation` section).
* GUI style parameters (`gui_style` section).

## Running the Application

Execute the GUI script:

```bash
python gui_chat.py
