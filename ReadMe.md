# Persistent Graph Memory System (MemSys)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) <!-- Assuming MIT, update if needed -->

## Overview

MemSys is a Python application designed to give conversational AI systems a **persistent, long-term memory**. Instead of being limited by the context window of a Large Language Model (LLM), MemSys stores conversational turns, extracted concepts, relationships, and even the AI's evolving self-perception in a **graph database**. This allows the AI to recall relevant past information semantically, leading to more coherent, context-aware, and personalized interactions.

Think of it as giving your AI a searchable diary and knowledge base that grows with every conversation.

**Core Technologies:**

*   **LLM Backend:** Designed primarily for [KoboldCpp](https://github.com/LostRuins/koboldcpp), but adaptable to other OpenAI-compatible APIs.
*   **Graph Database:** [NetworkX](https://networkx.org/) for flexible in-memory graph representation.
*   **Vector Search:** [FAISS](https://github.com/facebookresearch/faiss) for efficient semantic similarity search.
*   **Embeddings:** [Sentence Transformers](https://www.sbert.net/) for converting text into meaningful vector representations.
*   **NLP:** [spaCy](https://spacy.io/) for relationship extraction, [text2emotion](https://pypi.org/project/text2emotion/) for basic emotion analysis.
*   **GUI:** [PyQt6](https://riverbankcomputing.com/software/pyqt/) for the desktop chat interface.
*   **Configuration:** [PyYAML](https://pyyaml.org/) for easy setup customization.

## Features

MemSys boasts a rich set of features aimed at creating a more dynamic and persistent AI memory:

*   **🧠 Graph-Based Memory:** Stores conversation turns, summaries, concepts, intentions, and relationships as nodes. Edges represent temporal sequence, summarization links, concept mentions, causality, analogies, and more.
*   **💾 Persistence:** Saves the entire memory state (graph, embeddings, index, self-model, drives, last conversation) to disk, organized by AI "personality". Memory persists across application restarts.
*   **🎭 Multi-Personality Support:** Maintain separate memories and configurations for different AI personalities, easily switchable via the GUI menu.
*   **🔍 Semantic Retrieval:** Uses FAISS for fast vector similarity search combined with a graph-based activation spreading algorithm to retrieve relevant memory chains based on the current query and context.
*   **✨ Node Saliency & Feedback:** Nodes have a dynamic saliency score influencing retrieval. Users can provide feedback (👍/👎) on AI responses to adjust the saliency of related memories.
*   **📉 Memory Strength & Forgetting:** Implements a nuanced forgetting mechanism. Nodes gradually lose `memory_strength` based on factors like recency, activation, saliency, emotion, and connectivity. Weak nodes can eventually be purged.
*   **🔄 Memory Consolidation:** Periodically processes recent conversational turns to:
    *   Generate concise summaries.
    *   Extract key concepts using LLM and spaCy, deduplicating against existing ones.
    *   Identify and add various relationship edges (causal, analogy, hierarchical, etc.) between concepts using LLM and spaCy.
    *   Optionally prune original turns after summarization.
*   **👤 Autobiographical Self-Model (ASM):** The AI periodically reflects on key memories to generate and update a structured summary of its perceived traits, recurring themes, significant events, and core beliefs/values.
*   **💖 Subconscious Drives & Mood:** Simulates basic subconscious drives (e.g., Connection, Safety, Control) with short-term fluctuations and long-term learned tendencies. The current drive state influences the AI's mood (Valence/Arousal), which in turn biases memory retrieval.
*   **🗂️ Workspace Agent:** Allows the AI to interact with a dedicated workspace directory:
    *   Create, append, read, list, and delete text files.
    *   Consolidate multiple files into one using LLM synthesis.
    *   Add and read simple calendar events (stored in `calendar.jsonl`).
    *   Optionally perform actions silently without explicit chat confirmation.
*   **🖼️ Multimodal Input:** Supports attaching and processing images within the chat via KoboldCpp's multimodal capabilities.
*   **💬 Graphical User Interface (PyQt6):**
    *   Familiar chat interface with message bubbles.
    *   Collapsible sections to view retrieved memories and the AI's current drive state.
    *   Status bar showing backend readiness and the AI's current estimated mood.
    *   Buttons for sending messages, attaching files (via button, paste, or drag-and-drop), inserting emojis (via picker), resetting memory, and triggering manual consolidation.
    *   Personality selection menu.
*   **⏱️ Independent History & Re-Greeting:** Stores the absolute last few turns separately, ensuring accurate display of recent history upon reload. Greets the user appropriately if a significant time gap has occurred since the last interaction.
*   **🔧 Tuning Log:** Outputs detailed structured logs (`logs/tuning_log.jsonl`) for analyzing memory retrieval, consolidation, emotion/drive calculations, and other internal processes to aid in parameter tuning.

## Setup & Installation

Getting MemSys up and running involves a few steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/memsys.git # Replace with your repo URL
    cd memsys
    ```

2.  **Create a Python Environment (Recommended):**
    Using a virtual environment keeps dependencies organized.
    ```bash
    python -m venv .venv
    # Activate the environment:
    # Windows:
    .\.venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on FAISS:** The `requirements.txt` likely includes `faiss-cpu`. If you have an NVIDIA GPU and CUDA installed, you might get better performance by installing `faiss-gpu` instead. Consult the [FAISS documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for details.
    *   **Note on PyTorch:** `torch` is usually installed automatically as a dependency. If you encounter issues, you might need to install it manually following instructions on the [PyTorch website](https://pytorch.org/).

4.  **Download spaCy Model:**
    MemSys uses a spaCy model for advanced relationship extraction during consolidation. Download the recommended English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up LLM Backend (KoboldCpp):**
    MemSys needs a running LLM backend. KoboldCpp is recommended.
    *   **Download KoboldCpp:** Get the latest version from the [KoboldCpp GitHub releases](https://github.com/LostRuins/koboldcpp/releases).
    *   **Download an LLM Model:** You'll need a model file in GGUF format.
        *   **Recommendation:** A good starting point is an instruct-tuned model like **Gemma 27b Instruct** (`gemma-2-27b-it-gguf`). Larger models generally provide better results for analysis tasks. Find models on Hugging Face.
    *   **Download Multimodal Projector (Optional but Recommended):** For image support, download the corresponding `.mmproj` file for your chosen model (e.g., the one for Gemma 2).
    *   **Run KoboldCpp with API and Multimodal Support:** Launch KoboldCpp from your terminal, enabling the API and specifying the multimodal projector:
        ```bash
        # Example command (adjust paths and parameters as needed):
        koboldcpp.exe --model /path/to/your/gemma-2-27b-it.Q4_K_M.gguf --useapi --port 5001 --mmproj /path/to/your/mmproj-gemma-2-27b-it-f16.gguf --threads 8 --contextsize 8192 --usemirostat 2 5.0 0.1
        ```
        *   `--model`: Path to your downloaded GGUF model file.
        *   `--useapi`: **Crucial!** Enables the necessary API endpoints.
        *   `--port 5001`: The port KoboldCpp will listen on (matches default in `config.yaml`).
        *   `--mmproj`: **Required for image support.** Path to the downloaded `.mmproj` file.
        *   Adjust `--threads`, `--contextsize`, GPU layers (`--gpulayers`), etc., based on your hardware. Refer to KoboldCpp documentation.
    *   Keep the KoboldCpp terminal window running while you use MemSys.

## Configuration

Before running, review and potentially modify `config.yaml`:

*   **`base_memory_path`**: Where personality data folders will be stored.
*   **`kobold_api_url`**, **`kobold_chat_api_url`**: Ensure these match the URL and port of your running KoboldCpp instance (usually `http://localhost:5001/api/v1/generate` and `http://localhost:5001/v1/chat/completions`).
*   **`embedding_model`**, **`tokenizer_name`**: Models used for embeddings and token counting. Defaults are generally good (`all-MiniLM-L6-v2`, `google/gemma-7b-it`).
*   **`llm_models`**: Configure parameters (max length, temperature, etc.) for different LLM tasks (chat, summary, analysis).
*   **`features`**: Enable or disable specific features like saliency, emotion analysis, forgetting, etc.
*   **`activation`**, **`consolidation`**, **`forgetting`**, **`memory_strength`**, **`subconscious_drives`**: Fine-tune parameters controlling memory retrieval, consolidation, forgetting, and the drive system.
*   **`gui_style`**: Customize the appearance of the chat bubbles and UI elements.

## Running the Application

Once dependencies are installed and KoboldCpp is running with the API enabled:

1.  Activate your Python virtual environment (if you created one).
2.  Run the GUI script from the repository's root directory:
    ```bash
    python gui_chat.py
    ```

## Usage Guide

1.  **Select Personality:** Upon launching, choose a personality from the "Personality" menu. You can also create new ones. This loads the AI's specific memory set.
2.  **Chat:** Type your message in the input field and press Enter or click "Send".
3.  **Attach Images:**
    *   Click the `+` button to browse for an image file.
    *   Paste an image directly from your clipboard into the input field (Ctrl+V).
    *   Drag and drop an image file onto the input field.
    *   A placeholder `[Image Attached: filename.png]` will appear. Add any accompanying text and send.
4.  **Insert Emojis:** Click the `:)` button to open the emoji picker and select an emoji to insert at your cursor position.
5.  **View Context:**
    *   **Retrieved Memories:** If the AI recalled past information, a collapsible "[+] Show Retrieved Memories" section appears below its response. Click to expand and see the relevant context.
    *   **Drive State:** A collapsible "[+] Show AI Drive State" section at the top displays the AI's current estimated drive levels (relative to its baseline) and mood.
6.  **Provide Feedback:** Click the 👍 or 👎 icons below an AI message to provide feedback, which influences the saliency (importance) of related memories.
7.  **Workspace Actions:** Ask the AI to perform file or calendar tasks (e.g., "save this conversation summary to notes.txt", "list my files", "add meeting with Bob tomorrow 2pm to calendar"). The AI will plan and execute these actions, showing results in task bubbles. You can click filenames in success messages to open them.
8.  **Manual Commands:**
    *   `/reset`: Type this in the input field and press Enter to confirm resetting the *entire* memory for the current personality (use with caution!).
    *   `/consolidate`: Type this to manually trigger the memory consolidation process.
9.  **Status Bar:** Shows the current status (Ready, Loading, Processing), the AI's estimated mood, and temporary messages.

## Tuning & Debugging

*   The `logs/tuning_log.jsonl` file contains detailed, structured information about internal processes like memory retrieval, consolidation steps, emotion calculations, and drive updates. Analyzing this file can be very helpful for understanding the AI's internal state and tuning the parameters in `config.yaml`.
*   Standard application logs are printed to the console. Increase logging levels for more detail if needed.

---

*Happy Chatting!*

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
