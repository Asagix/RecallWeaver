# Persistent Graph Memory System (RecallWeaver)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) <!-- Assuming MIT, update if needed -->

![Screenshot_Githubg](https://github.com/user-attachments/assets/7753a010-cb43-4d64-8ef4-597ed0debd3c)



## Overview

RecallWeaver is a Python application designed to give conversational AI systems a **persistent, long-term memory**. Instead of being limited by the context window of a Large Language Model (LLM), RecallWeaver stores conversational turns, extracted concepts, relationships, and even the AI's evolving self-perception in a **graph database**. This allows the AI to recall relevant past information semantically, leading to more coherent, context-aware, and personalized interactions.

Think of it as giving your AI a searchable diary and knowledge base that grows with every conversation.

**Core Technologies:**

*   **LLM Backend:** Designed primarily for [KoboldCpp](https://github.com/LostRuins/koboldcpp), but adaptable to other OpenAI-compatible APIs.
*   **Graph Database:** [NetworkX](https://networkx.org/) for flexible in-memory graph representation.
*   **Vector Search:** [FAISS](https://github.com/facebookresearch/faiss) for efficient semantic similarity search.
*   **Embeddings:** [Sentence Transformers](https://www.sbert.net/) for converting text into meaningful vector representations.
*   **NLP:** [spaCy](https://spacy.io/) for relationship extraction, [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for emotion classification, [VADER](https://github.com/cjhutto/vaderSentiment) for sentiment analysis.
*   **GUI:** [PyQt6](https://riverbankcomputing.com/software/pyqt/) for the desktop chat interface.
*   **Configuration:** [PyYAML](https://pyyaml.org/) for easy setup customization.

## Features

RecallWeaver boasts a rich set of features aimed at creating a more dynamic and persistent AI memory:

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
*   **🎭 Emotional Core:** Simulates a richer emotional state including basic emotions (via Transformer model), needs (Competence, Esteem, Belonging, etc.), fears (Failure, Rejection, etc.), and preferences (Clarity, Politeness, etc.). Uses LLM interpretation for needs/fears/prefs analysis and generates dynamic system instructions to influence the AI's tone and focus.
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

Getting RecallWeaver up and running involves a few steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Asagix/RecallWeaver.git
    cd RecallWeaver
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
    *   **Note on PyTorch:** `torch` is required by the `transformers` library for the emotion model. It's listed in `requirements.txt` and usually installed automatically. If you encounter issues (especially GPU-related), you might need to install a specific version manually following instructions on the [PyTorch website](https://pytorch.org/).
    *   **Note on Transformers:** The first time you run the application after installation, the `transformers` library might download the specified emotion classification model (e.g., `SamLowe/roberta-base-go_emotions`), which can take some time and disk space.

4.  **Download spaCy Model:**
    RecallWeaver uses a spaCy model for advanced relationship extraction during consolidation. Download the recommended English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up LLM Backend (KoboldCpp):**
    RecallWeaver needs a running LLM backend. KoboldCpp is recommended.
    *   **Download KoboldCpp:** Get the latest version from the [KoboldCpp GitHub releases](https://github.com/LostRuins/koboldcpp/releases).
    *   **Download an LLM Model:** You'll need a model file in GGUF format.
        *   **Recommendation:** A good starting point is an instruct-tuned model like **Gemma3 27b Instruct** (`gemma-3-27b-it-gguf`). Larger models generally provide better results for analysis tasks. Find models on Hugging Face.
    *   **Download Multimodal Projector (Optional but Recommended):** For image support, download the corresponding `.mmproj` file for your chosen model (e.g., the one for Gemma 3).
    *   **Run KoboldCpp with API and Multimodal Support:** Launch KoboldCpp from your terminal, enabling the API and specifying the multimodal projector:
        ```bash
        # Example command (adjust paths and parameters as needed):
        koboldcpp.exe --model /path/to/your/gemma-3-27b-it.Q4_K_M.gguf --useapi --port 5001 --mmproj /path/to/your/mmproj-gemma-3-27b-it-f16.gguf --threads 8 --contextsize 8192 --usemirostat 2 5.0 0.1
        ```
        *   `--model`: Path to your downloaded GGUF model file.
        *   `--useapi`: **Crucial!** Enables the necessary API endpoints.
        *   `--port 5001`: The port KoboldCpp will listen on (matches default in `config.yaml`).
        *   `--mmproj`: **Required for image support.** Path to the downloaded `.mmproj` file.
        *   Adjust `--threads`, `--contextsize`, GPU layers (`--gpulayers`), etc., based on your hardware. Refer to KoboldCpp documentation.
    *   Keep the KoboldCpp terminal window running while you use RecallWeaver.

## Configuration

Before running, review and potentially modify `config.yaml`:

*   **`base_memory_path`**: Where personality data folders will be stored.
*   **`kobold_api_url`**, **`kobold_chat_api_url`**: Ensure these match the URL and port of your running KoboldCpp instance (usually `http://localhost:5001/api/v1/generate` and `http://localhost:5001/v1/chat/completions`).
*   **`embedding_model`**, **`tokenizer_name`**: Models used for embeddings and token counting. Defaults are generally good (`all-MiniLM-L6-v2`, `google/gemma3-27b-it`).
*   **`llm_models`**: Configure parameters (max length, temperature, etc.) for different LLM tasks (chat, summary, analysis).
*   **`features`**: Enable or disable specific features like saliency, emotion analysis, forgetting, etc.
*   **`activation`**, **`consolidation`**, **`forgetting`**, **`memory_strength`**, **`subconscious_drives`**, **`emotional_core`**: Fine-tune parameters controlling memory retrieval, consolidation, forgetting, the drive system, and the new emotional core simulation.
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



