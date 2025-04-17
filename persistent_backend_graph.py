# persistent_backend_graph.py
import math
import os
import subprocess
import sys

import spacy
import json
import logging
import time
import uuid
import re
import networkx as nx
import numpy as np
import faiss
import requests
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from networkx.readwrite import json_graph
from datetime import datetime, timezone, timedelta
# Import zoneinfo safely
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    logging.warning("zoneinfo module not found. Using UTC. Consider `pip install tzdata`.") # Use logging directly here
    ZoneInfo = None # type: ignore
    ZoneInfoNotFoundError = Exception # Placeholder
from collections import defaultdict

# *** Import Emotion Analysis Library ***
try:
    import text2emotion as te
except ImportError:
    logging.warning("text2emotion library not found. Emotion analysis will be disabled. Run `pip install text2emotion`")
    te = None

# *** Import file manager ***
import file_manager # Assuming file_manager.py exists in the same directory

# --- Configuration ---
DEFAULT_CONFIG_PATH = "config.yaml"

# --- Logging Setup ---
# General logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Set default level higher
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set level for general logger

# --- Dedicated Tuning Logger ---
# Logs structured data for parameter analysis to a separate file
tuning_logger = logging.getLogger('TuningLogger')
tuning_logger.setLevel(logging.DEBUG) # Capture all tuning events
tuning_logger.propagate = False # Prevent tuning logs from going to the main handler

# Remove existing handlers for tuning_logger if any (e.g., during reload)
for handler in tuning_logger.handlers[:]:
    tuning_logger.removeHandler(handler)
    handler.close()

# Add a specific handler for the tuning log file
try:
    # Ensure the log directory exists (relative to this script)
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tuning_log_file = os.path.join(log_dir, 'tuning_log.jsonl')

    # Use FileHandler to append to the tuning log file
    tuning_handler = logging.FileHandler(tuning_log_file, mode='a', encoding='utf-8')
    # Use a simple formatter that just outputs the message (which will be JSON)
    tuning_formatter = logging.Formatter('%(message)s')
    tuning_handler.setFormatter(tuning_formatter)
    tuning_logger.addHandler(tuning_handler)
    tuning_logger.info(json.dumps({"event_type": "TUNING_LOG_INIT", "timestamp": datetime.now(timezone.utc).isoformat()}))
except Exception as e:
    logger.error(f"!!! Failed to configure tuning logger: {e}. Tuning logs will not be saved. !!!", exc_info=True)
    # Optionally disable tuning logging if setup fails
    tuning_logger.disabled = True


# --- Helper for Tuning Log ---
def log_tuning_event(event_type: str, data: dict):
    """Logs a structured event to the tuning log file."""
    if tuning_logger.disabled: return
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data # The specific data for this event type
        }
        tuning_logger.debug(json.dumps(log_entry)) # Use debug level for events
    except Exception as e:
        # Log error to the main logger if tuning log fails
        logger.error(f"Error logging tuning event '{event_type}': {e}", exc_info=True)


class GraphMemoryClient:
    """ Manages the persistent graph memory system, loading config from YAML. """
    def __init__(self, config_path=DEFAULT_CONFIG_PATH, personality_name=None):
        """Initializes the GraphMemoryClient for a specific personality."""
        logger.info(f"Initializing GraphMemoryClient (Personality: {personality_name or 'Default'})...")
        self._load_config(config_path) # Load main config

        # Determine personality and data directory
        if personality_name is None:
            personality_name = self.config.get('default_personality', 'default')
            logger.info(f"No personality specified, using default: {personality_name}")
        self.personality = personality_name

        base_memory_path = self.config.get('base_memory_path', 'memory_sets')
        self.data_dir = os.path.join(base_memory_path, self.personality) # Specific dir for this personality
        logger.info(f"Using data directory for '{self.personality}': {os.path.abspath(self.data_dir)}")

        # Construct file paths relative to the specific data_dir
        self.graph_file = os.path.join(self.data_dir, "memory_graph.json")
        self.index_file = os.path.join(self.data_dir, "memory_index.faiss")
        self.embeddings_file = os.path.join(self.data_dir, "memory_embeddings.npy")
        self.mapping_file = os.path.join(self.data_dir, "memory_mapping.json")
        self.asm_file = os.path.join(self.data_dir, "asm.json")
        self.drives_file = os.path.join(self.data_dir, "drives.json") # NEW: Drives file path

        # API URLs
        self.kobold_api_url = self.config.get('kobold_api_url', "http://localhost:5001/api/v1/generate")
        base_kobold_url = self.kobold_api_url.rsplit('/api/', 1)[0] if '/api/' in self.kobold_api_url else self.kobold_api_url
        self.kobold_chat_api_url = self.config.get('kobold_chat_api_url', f"{base_kobold_url}/v1/chat/completions") # Use updated logic
        logger.info(f"Using Kobold Generate API URL: {self.kobold_api_url}")
        logger.info(f"Using Kobold Chat Completions API URL: {self.kobold_chat_api_url}")

        # Initialize attributes
        self.graph = nx.DiGraph()
        self.index = None
        self.embeddings = {}
        self.faiss_id_to_uuid = {}
        self.uuid_to_faiss_id = {}
        self.last_added_node_uuid = None
        self.tokenizer = None
        self.embedder = None # Initialize embedder attribute explicitly
        self.embedding_dim = 0 # Initialize embedding_dim
        self.nlp = None # <<< Initialize spaCy model attribute
        # --- State for Contextual Retrieval Bias ---
        self.last_interaction_concept_uuids = set()
        self.last_interaction_mood = (0.0, 0.1) # Default mood (Valence, Arousal)
        # --- Autobiographical Self-Model ---
        self.autobiographical_model = {} # Initialize empty ASM
        # --- Subconscious Drive State (Combined) ---
        self.drive_state = {
            "short_term": {}, # {drive_name: activation_level} - Fluctuates based on recent events
            "long_term": {}   # {drive_name: level} - Stable, reflects core tendencies
        }

        os.makedirs(self.data_dir, exist_ok=True)
        embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        tokenizer_name = self.config.get('tokenizer_name', 'google/gemma-7b-it')
        # Use a default spacy model name if not specified in config
        spacy_model_name = self.config.get('spacy_model_name', 'en_core_web_sm') # Get model name from config (optional)

        # --- Load spaCy Model ---
        # Check feature flag before attempting to load
        features_cfg = self.config.get('features', {})
        rich_assoc_enabled = features_cfg.get('enable_rich_associations', False)
        if rich_assoc_enabled: # Only load if feature is intended to be used
            try:
                logger.info(f"Checking/Loading spaCy model: {spacy_model_name}")

                # Check if model is installed
                if not spacy.util.is_package(spacy_model_name):
                    logger.warning(f"spaCy model '{spacy_model_name}' not found. Attempting download...")
                    # Construct the command using the current Python executable
                    command = [sys.executable, "-m", "spacy", "download", spacy_model_name]
                    try:
                        # Run the command
                        result = subprocess.run(command, check=True, capture_output=True, text=True)
                        logger.info(f"Successfully downloaded spaCy model '{spacy_model_name}'.\nOutput:\n{result.stdout}")
                        # Mark nlp as potentially loadable, not setting to None yet
                        self.nlp = True # Use True as a temporary flag indicating download attempt/success
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to download spaCy model '{spacy_model_name}'. "
                                     f"Return code: {e.returncode}\nError Output:\n{e.stderr}\nStdout:\n{e.stdout}")
                        logger.error(f"Please try installing it manually: `python -m spacy download {spacy_model_name}`")
                        self.nlp = False # Set to False to indicate download failure
                    except FileNotFoundError:
                         logger.error(f"Could not run spacy download command. Is '{sys.executable}' correct and spacy installed?")
                         self.nlp = False
                    except Exception as download_e:
                         logger.error(f"An unexpected error occurred during spacy model download: {download_e}", exc_info=True)
                         self.nlp = False
                else:
                    logger.info(f"spaCy model '{spacy_model_name}' already installed.")
                    self.nlp = True # Mark as potentially loadable

                # Attempt to load the model only if download didn't fail or was skipped
                if self.nlp is True: # Check flag
                    try:
                        self.nlp = spacy.load(spacy_model_name)
                        logger.info(f"spaCy model '{spacy_model_name}' loaded successfully.")
                    except OSError as e:
                        logger.error(f"Could not load spaCy model '{spacy_model_name}' even after download check/attempt. {e}")
                        logger.error(f"Make sure it's installed correctly (`python -m spacy download {spacy_model_name}`). "
                                     "Rich association features will be disabled.")
                        self.nlp = None # Ensure it's None if loading fails
                    except Exception as e: # Catch other loading errors
                         logger.error(f"An unexpected error occurred loading the spaCy model '{spacy_model_name}': {e}", exc_info=True)
                         self.nlp = None
                else:
                     # If self.nlp is False (download failed), set it to None
                     logger.warning("Skipping spaCy model load due to download failure.")
                     self.nlp = None

            except ImportError:
                 logger.error("spaCy library not found. Please install it (`pip install spacy`). Rich association features will be disabled.")
                 self.nlp = None
            except Exception as e:
                 logger.error(f"An unexpected error occurred during spaCy setup: {e}", exc_info=True)
                 self.nlp = None
        else:
             logger.info("Rich association extraction feature disabled. Skipping spaCy model load.")
             self.nlp = None


        # Load Embedder
        try:
             logger.info(f"Loading embed model: {embedding_model_name}")
             self.embedder = SentenceTransformer(embedding_model_name, trust_remote_code=True)
             self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
             logger.info(f"Embed model loaded. Dim: {self.embedding_dim}")
             logger.debug(f"INIT Check 1: Has embedder? {hasattr(self, 'embedder')}, Type: {type(self.embedder)}, Dim: {self.embedding_dim}")
        except Exception as e:
             logger.error(f"Failed loading embed model: {e}", exc_info=True)
             self.embedder = None
             self.embedding_dim = 0
             # Consider if this should be a fatal error preventing initialization
             # raise # Or handle more gracefully

        # Load Tokenizer
        try:
             logger.info(f"Loading tokenizer: {tokenizer_name}")
             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
             logger.info("Tokenizer loaded.")
        except Exception as e:
             logger.error(f"Failed loading tokenizer '{tokenizer_name}': {e}", exc_info=True)
             self.tokenizer = None # Ensure tokenizer is None if loading fails
             # Decide if this is fatal

        self._load_memory() # Loads data from self.data_dir

        # Set last added node UUID
        if not self.last_added_node_uuid and self.graph.number_of_nodes() > 0:
             try:
                  # Find the node with the latest timestamp among active nodes
                  latest_node = self._find_latest_node_uuid()
                  if latest_node:
                       self.last_added_node_uuid = latest_node
                       logger.info(f"Set last_added_node_uuid from loaded graph to: {self.last_added_node_uuid[:8]}")
                  else:
                      logger.warning("Could not determine last added node from loaded graph.")
             except Exception as e:
                  logger.error(f"Error finding latest node during init: {e}", exc_info=True)


        logger.debug(f"INIT END: Has embedder? {hasattr(self, 'embedder')}")
        if hasattr(self, 'embedder') and self.embedder:
             logger.debug(f"INIT END: Embedder type: {type(self.embedder)}, Dim: {getattr(self, 'embedding_dim', 'Not Set')}")
        else:
             logger.error("INIT END: EMBEDDER ATTRIBUTE IS MISSING OR NONE!")

        # --- Log Key Config Parameters for Tuning ---
        try:
            key_params = {
                "personality": self.personality,
                "activation": self.config.get('activation', {}),
                "saliency": self.config.get('saliency', {}),
                "forgetting_weights": self.config.get('forgetting', {}).get('weights', {}),
                "forgetting_thresholds": {
                    "score_threshold": self.config.get('forgetting', {}).get('score_threshold'),
                    "candidate_min_age_hours": self.config.get('forgetting', {}).get('candidate_min_age_hours'),
                    "candidate_min_activation": self.config.get('forgetting', {}).get('candidate_min_activation'),
                },
                "memory_strength": self.config.get('memory_strength', {}),
                "consolidation": {
                    "turn_count": self.config.get('consolidation', {}).get('turn_count'),
                    "min_nodes": self.config.get('consolidation', {}).get('min_nodes'),
                    "concept_similarity_threshold": self.config.get('consolidation', {}).get('concept_similarity_threshold'),
                    "prune_summarized_turns": self.config.get('consolidation', {}).get('prune_summarized_turns'),
                    "inference_enabled": self.config.get('consolidation', {}).get('inference', {}).get('enable'),
                },
                "prompting_budget": {
                    "memory_budget_ratio": self.config.get('prompting', {}).get('memory_budget_ratio'),
                    "history_budget_ratio": self.config.get('prompting', {}).get('history_budget_ratio'),
                }
                # Add other key sections as needed
            }
            log_tuning_event("CONFIG_SNAPSHOT", key_params)
        except Exception as e:
            logger.error(f"Failed to log config snapshot for tuning: {e}", exc_info=True)

        logger.info(f"GraphMemoryClient initialized for personality '{self.personality}'.")


    # --- Emotion Analysis Helper ---
    def _analyze_and_update_emotion(self, node_uuid: str):
        """Analyzes text of a node and updates its emotion attributes."""
        if not te: # Check if library was imported
            # logger.debug("text2emotion library not available, skipping emotion analysis.")
            return
        if not self.config.get('features', {}).get('enable_emotion_analysis', False):
            # logger.debug("Emotion analysis feature disabled in config.")
            return

        if node_uuid not in self.graph:
            logger.warning(f"Cannot analyze emotion for non-existent node: {node_uuid}")
            return

        node_data = self.graph.nodes[node_uuid]
        text_to_analyze = node_data.get('text')

        if not text_to_analyze:
            # logger.debug(f"Node {node_uuid[:8]} has no text, skipping emotion analysis.")
            return

        try:
            # logger.debug(f"Analyzing emotion for node {node_uuid[:8]}...")
            emotion_scores = te.get_emotion(text_to_analyze)
            # Example: Map primary emotions to valence/arousal (Russell's Circumplex Model)
            # This is a VERY simplified mapping and needs refinement.
            # Valence: Happy(+) vs Sad/Angry/Fear(-)
            # Arousal: Angry/Fear/Surprise(+) vs Sad(-) vs Happy/Neutral(~)
            valence = 0.0
            arousal = 0.0 # Start slightly above zero baseline

            # Positive Valence
            valence += emotion_scores.get('Happy', 0.0) * 0.8
            valence += emotion_scores.get('Surprise', 0.0) * 0.2 # Surprise can be +/-

            # Negative Valence
            valence -= emotion_scores.get('Sad', 0.0) * 0.9
            valence -= emotion_scores.get('Angry', 0.0) * 0.7
            valence -= emotion_scores.get('Fear', 0.0) * 0.8

            # Arousal
            arousal += emotion_scores.get('Angry', 0.0) * 0.8
            arousal += emotion_scores.get('Fear', 0.0) * 0.9
            arousal += emotion_scores.get('Surprise', 0.0) * 0.7
            # Sadness can lower arousal slightly?
            arousal -= emotion_scores.get('Sad', 0.0) * 0.3
            # Happy can have moderate arousal
            arousal += emotion_scores.get('Happy', 0.0) * 0.4

            # Clamp values (e.g., -1 to 1 for valence, 0 to 1 for arousal)
            final_valence = max(-1.0, min(1.0, valence))
            final_arousal = max(0.0, min(1.0, arousal)) # Ensure arousal is non-negative

            # Update node attributes
            node_data['emotion_valence'] = final_valence
            node_data['emotion_arousal'] = final_arousal

            logger.debug(f"Updated emotion for node {node_uuid[:8]}: V={final_valence:.2f}, A={final_arousal:.2f} (Scores: {emotion_scores})")

        except Exception as e:
            logger.error(f"Error during text2emotion analysis for node {node_uuid[:8]}: {e}", exc_info=True)
            # Optionally reset to default if analysis fails?
            # node_data['emotion_valence'] = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
            # node_data['emotion_arousal'] = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)


    # --- Config Loading Method ---
    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        # (Keep implementation from previous version)
        logger.info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
            if not self.config: raise ValueError("Empty or invalid config file")
            logger.debug(f"Config loaded successfully: {self.config}")
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}. Using minimal defaults.")
            # Define comprehensive defaults here if config file is critical
            self.config = {'base_memory_path': 'memory_sets', 'default_personality': 'default', 'workspace_dir': 'Workspace', 'calendar_file': 'calendar.jsonl','embedding_model': 'all-MiniLM-L6-v2', 'tokenizer_name': 'google/gemma-7b-it', 'kobold_api_url': 'http://localhost:5001/api/v1/generate', 'activation': {'initial': 1.0, 'threshold': 0.1, 'node_decay_rate': 0.02, 'edge_decay_rate': 0.01, 'spreading_depth': 3, 'max_initial_nodes': 7, 'propagation_factor_base': 0.7, 'propagation_factors': {'TEMPORAL_fwd': 1.0, 'TEMPORAL_bwd': 0.8, 'SUMMARY_OF_fwd': 1.2, 'SUMMARY_OF_bwd': 0.4, 'MENTIONS_CONCEPT_fwd': 1.0, 'MENTIONS_CONCEPT_bwd': 0.9, 'ASSOCIATIVE': 0.8, 'HIERARCHICAL_fwd': 1.1, 'HIERARCHICAL_bwd': 0.5, 'UNKNOWN': 0.5}}, 'prompting': {'max_context_tokens': 4096, 'context_headroom': 250, 'memory_budget_ratio': 0.45, 'history_budget_ratio': 0.55}, 'consolidation': {'turn_count': 10, 'min_nodes': 5, 'concept_similarity_threshold': 0.3}, 'forget_topic': {'similarity_k': 15}, 'modification_keywords': []}
        except yaml.YAMLError as e: logger.error(f"Error parsing config file {config_path}: {e}", exc_info=True); raise ValueError(f"Invalid YAML: {config_path}")
        except Exception as e: logger.error(f"Unexpected error loading config: {e}", exc_info=True); raise


    # --- Helper Methods (_get_embedding, _load_memory, _rebuild..., _save_memory) ---
    # (Keep implementations from previous version)
    def _get_embedding(self, text: str) -> np.ndarray | None: # Return None on error
        """Generates a vector embedding for the given text."""
        # --- DEBUG Check at start of function ---
        logger.debug(f"GET_EMBEDDING START: Has embedder? {hasattr(self, 'embedder')}")
        if not hasattr(self, 'embedder') or self.embedder is None:
             logger.error("GET_EMBEDDING ERROR: self.embedder is missing or None!")
             return None # Return None explicitly if embedder missing

        if not text:
             # Return None instead of zeros? Or zeros of correct dim if available?
             if hasattr(self, 'embedding_dim') and self.embedding_dim > 0:
                 logger.warning("_get_embedding called with empty text, returning zeros.")
                 return np.zeros(self.embedding_dim, dtype='float32')
             else:
                 logger.error("_get_embedding called with empty text, and embedding_dim unknown!")
                 return None # Cannot return zeros if dim is unknown

        try:
             logger.debug(f"GET_EMBEDDING: Calling self.embedder.encode for text: '{text[:50]}...'")
             embedding = self.embedder.encode(text, convert_to_numpy=True)
             # Ensure embedding is float32 numpy array
             if isinstance(embedding, np.ndarray):
                 return embedding.astype('float32')
             else:
                 logger.error(f"Embedder returned unexpected type: {type(embedding)}")
                 return None # Return None if not numpy array
        except Exception as e:
             logger.error(f"Embedding encode error: {e}", exc_info=True)
             # Don't try to access self.embedding_dim if self.embedder might be missing
             return None # Return None on error


    def _load_memory(self):
        """Loads memory components from disk."""
        # (Keep implementation from previous version)
        logger.info("Attempting load memory..."); loaded_something = False
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r') as f: data = json.load(f); self.graph = json_graph.node_link_graph(data)
                logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
                loaded_something = True
            except Exception as e: logger.error(f"Failed loading graph: {e}", exc_info=True); self.graph = nx.DiGraph()
        else: logger.info(f"Graph file not found."); self.graph = nx.DiGraph()
        if os.path.exists(self.embeddings_file):
             try:
                 loaded_embeds = np.load(self.embeddings_file, allow_pickle=True).item()
                 self.embeddings = {u: np.array(e, dtype='float32') for u, e in loaded_embeds.items() if e and np.array(e).shape == (self.embedding_dim,)}
                 if len(loaded_embeds) != len(self.embeddings): logger.warning(f"Removed {len(loaded_embeds) - len(self.embeddings)} invalid embeds.")
                 logger.info(f"Loaded {len(self.embeddings)} valid embeddings.")
                 loaded_something = True
             except Exception as e: logger.error(f"Failed loading embeddings: {e}", exc_info=True); self.embeddings = {}
        else: logger.info(f"Embeddings file not found."); self.embeddings = {}
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f: loaded_map = json.load(f)
                valid_uuids = set(self.graph.nodes()) & set(self.embeddings.keys())
                self.faiss_id_to_uuid = {int(k): v for k, v in loaded_map.items() if v in valid_uuids}
                self.uuid_to_faiss_id = {v: k for k, v in self.faiss_id_to_uuid.items()}
                logger.info(f"Loaded FAISS mapping ({len(self.faiss_id_to_uuid)} valid entries).")
                loaded_something = True
            except Exception as e: logger.error(f"Failed loading FAISS mapping: {e}", exc_info=True); self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}
        else: logger.info(f"FAISS mapping file not found."); self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}
        index_needs_rebuild = False
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors).")
                if self.index.d != self.embedding_dim: logger.warning(f"Index/model dim mismatch. Rebuilding."); index_needs_rebuild = True
                elif self.index.ntotal != len(self.uuid_to_faiss_id): logger.warning(f"Index size ({self.index.ntotal}) != map size ({len(self.uuid_to_faiss_id)}). Rebuilding."); index_needs_rebuild = True
                loaded_something = True
            except Exception as e: logger.error(f"Failed loading FAISS index, will rebuild: {e}", exc_info=True); self.index = None; index_needs_rebuild = True
        else:
            logger.info(f"FAISS index file not found."); self.index = None
            if self.graph.number_of_nodes() > 0: index_needs_rebuild = True
        if index_needs_rebuild: self._rebuild_index_from_graph_embeddings()
        elif self.index is None: logger.info("Initializing empty FAISS index."); self.index = faiss.IndexFlatL2(self.embedding_dim)
        # --- Load ASM ---
        if os.path.exists(self.asm_file):
            try:
                with open(self.asm_file, 'r') as f: self.autobiographical_model = json.load(f)
                logger.info(f"Loaded Autobiographical Self-Model ({len(self.autobiographical_model)} keys).")
                loaded_something = True
            except Exception as e: logger.error(f"Failed loading ASM: {e}", exc_info=True); self.autobiographical_model = {}
        else: logger.info("ASM file not found."); self.autobiographical_model = {}
        # --- Load Drive State ---
        self._load_drive_state() # Load or initialize drive state

        if not loaded_something: logger.info("No existing memory data found.")
        else: logger.info("Memory loading complete.")

    def _rebuild_index_from_graph_embeddings(self):
        """Rebuilds FAISS index based on current graph nodes/embeddings."""
        logger.info(f"Rebuilding FAISS index from {self.graph.number_of_nodes()} graph nodes...")
        # forgetting_enabled = self.config.get('features', {}).get('enable_forgetting', False) # No longer needed here

        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph empty, initializing empty index.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_id_to_uuid = {}
            self.uuid_to_faiss_id = {}
            return

        try:
            # Use the current embedding dimension
            if not hasattr(self, 'embedding_dim') or self.embedding_dim <= 0:
                 logger.error("Cannot rebuild index: embedding_dim not set or invalid.")
                 # Should we try to determine it again? For now, error out.
                 raise ValueError("Embedding dimension unknown during index rebuild.")

            new_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim)) # Use IndexIDMap for easier removal later? No, stick to basic for now.
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            new_map = {}
            new_inv_map = {}
            emb_list = []
            uuid_list_for_index = [] # Keep track of UUIDs added to emb_list

            nodes_in_graph = list(self.graph.nodes(data=True)) # Get data as well

            nodes_with_valid_embeddings = 0
            for node_uuid, node_data in nodes_in_graph:
                # --- Filter by Valid Embedding ---
                embedding = self.embeddings.get(node_uuid)
                if embedding is not None and isinstance(embedding, np.ndarray) and embedding.shape == (self.embedding_dim,):
                    emb_list.append(embedding.astype('float32'))
                    uuid_list_for_index.append(node_uuid) # Track UUID associated with this embedding
                    nodes_with_valid_embeddings += 1
                else:
                    logger.warning(f"Skipping node {node_uuid[:8]} in rebuild (invalid/missing embedding).")

            # Add embeddings to the new index
            if emb_list:
                embeddings_np = np.vstack(emb_list)
                new_index.add(embeddings_np)
                logger.info(f"Added {new_index.ntotal} vectors (from {nodes_with_valid_embeddings} nodes with valid embeddings) to new FAISS index.")

                # Rebuild the mappings ONLY for the nodes added to the index
                current_faiss_id = 0
                for node_uuid in uuid_list_for_index:
                     new_map[current_faiss_id] = node_uuid
                     new_inv_map[node_uuid] = current_faiss_id
                     current_faiss_id += 1

                if new_index.ntotal != len(uuid_list_for_index):
                     logger.error(f"CRITICAL MISMATCH during rebuild: Index total ({new_index.ntotal}) != Added UUID count ({len(uuid_list_for_index)})")
                     # Fallback to empty?
                     new_index = faiss.IndexFlatL2(self.embedding_dim)
                     new_map = {}
                     new_inv_map = {}

            else:
                logger.warning("No valid embeddings found for active nodes during rebuild. Index will be empty.")
                # new_index is already an empty IndexFlatL2

            # Update the client's attributes
            self.index = new_index
            self.faiss_id_to_uuid = new_map
            self.uuid_to_faiss_id = new_inv_map

            logger.info(f"FAISS index rebuild complete. Index Size: {self.index.ntotal}. Map Size: {len(self.faiss_id_to_uuid)}.")

        except Exception as e:
            logger.error(f"Error during FAISS index rebuild: {e}", exc_info=True)
            # Fallback to an empty index on error
            logger.warning("Falling back to empty FAISS index due to rebuild error.")
            if hasattr(self, 'embedding_dim') and self.embedding_dim > 0:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                 self.index = None # Cannot create if dim unknown
                 logger.error("Cannot fallback to empty index: embedding_dim unknown.")
            self.faiss_id_to_uuid = {}
            self.uuid_to_faiss_id = {}

    def _save_memory(self):
        """Saves all memory components to disk."""
        # (Keep implementation from previous version - fixed save call)
        logger.info("Saving memory components..."); start_time = time.time()
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            try: graph_data = json_graph.node_link_data(self.graph); # Use defaults
            except Exception as e: logger.error(f"Error preparing graph data: {e}"); graph_data = None
            if graph_data:
                try:
                    with open(self.graph_file, 'w') as f: json.dump(graph_data, f, indent=4)
                except Exception as e: logger.error(f"Failed saving graph: {e}")
            try: embeds_to_save = {u: e.tolist() for u, e in self.embeddings.items()}; np.save(self.embeddings_file, embeds_to_save)
            except Exception as e: logger.error(f"Failed saving embeddings: {e}")
            if self.index and self.index.ntotal > 0:
                try: faiss.write_index(self.index, self.index_file)
                except Exception as e: logger.error(f"Failed saving FAISS index: {e}")
            elif self.index is not None and self.index.ntotal == 0:
                 if os.path.exists(self.index_file):
                     try: os.remove(self.index_file)
                     except OSError as e: logger.error(f"Error removing empty index file: {e}")
            try: map_to_save = {str(k): v for k, v in self.faiss_id_to_uuid.items()};
            except Exception as e: logger.error(f"Error preparing mapping: {e}"); map_to_save = None
            if map_to_save is not None:
                try:
                    with open(self.mapping_file, 'w') as f: json.dump(map_to_save, f, indent=4)
                except Exception as e: logger.error(f"Failed saving FAISS mapping: {e}")
            # --- Save ASM ---
            try:
                with open(self.asm_file, 'w') as f: json.dump(self.autobiographical_model, f, indent=4)
            except Exception as e: logger.error(f"Failed saving ASM: {e}")
            # --- Save Drive State ---
            self._save_drive_state()

            logger.info(f"Memory saving done ({time.time() - start_time:.2f}s).")
        except Exception as e: logger.error(f"Unexpected save error: {e}", exc_info=True)

    # --- Memory Node Management ---
    # (Keep add_memory_node, _rollback_add, delete_memory_entry, _find_latest_node_uuid, edit_memory_entry, forget_topic)
    # ... (methods unchanged) ...
    def add_memory_node(self, text: str, speaker: str, node_type: str = 'turn', timestamp: str = None, base_strength: float = 0.5) -> str | None:
        """Adds a new memory node with enhanced attributes to the graph and index."""
        logger.debug(f"ADD_MEMORY_NODE START: Has embedder? {hasattr(self, 'embedder')}")

        if not text: logger.warning("Skip adding empty node."); return None
        log_text = text[:80] + '...' if len(text) > 80 else text
        logger.info(f"Adding node: Spk={speaker}, Typ={node_type}, Txt='{log_text}'")
        current_time = time.time()
        node_uuid = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        # --- Get config values safely ---
        features_cfg = self.config.get('features', {})
        saliency_enabled = features_cfg.get('enable_saliency', False)
        emotion_cfg = self.config.get('emotion_analysis', {})
        default_valence = emotion_cfg.get('default_valence', 0.0)
        default_arousal = emotion_cfg.get('default_arousal', 0.1)
        saliency_cfg = self.config.get('saliency', {})
        initial_scores = saliency_cfg.get('initial_scores', {})
        emotion_influence = saliency_cfg.get('emotion_influence_factor', 0.0)


        # --- Calculate Initial Saliency ---
        initial_saliency = 0.0 # Default if disabled or error
        if saliency_enabled:
            base_saliency = initial_scores.get(node_type, initial_scores.get('default', 0.5))
            # Influence initial saliency by default arousal
            initial_saliency = base_saliency + (default_arousal * emotion_influence)
            initial_saliency = max(0.0, min(1.0, initial_saliency)) # Clamp between 0 and 1
            logger.debug(f"Calculated initial saliency for {node_uuid[:8]} ({node_type}): {initial_saliency:.3f} (Base: {base_saliency}, ArousalInf: {default_arousal * emotion_influence:.3f})")
        else:
             logger.debug(f"Saliency calculation disabled. Setting score to 0.0 for {node_uuid[:8]}.")


        # --- Get embedding ---
        embedding = self._get_embedding(text)
        if embedding is None:
             logger.error(f"Failed to get embedding for node {node_uuid}. Node not added.")
             return None

        # --- Add to graph with NEW attributes ---
        try:
            self.graph.add_node(
                node_uuid,
                uuid=node_uuid,
                text=text,
                speaker=speaker,
                timestamp=timestamp,
                node_type=node_type,
                # status='active', # REMOVED: Replaced by memory_strength
                memory_strength=self.config.get('memory_strength', {}).get('initial_value', 1.0), # NEW: Initial strength
                access_count=0, # Initial access count
                emotion_valence=default_valence, # Default emotion
                emotion_arousal=default_arousal, # Default emotion
                saliency_score=initial_saliency, # NEW: Calculated initial saliency
                # --- Existing attributes ---
                base_strength=float(base_strength),
                activation_level=0.0, # Initial activation, updated during retrieval
                last_accessed_ts=current_time, # Timestamp of last access/creation
                # --- NEW: Decay Resistance ---
                decay_resistance_factor=self.config.get('forgetting', {}).get('decay_resistance', {}).get(node_type, 1.0),
                # --- NEW: Feedback Score ---
                user_feedback_score=0 # Initialize feedback score
            )
            # Access node data *after* adding it to the graph
            node_data = self.graph.nodes[node_uuid]
            logger.debug(f"Node {node_uuid[:8]} added with decay_resistance: {node_data.get('decay_resistance_factor')}")
            logger.debug(f"Node {node_uuid[:8]} added to graph with new attributes (Strength: {node_data.get('memory_strength')}, Saliency: {node_data.get('saliency_score')}).")
        except Exception as e:
             logger.error(f"Failed adding node {node_uuid} to graph: {e}")
             return None

        # --- Add embedding to dictionary ---
        self.embeddings[node_uuid] = embedding

        # --- Add to FAISS ---
        try:
            if self.index is None:
                if hasattr(self, 'embedding_dim') and self.embedding_dim > 0:
                     logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
                     self.index = faiss.IndexFlatL2(self.embedding_dim)
                else:
                     logger.error("Cannot initialize FAISS index: embedding_dim not set.")
                     self._rollback_add(node_uuid)
                     return None

            # Add to FAISS index (no status check needed here anymore)
            self.index.add(np.array([embedding], dtype='float32'))
            new_faiss_id = self.index.ntotal - 1
            self.faiss_id_to_uuid[new_faiss_id] = node_uuid
            self.uuid_to_faiss_id[node_uuid] = new_faiss_id
            logger.debug(f"Embedding {node_uuid[:8]} added to FAISS ID {new_faiss_id}.")

        except Exception as e:
             logger.error(f"Failed adding embed {node_uuid} to FAISS: {e}")
             self._rollback_add(node_uuid)
             return None

        # --- Link temporally ---
        if self.last_added_node_uuid and self.last_added_node_uuid in self.graph:
            try:
                # Add temporal edge (no status check needed on predecessor)
                self.graph.add_edge(
                    self.last_added_node_uuid, node_uuid,
                    type='TEMPORAL', base_strength=0.8, last_traversed_ts=current_time
                )
                logger.debug(f"Added T-edge {self.last_added_node_uuid[:8]}->{node_uuid[:8]}.")
            except Exception as e:
                 logger.error(f"Failed adding T-edge: {e}")

        self.last_added_node_uuid = node_uuid
        logger.info(f"Successfully added node {node_uuid}.")
        return node_uuid

    def _rollback_add(self, node_uuid):
        """Internal helper to undo adding node/embedding if FAISS fails."""
        if node_uuid in self.graph: self.graph.remove_node(node_uuid)
        if node_uuid in self.embeddings: del self.embeddings[node_uuid]
        logger.warning(f"Rolled back add for {node_uuid}.")

    def delete_memory_entry(self, node_uuid: str) -> bool:
        """Deletes a specific memory node."""
        logger.info(f"Attempting delete: UUID={node_uuid[:8]}...")
        if node_uuid not in self.graph: logger.warning(f"Node {node_uuid} not found."); return False
        try:
            t_pred = next((p for p, _, d in self.graph.in_edges(node_uuid, data=True) if d.get('type')=='TEMPORAL'), None)
            if node_uuid in self.embeddings: del self.embeddings[node_uuid]; logger.debug(f"Removed embed {node_uuid[:8]}.")
            faiss_id = self.uuid_to_faiss_id.get(node_uuid)
            if faiss_id is not None:
                if faiss_id in self.faiss_id_to_uuid: del self.faiss_id_to_uuid[faiss_id]
                if node_uuid in self.uuid_to_faiss_id: del self.uuid_to_faiss_id[node_uuid]; logger.debug(f"Removed mappings {node_uuid[:8]}.")
            if node_uuid in self.graph: self.graph.remove_node(node_uuid); logger.debug(f"Removed node {node_uuid[:8]} from graph.")
            else: logger.warning(f"Node {node_uuid[:8]} already removed before explicit graph removal.")
            logger.info("Rebuilding FAISS index after deletion."); self._rebuild_index_from_graph_embeddings()
            if self.last_added_node_uuid == node_uuid: self.last_added_node_uuid = t_pred if t_pred and t_pred in self.graph else self._find_latest_node_uuid(); logger.info(f"Updated last_added_node_uuid to: {self.last_added_node_uuid}")
            logger.info(f"Successfully deleted {node_uuid[:8]}."); return True
        except Exception as e: logger.error(f"Error deleting {node_uuid}: {e}", exc_info=True); return False

    def _find_latest_node_uuid(self):
        """Finds the UUID of the node with the latest timestamp."""
        if not self.graph: return None
        valid_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('timestamp')]
        return max(valid_nodes, key=lambda x: x[1]['timestamp'])[0] if valid_nodes else None

    def edit_memory_entry(self, node_uuid: str, new_text: str) -> str | None:
        """Edits the text of a specific memory node by replacing it."""
        logger.info(f"Editing: UUID={node_uuid[:8]}, Text='{new_text[:50]}...'");
        if not new_text: logger.warning("Cannot edit with empty text."); return None
        if node_uuid not in self.graph: logger.warning(f"Node {node_uuid} not found."); return None
        try:
            orig_data = self.graph.nodes[node_uuid].copy()
            t_pred = next((p for p, _, d in self.graph.in_edges(node_uuid, data=True) if d.get('type')=='TEMPORAL'), None)
            t_succ = next((s for _, s, d in self.graph.out_edges(node_uuid, data=True) if d.get('type')=='TEMPORAL'), None)
        except Exception as e: logger.error(f"Failed get original data/neighbors for {node_uuid}: {e}"); return None
        delete_ok = self.delete_memory_entry(node_uuid)
        if not delete_ok: logger.error(f"Failed delete original {node_uuid}. Abort edit."); return None
        try:
            new_uuid = self.add_memory_node(text=new_text, speaker=orig_data.get('speaker'), node_type=orig_data.get('node_type', 'turn'), timestamp=orig_data.get('timestamp'), base_strength=orig_data.get('base_strength', 0.5))
            if new_uuid:
                current_time = time.time()
                if t_pred and t_pred in self.graph: self.graph.add_edge(t_pred, new_uuid, type='TEMPORAL', base_strength=0.8, last_traversed_ts=current_time); logger.debug(f"Relinked T-Pred {t_pred[:8]} -> {new_uuid[:8]}.")
                if t_succ and t_succ in self.graph: self.graph.add_edge(new_uuid, t_succ, type='TEMPORAL', base_strength=0.8, last_traversed_ts=current_time); logger.debug(f"Relinked {new_uuid[:8]} -> T-Succ {t_succ[:8]}.")
                logger.info(f"Successfully edited {node_uuid[:8]} -> {new_uuid[:8]}."); return new_uuid
            else: logger.error(f"Failed add edited node after delete {node_uuid}."); return None
        except Exception as e: logger.error(f"Error adding/relinking edited node for {node_uuid}: {e}"); return None

    def forget_topic(self, topic: str) -> tuple[bool, str]:
        """Identifies and deletes nodes related to a topic via similarity."""
        # (Keep implementation from previous version)
        similarity_k = self.config.get('forget_topic', {}).get('similarity_k', 15)
        logger.info(f"Forget topic request: '{topic[:50]}...'")
        if not topic: return False, "No topic provided."
        if self.index is None or self.index.ntotal == 0: return False, "Memory empty."
        logger.debug(f"Searching nodes similar to '{topic[:30]}...' (k={similarity_k})")
        similar_nodes = self._search_similar_nodes(topic, k=similarity_k)
        if not similar_nodes: return False, f"No memories found related to '{topic}'."
        nodes_to_delete = [uuid for uuid, score in similar_nodes]; logger.info(f"Found {len(nodes_to_delete)} candidate nodes for forgetting.")
        deleted_count, failed_count = 0, 0; deleted_uuids = []
        for node_uuid in list(nodes_to_delete):
            if node_uuid in self.graph:
                if self.delete_memory_entry(node_uuid): deleted_count += 1; deleted_uuids.append(node_uuid[:8])
                else: failed_count += 1; logger.warning(f"Failed deleting node {node_uuid} during forget.")
            else: logger.debug(f"Node {node_uuid} already deleted before forget loop.")
        if deleted_count > 0: message = f"Forgot topic '{topic}'. Deleted {deleted_count} entries." + (f" ({failed_count} fails)." if failed_count else ""); logger.info(message + f" UUIDs: {deleted_uuids}"); return True, message
        else: message = f"Could not delete memories for '{topic}'."; logger.warning(message); return False, message


    # --- Activation Spreading & Retrieval ---
    # (Keep _calculate_node_decay, _calculate_dynamic_edge_strength, _search_similar_nodes, retrieve_memory_chain from previous version)
    def _calculate_node_decay(self, node_data: dict, current_time: float) -> float:
        """Calculates exponential node activation decay multiplier."""
        node_decay_rate = self.config.get('activation', {}).get('node_decay_rate', 0.02)
        last_accessed = node_data.get('last_accessed_ts', current_time); time_delta = max(0, current_time - last_accessed); decay_mult = (1.0 - node_decay_rate) ** time_delta; return decay_mult
    def _calculate_dynamic_edge_strength(self, edge_data: dict, current_time: float) -> float:
        """Calculates exponential dynamic edge strength based on decay."""
        edge_decay_rate = self.config.get('activation', {}).get('edge_decay_rate', 0.01)
        base = edge_data.get('base_strength', 0.5); last_trav = edge_data.get('last_traversed_ts', current_time); time_delta = max(0, current_time - last_trav); decay_mult = (1.0 - edge_decay_rate) ** time_delta; dyn_str = base * decay_mult; return max(0.0, dyn_str)

    def _search_similar_nodes(self, query_text: str, k: int = None, node_type_filter: str = None, query_type: str = 'other') -> list[tuple[str, float]]:
        """
        Searches FAISS for nodes similar to query_text.
        Optionally filters by node_type_filter.
        Optionally biases search based on query_type ('episodic', 'semantic', 'other').
        """
        # --- Config Access ---
        act_cfg = self.config.get('activation', {})
        # features_cfg = self.config.get('features', {}) # No longer needed for status check
        # forgetting_enabled = features_cfg.get('enable_forgetting', False) # No longer needed for status check
        if k is None: k = act_cfg.get('max_initial_nodes', 7)

        if not query_text or self.index is None or self.index.ntotal == 0: return []
        try:
            q_embed = self._get_embedding(query_text)
            if q_embed is None or q_embed.shape != (self.embedding_dim,): return []

            q_embed_np = np.array([q_embed], dtype='float32')
            # Search more initially if filtering later by type or biasing by query_type
            search_multiplier = 3 # Default multiplier
            if query_type == 'episodic': search_multiplier = 5 # Search even more if biasing towards turns
            elif query_type == 'semantic': search_multiplier = 4 # Search more if biasing towards concepts/summaries

            search_k = k * search_multiplier
            actual_k = min(search_k, self.index.ntotal)

            if actual_k == 0: return []
            dists, idxs = self.index.search(q_embed_np, actual_k)
            results = []
            logger.debug(f"FAISS Search Results (Top {actual_k}, filter='{node_type_filter}', query_type='{query_type}'):")

            if len(idxs) > 0:
                for i, faiss_id in enumerate(idxs[0]):
                    fid_int = int(faiss_id); dist = float(dists[0][i])
                    logger.debug(f"  Rank {i+1}: ID={fid_int}, Dist={dist:.4f}")

                    if fid_int != -1:
                        node_uuid = self.faiss_id_to_uuid.get(fid_int)
                        if node_uuid and node_uuid in self.graph:
                            node_data = self.graph.nodes[node_uuid]
                            # node_status = node_data.get('status', 'active') # No longer needed
                            node_type = node_data.get('node_type')

                            # --- Filter 1: Explicit Node Type Filter ---
                            if node_type_filter and node_type != node_type_filter:
                                logger.debug(f"    -> Filtered (Explicit): UUID={node_uuid[:8]} (Type {node_type} != {node_type_filter})")
                                continue # Skip node type mismatch

                            node_type = node_data.get('node_type')

                            # --- Filter 1: Explicit Node Type Filter ---
                            if node_type_filter and node_type != node_type_filter:
                                logger.debug(f"    -> Filtered (Explicit): UUID={node_uuid[:8]} (Type {node_type} != {node_type_filter})")
                                continue # Skip node type mismatch

                            # --- Score Adjustment/Penalty based on Query Type ---
                            adjusted_dist = dist # Start with original distance
                            penalty_applied = False
                            if query_type == 'episodic' and node_type != 'turn':
                                # Apply a penalty to non-turn nodes for episodic queries
                                # Making them seem "further away"
                                penalty_factor = 1.5 # Example: Increase distance by 50%
                                adjusted_dist *= penalty_factor
                                penalty_applied = True
                                logger.debug(f"    -> Penalized (Episodic Bias): UUID={node_uuid[:8]} (Type {node_type} != 'turn'). Dist {dist:.3f} -> {adjusted_dist:.3f}")
                            elif query_type == 'semantic' and node_type not in ['summary', 'concept']:
                                # Apply a smaller penalty to non-summary/concept nodes for semantic queries
                                penalty_factor = 1.2 # Example: Increase distance by 20%
                                adjusted_dist *= penalty_factor
                                penalty_applied = True
                                logger.debug(f"    -> Penalized (Semantic Bias): UUID={node_uuid[:8]} (Type {node_type} not summary/concept). Dist {dist:.3f} -> {adjusted_dist:.3f}")

                            # --- Passed Filters ---
                            # Store the *adjusted* distance
                            results.append((node_uuid, adjusted_dist))
                            logger.debug(f"    -> Added Candidate: UUID={node_uuid[:8]} (Type: {node_type}, AdjDist: {adjusted_dist:.3f})")

                        else: logger.debug(f"    -> UUID for FAISS ID {fid_int} not in graph/map.")
                    else: logger.debug(f"    -> Invalid FAISS ID -1 encountered.")

                    # Stop if we have enough results after filtering
                    if len(results) >= k:
                        logger.debug(f"    -> Reached target k={k} results. Stopping search.")
                        break

            # Sort final results by the potentially *adjusted* distance
            results.sort(key=lambda item: item[1]) # item[1] is now the adjusted_dist
            logger.info(f"Found {len(results)} potentially relevant nodes (type='{node_type_filter or 'any'}', query_type='{query_type}') for query '{query_text[:30]}...'")
            # Return only top k based on adjusted distance
            final_results = [(uuid, dist) for uuid, dist in results[:k]]
            logger.debug(f" Final top {k} nodes after sorting by adjusted distance: {final_results}")
            return final_results

        except Exception as e: logger.error(f"FAISS search error: {e}", exc_info=True); return []

    def retrieve_memory_chain(self, initial_node_uuids: list[str],
                              recent_concept_uuids: list[str] | None = None,
                              current_mood: tuple[float, float] | None = None) -> list[dict]:
        """
        Retrieves relevant memories using activation spreading.
        Considers memory strength, saliency, edge types, optionally boosts recently mentioned concepts,
        and optionally biases based on emotional context similarity to current_mood.
        """
        # --- Config Access ---
        act_cfg = self.config.get('activation', {})
        features_cfg = self.config.get('features', {})
        saliency_cfg = self.config.get('saliency', {})
        # forgetting_cfg = self.config.get('forgetting', {}) # No longer needed for status check

        initial_activation = act_cfg.get('initial', 1.0)
        spreading_depth = act_cfg.get('spreading_depth', 3)
        activation_threshold = act_cfg.get('threshold', 0.1)
        prop_base = act_cfg.get('propagation_factor_base', 0.65)
        prop_factors = act_cfg.get('propagation_factors', {})
        # --- Load ALL propagation factors ---
        prop_temporal_fwd = prop_factors.get('TEMPORAL_fwd', 1.0)
        prop_temporal_bwd = prop_factors.get('TEMPORAL_bwd', 0.8)
        prop_summary_fwd = prop_factors.get('SUMMARY_OF_fwd', 1.1)
        prop_summary_bwd = prop_factors.get('SUMMARY_OF_bwd', 0.4)
        prop_concept_fwd = prop_factors.get('MENTIONS_CONCEPT_fwd', 1.0)
        prop_concept_bwd = prop_factors.get('MENTIONS_CONCEPT_bwd', 0.9)
        prop_assoc = prop_factors.get('ASSOCIATIVE', 0.8)
        prop_hier_fwd = prop_factors.get('HIERARCHICAL_fwd', 1.1)
        prop_hier_bwd = prop_factors.get('HIERARCHICAL_bwd', 0.5)
        # Load NEW factors (provide defaults matching config)
        prop_causes = prop_factors.get('CAUSES', 1.1)
        prop_part_of = prop_factors.get('PART_OF', 1.0)
        prop_has_prop = prop_factors.get('HAS_PROPERTY', 0.9)
        prop_enables = prop_factors.get('ENABLES', 1.0)
        prop_prevents = prop_factors.get('PREVENTS', 1.0)
        prop_contradicts = prop_factors.get('CONTRADICTS', 1.0)
        prop_supports = prop_factors.get('SUPPORTS', 1.0)
        prop_example_of = prop_factors.get('EXAMPLE_OF', 0.9)
        prop_measures = prop_factors.get('MEASURES', 0.9)
        prop_location_of = prop_factors.get('LOCATION_OF', 0.9)
        prop_analogy = prop_factors.get('ANALOGY', 0.8)
        prop_inferred = prop_factors.get('INFERRED_RELATED_TO', 0.6)
        prop_spacy = prop_factors.get('SPACY_REL', 0.7) # Generic for spaCy
        prop_unknown = prop_factors.get('UNKNOWN', 0.5) # Fallback

        guaranteed_saliency_threshold = act_cfg.get('guaranteed_saliency_threshold', 0.85)
        priming_boost_factor = act_cfg.get('priming_boost_factor', 1.0) # Get priming boost factor

        # --- Get Last Turn UUIDs for Priming (Internal) ---
        last_turn_uuids_for_priming = set()
        if self.last_added_node_uuid and self.last_added_node_uuid in self.graph:
            last_turn_uuids_for_priming.add(self.last_added_node_uuid)
            # Try to get the node before the last one via temporal link
            try:
                preds = list(self.graph.predecessors(self.last_added_node_uuid))
                # Find the temporal predecessor if multiple exist
                temporal_pred = next((p for p in preds if self.graph.get_edge_data(p, self.last_added_node_uuid, {}).get('type') == 'TEMPORAL'), None)
                if temporal_pred:
                    last_turn_uuids_for_priming.add(temporal_pred)
            except Exception as e:
                logger.warning(f"Could not get temporal predecessor for priming: {e}")
        logger.debug(f"Priming UUIDs identified internally: {last_turn_uuids_for_priming}")

        saliency_enabled = features_cfg.get('enable_saliency', False)
        activation_influence = saliency_cfg.get('activation_influence', 0.0) if saliency_enabled else 0.0
        context_focus_boost = act_cfg.get('context_focus_boost', 0.0) # Get boost factor, default 0 (no boost)
        recent_concept_uuids_set = set(recent_concept_uuids) if recent_concept_uuids else set() # Use set for faster lookup

        # --- Drive State Influence on Mood ---
        drive_cfg = self.config.get('subconscious_drives', {})
        mood_influence_cfg = drive_cfg.get('mood_influence', {})
        drives_enabled = drive_cfg.get('enabled', False)
        effective_mood = current_mood if current_mood else (0.0, 0.1) # Use provided mood or default

        # Check if drives are enabled and we have state data
        if drives_enabled and mood_influence_cfg and self.drive_state["short_term"]:
            logger.debug(f"Calculating mood adjustment based on drive state: ShortTerm={self.drive_state['short_term']}, LongTerm={self.drive_state['long_term']}")
            base_valence, base_arousal = effective_mood
            valence_adjustment = 0.0
            arousal_adjustment = 0.0
            valence_factors = mood_influence_cfg.get('valence_factors', {})
            arousal_factors = mood_influence_cfg.get('arousal_factors', {})
            base_drives = drive_cfg.get('base_drives', {}) # Use base_drives config
            long_term_influence = drive_cfg.get('long_term_influence_on_baseline', 1.0)

            for drive_name, current_activation in self.drive_state["short_term"].items():
                # Calculate the dynamic baseline for comparison
                config_baseline = base_drives.get(drive_name, 0.0)
                long_term_level = self.drive_state["long_term"].get(drive_name, 0.0)
                dynamic_baseline = config_baseline + (long_term_level * long_term_influence)

                # Calculate deviation from the *dynamic* baseline
                deviation = current_activation - dynamic_baseline
                # Positive deviation = drive level is *higher* than baseline (need potentially met/overshot).
                # Negative deviation = drive level is *lower* than baseline (need potentially unmet).

                # Apply factors based on deviation
                # Note: The sign of the factor in config determines the direction of influence.
                # e.g., Safety valence factor is negative (-0.25), so if deviation is negative (unmet need),
                # the valence adjustment will be positive (-0.25 * negative_dev = positive), pushing valence up slightly (less negative).
                # If deviation is positive (met need), adjustment is negative (-0.25 * positive_dev = negative), pushing valence down.
                # This seems counter-intuitive for Safety valence. Let's rethink.

                # --- Revised Mood Influence Logic ---
                # We want unmet needs (negative deviation) to generally decrease valence and increase arousal (stress/motivation).
                # We want met needs (positive deviation) to generally increase valence and potentially decrease arousal (calm/satisfaction).
                # The factors in config should represent the *strength* of this effect.
                valence_factor = valence_factors.get(drive_name, 0.0)
                arousal_factor = arousal_factors.get(drive_name, 0.0)

                # Valence: If factor is positive, positive deviation increases valence. If factor is negative, positive deviation decreases valence.
                valence_adj = valence_factor * deviation
                # Arousal: If factor is positive, positive deviation increases arousal. If factor is negative, positive deviation decreases arousal.
                # However, arousal is often increased by *both* strong satisfaction and strong frustration.
                # Let's use the *magnitude* of the deviation for arousal, scaled by the factor's sign.
                arousal_adj = arousal_factor * deviation # Keep simple for now, factor sign determines direction. Review config factors.
                arousal_adj = arousal_factors.get(drive_name, 0.0) * deviation
                valence_adjustment += valence_adj
                arousal_adjustment += arousal_adj
                logger.debug(f"  Drive '{drive_name}': Act={current_activation:.2f}, DynBase={dynamic_baseline:.2f}, Dev={deviation:.2f} -> V_adj={valence_adj:.3f}, A_adj={arousal_adj:.3f}")

            # Clamp total adjustment
            max_adj = mood_influence_cfg.get('max_mood_adjustment', 0.3)
            valence_adjustment = max(-max_adj, min(max_adj, valence_adjustment))
            arousal_adjustment = max(-max_adj, min(max_adj, arousal_adjustment))

            # Apply adjustment and clamp final mood
            adjusted_valence = max(-1.0, min(1.0, base_valence + valence_adjustment))
            adjusted_arousal = max(0.0, min(1.0, base_arousal + arousal_adjustment)) # Arousal >= 0

            effective_mood = (adjusted_valence, adjusted_arousal)
            logger.info(f"Mood adjusted by drives: Original=({base_valence:.2f},{base_arousal:.2f}) -> Adjusted=({effective_mood[0]:.2f},{effective_mood[1]:.2f})")
        else:
             logger.debug("Subconscious drives disabled or no config/state found, using original mood.")


        # --- Emotional Context Config (Uses effective_mood) ---
        emo_ctx_cfg = act_cfg.get('emotional_context', {})
        # Enable emotional context bias if the feature is on AND we have a valid mood (original or adjusted)
        emo_ctx_enabled = emo_ctx_cfg.get('enable', False) and effective_mood is not None
        emo_max_dist = emo_ctx_cfg.get('max_distance', 1.414)
        emo_boost = emo_ctx_cfg.get('boost_factor', 0.0) # Additive boost
        emo_penalty = emo_ctx_cfg.get('penalty_factor', 0.0) # Subtractive penalty

        logger.info(f"Starting retrieval. Initial nodes: {initial_node_uuids} (SalInf: {activation_influence:.2f}, GuarSal>=: {guaranteed_saliency_threshold}, FocusBoost: {context_focus_boost}, RecentConcepts: {len(recent_concept_uuids_set)}, EmoCtx: {emo_ctx_enabled})")
        # --- Tuning Log: Retrieval Start ---
        # Note: interaction_id is not directly available here, log context separately
        log_tuning_event("RETRIEVAL_START", {
            "personality": self.personality,
            "initial_node_uuids": initial_node_uuids,
            "recent_concept_uuids": list(recent_concept_uuids_set),
            "current_mood": effective_mood, # Log the potentially adjusted mood
            "saliency_influence": activation_influence,
            "guaranteed_saliency_threshold": guaranteed_saliency_threshold,
            "context_focus_boost": context_focus_boost,
            "emotional_context_enabled": emo_ctx_enabled,
        })

        if self.graph.number_of_nodes() == 0: logger.warning("Graph empty."); return []

        activation_levels = defaultdict(float)
        current_time = time.time()
        valid_initial_nodes = set()

        # --- Initialize Activation for initial nodes ---
        for uuid in initial_node_uuids:
            if uuid in self.graph:
                node_data = self.graph.nodes[uuid]
                # Apply initial strength modulation
                initial_strength = node_data.get('memory_strength', 1.0)
                base_initial_activation = initial_activation * initial_strength

                # --- Apply Context Focus Boost ---
                boost_applied = 1.0 # Default: no boost
                if context_focus_boost > 0 and recent_concept_uuids_set:
                    is_recent_concept = uuid in recent_concept_uuids_set
                    mentions_recent_concept = False
                    if not is_recent_concept: # Only check outgoing edges if node itself isn't the concept
                         try:
                              for succ_uuid in self.graph.successors(uuid):
                                   if succ_uuid in recent_concept_uuids_set:
                                        edge_data = self.graph.get_edge_data(uuid, succ_uuid)
                                        if edge_data and edge_data.get('type') == 'MENTIONS_CONCEPT':
                                             mentions_recent_concept = True
                                             break
                         except Exception as e: logger.warning(f"Error checking concept links for focus boost on {uuid[:8]}: {e}")

                    if is_recent_concept or mentions_recent_concept:
                         boost_applied = 1.0 + context_focus_boost
                         logger.debug(f"Applying context focus boost ({boost_applied:.2f}) to node {uuid[:8]} (IsRecent: {is_recent_concept}, MentionsRecent: {mentions_recent_concept})")

                final_initial_activation = base_initial_activation * boost_applied

                # --- Apply Priming Boost ---
                priming_applied = 1.0
                if uuid in last_turn_uuids_for_priming: # Check against internally derived set
                    priming_applied = priming_boost_factor
                    logger.debug(f"Applying priming boost ({priming_applied:.2f}) to last turn node {uuid[:8]}")

                final_initial_activation *= priming_applied # Apply priming boost multiplicatively

                activation_levels[uuid] = final_initial_activation
                node_data['last_accessed_ts'] = current_time # Update access time
                valid_initial_nodes.add(uuid)
                logger.debug(f"Initialized node {uuid[:8]} - Strength: {initial_strength:.3f}, BaseAct: {base_initial_activation:.3f}, CtxBoost: {boost_applied:.2f}, Priming: {priming_applied:.2f}, FinalAct: {final_initial_activation:.3f}")
            else:
                logger.warning(f"Initial node {uuid} not in graph.")

        if not activation_levels: logger.warning("No valid initial nodes found in graph."); return []

        logger.debug(f"Valid initial nodes: {len(valid_initial_nodes)}")
        active_nodes = set(activation_levels.keys()) # Nodes currently considered for spreading FROM

        # --- Activation Spreading Loop ---
        for depth in range(spreading_depth):
            logger.debug(f"--- Spreading Step {depth + 1} ---")
            newly_activated = defaultdict(float) # Activation gained in this step
            nodes_to_process = list(active_nodes) # Process nodes active at start of step
            logger.debug(f" Processing {len(nodes_to_process)} nodes.")

            for source_uuid in nodes_to_process:
                source_data = self.graph.nodes.get(source_uuid)
                if not source_data: continue
                source_act = activation_levels.get(source_uuid, 0)
                # Safely get saliency score, default to 0 if missing or not a number for calculation
                raw_saliency = source_data.get('saliency_score', 0.0)
                source_saliency = raw_saliency if isinstance(raw_saliency, (int, float)) else 0.0

                if source_act < 1e-6: continue # Skip if effectively inactive

                neighbors = set(self.graph.successors(source_uuid)) | set(self.graph.predecessors(source_uuid))

                for neighbor_uuid in neighbors:
                    if neighbor_uuid == source_uuid: continue
                    neighbor_data = self.graph.nodes.get(neighbor_uuid)
                    if not neighbor_data: continue

                    # No status check needed here anymore

                    is_forward = self.graph.has_edge(source_uuid, neighbor_uuid)
                    edge_data = self.graph.get_edge_data(source_uuid, neighbor_uuid) if is_forward else self.graph.get_edge_data(neighbor_uuid, source_uuid)
                    if not edge_data: continue

                    edge_type = edge_data.get('type', 'UNKNOWN')
                    type_factor = prop_unknown # Default

                    # --- Assign type_factor based on edge_type ---
                    # Note: Directionality might matter more for some types than others.
                    # For now, many new types use the same factor regardless of direction.
                    if edge_type == 'TEMPORAL': type_factor = prop_temporal_fwd if is_forward else prop_temporal_bwd
                    elif edge_type == 'SUMMARY_OF': type_factor = prop_summary_fwd if is_forward else prop_summary_bwd
                    elif edge_type == 'MENTIONS_CONCEPT': type_factor = prop_concept_fwd if is_forward else prop_concept_bwd
                    elif edge_type == 'ASSOCIATIVE': type_factor = prop_assoc
                    elif edge_type == 'HIERARCHICAL': type_factor = prop_hier_fwd if is_forward else prop_hier_bwd
                    elif edge_type == 'CAUSES': type_factor = prop_causes # Assume forward A->B means A causes B
                    elif edge_type == 'PART_OF': type_factor = prop_part_of
                    elif edge_type == 'HAS_PROPERTY': type_factor = prop_has_prop
                    elif edge_type == 'ENABLES': type_factor = prop_enables
                    elif edge_type == 'PREVENTS': type_factor = prop_prevents
                    elif edge_type == 'CONTRADICTS': type_factor = prop_contradicts
                    elif edge_type == 'SUPPORTS': type_factor = prop_supports
                    elif edge_type == 'EXAMPLE_OF': type_factor = prop_example_of
                    elif edge_type == 'MEASURES': type_factor = prop_measures
                    elif edge_type == 'LOCATION_OF': type_factor = prop_location_of
                    elif edge_type == 'ANALOGY': type_factor = prop_analogy
                    elif edge_type == 'INFERRED_RELATED_TO': type_factor = prop_inferred
                    elif edge_type.startswith('SPACY_'): type_factor = prop_spacy # Generic for all spaCy types
                    # else: type_factor remains prop_unknown

                    dyn_str = self._calculate_dynamic_edge_strength(edge_data, current_time)
                    saliency_boost = 1.0 + (source_saliency * activation_influence) if saliency_enabled else 1.0
                    # --- Apply neighbor's memory strength ---
                    neighbor_strength = neighbor_data.get('memory_strength', 1.0)
                    base_act_pass = source_act * dyn_str * prop_base * type_factor * saliency_boost * neighbor_strength

                    # --- Apply Emotional Context Bias ---
                    emo_adjustment = 0.0
                    if emo_ctx_enabled and base_act_pass > 1e-6: # Only calculate if base activation is non-negligible
                        try:
                            # Use defaults from config if node lacks emotion data
                            default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                            default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                            neighbor_v = neighbor_data.get('emotion_valence', default_v)
                            neighbor_a = neighbor_data.get('emotion_arousal', default_a)
                            mood_v, mood_a = effective_mood # Unpack potentially adjusted mood

                            # Calculate Euclidean distance in V/A space
                            dist_sq = (neighbor_v - mood_v)**2 + (neighbor_a - mood_a)**2
                            emo_dist = math.sqrt(dist_sq)

                            # Normalize distance (0=close, 1=far)
                            norm_dist = min(1.0, emo_dist / emo_max_dist) if emo_max_dist > 0 else 0.0

                            # Calculate adjustment: Boost for close (low norm_dist), penalize for far (high norm_dist)
                            # Linear scaling: Adjustment ranges from +emo_boost (at dist=0) to -emo_penalty (at dist=max_dist)
                            emo_adjustment = emo_boost * (1.0 - norm_dist) - emo_penalty * norm_dist
                            # logger.debug(f"    EmoCtx: Mood=({mood_v:.2f},{mood_a:.2f}), Nbr=({neighbor_v:.2f},{neighbor_a:.2f}), Dist={emo_dist:.3f}, NormDist={norm_dist:.3f}, Adjust={emo_adjustment:.3f}")

                        except Exception as e:
                             logger.warning(f"Error calculating emotional context bias for {neighbor_uuid[:8]}: {e}")
                             emo_adjustment = 0.0 # Default to no adjustment on error

                    # Apply adjustment (additive/subtractive)
                    act_pass = base_act_pass + emo_adjustment
                    # Ensure activation doesn't go below zero due to penalty
                    act_pass = max(0.0, act_pass)

                    if act_pass > 1e-6:
                        newly_activated[neighbor_uuid] += act_pass
                        # logger.debug(f"  Spread: {source_uuid[:8]}(A:{source_act:.2f},S:{source_saliency:.2f}) -> {neighbor_uuid[:8]}(Str:{neighbor_strength:.2f}) ({edge_type},{'F' if is_forward else 'B'}), EdgeStr:{dyn_str:.2f}, Factor:{type_factor:.2f}, SalBoost:{saliency_boost:.2f} => Pass:{act_pass:.3f}")
                        # logger.debug(f"  Spread: {source_uuid[:8]}(A:{source_act:.2f},S:{source_saliency:.2f}) -> {neighbor_uuid[:8]} ({edge_type},{'F' if is_forward else 'B'}), Str:{dyn_str:.2f}, Factor:{type_factor:.2f}, Boost:{saliency_boost:.2f} => Pass:{act_pass:.3f}")

                        edge_key = (source_uuid, neighbor_uuid) if is_forward else (neighbor_uuid, source_uuid)
                        if edge_key in self.graph.edges: self.graph.edges[edge_key]['last_traversed_ts'] = current_time

            # --- Apply Decay and Combine Activation ---
            nodes_to_decay = list(activation_levels.keys())
            active_nodes.clear()
            all_involved_nodes = set(nodes_to_decay) | set(newly_activated.keys())

            for uuid in all_involved_nodes:
                 node_data = self.graph.nodes.get(uuid)
                 if not node_data: continue

                 current_activation = activation_levels.get(uuid, 0.0)
                 if current_activation > 0:
                     decay_mult = self._calculate_node_decay(node_data, current_time)
                     activation_levels[uuid] *= decay_mult
                     # logger.debug(f"  Decay: {uuid[:8]} ({current_activation:.3f} * {decay_mult:.3f} -> {activation_levels[uuid]:.3f})")

                 activation_levels[uuid] += newly_activated.get(uuid, 0.0)
                 node_data['last_accessed_ts'] = current_time

                 if activation_levels[uuid] > 1e-6:
                     active_nodes.add(uuid)
                 elif uuid in activation_levels:
                     del activation_levels[uuid]

            logger.debug(f" Step {depth+1} finished. Active Nodes: {len(active_nodes)}. Max Activation: {max(activation_levels.values()) if activation_levels else 0:.3f}")
            if not active_nodes: break

        # --- Interference Simulation Step ---
        interference_cfg = act_cfg.get('interference', {})
        if interference_cfg.get('enable', False) and self.index and self.index.ntotal > 0:
            logger.info("--- Applying Interference Simulation ---")
            check_threshold = interference_cfg.get('check_threshold', 0.15)
            sim_threshold = interference_cfg.get('similarity_threshold', 0.25) # L2 distance
            penalty_factor = interference_cfg.get('penalty_factor', 0.90)
            k_neighbors = interference_cfg.get('max_neighbors_check', 5)
            penalized_nodes = set() # Track nodes penalized in this step
            interference_applied_count = 0

            # Iterate through nodes activated above the check threshold
            nodes_to_check = sorted(activation_levels.items(), key=lambda item: item[1], reverse=True)

            for source_uuid, source_activation in nodes_to_check:
                if source_uuid in penalized_nodes: continue # Already penalized, skip check
                if source_activation < check_threshold: continue # Below threshold to cause interference

                source_embedding = self.embeddings.get(source_uuid)
                if source_embedding is None: continue

                # Find nearest neighbors in embedding space
                try:
                    source_embed_np = np.array([source_embedding], dtype='float32')
                    distances, indices = self.index.search(source_embed_np, k_neighbors + 1) # Search k+1 to include self potentially

                    local_cluster = [] # (uuid, activation, distance)
                    if len(indices) > 0:
                        for i, faiss_id in enumerate(indices[0]):
                            neighbor_uuid = self.faiss_id_to_uuid.get(int(faiss_id))
                            if neighbor_uuid == source_uuid: continue # Skip self
                            neighbor_activation = activation_levels.get(neighbor_uuid)
                            distance = distances[0][i]

                            # Check if neighbor is activated and close enough
                            if neighbor_activation is not None and distance <= sim_threshold:
                                local_cluster.append((neighbor_uuid, neighbor_activation, distance))

                    # Apply interference if similar activated neighbors found
                    if local_cluster:
                        # Include source node itself in the cluster for comparison
                        cluster_with_source = [(source_uuid, source_activation, 0.0)] + local_cluster
                        # Find node with max activation in the cluster
                        dominant_uuid, max_act, _ = max(cluster_with_source, key=lambda item: item[1])

                        # Penalize non-dominant nodes in the cluster
                        for neighbor_uuid, neighbor_activation, dist in cluster_with_source:
                            if neighbor_uuid != dominant_uuid and neighbor_uuid not in penalized_nodes:
                                original_activation = activation_levels[neighbor_uuid]
                                activation_levels[neighbor_uuid] *= penalty_factor
                                penalized_nodes.add(neighbor_uuid)
                                interference_applied_count += 1
                                logger.debug(f"  Interference: Dominant '{dominant_uuid[:8]}' ({max_act:.3f}) penalized '{neighbor_uuid[:8]}'. "
                                             f"Activation {original_activation:.3f} -> {activation_levels[neighbor_uuid]:.3f} (Dist: {dist:.3f})")

                except Exception as e:
                    logger.error(f"Error during interference check for node {source_uuid[:8]}: {e}", exc_info=True)

            if interference_applied_count > 0:
                 logger.info(f"Interference applied to {interference_applied_count} node activations.")
            else:
                 logger.info("No interference applied in this retrieval.")
        else:
             logger.debug("Interference simulation disabled or index unavailable.")
        # --- Tuning Log: Interference Result ---
        log_tuning_event("RETRIEVAL_INTERFERENCE_RESULT", {
            "personality": self.personality,
            "initial_node_uuids": initial_node_uuids, # Include for context
            "interference_applied_count": interference_applied_count if 'interference_applied_count' in locals() else 0,
            "penalized_node_uuids": list(penalized_nodes) if 'penalized_nodes' in locals() else [],
        })


        # --- Final Selection & Update ---
        relevant_nodes_dict = {} # Use dict to avoid duplicates easily: uuid -> node_info
        processed_uuids_for_access_count = set()
        guaranteed_added_count = 0

        # Pass 1: Select nodes above activation threshold
        for uuid, final_activation in activation_levels.items():
            if final_activation >= activation_threshold:
                node_data = self.graph.nodes.get(uuid)
                # No status check needed here
                if node_data:
                    # Increment access count only once per retrieval
                    if uuid not in processed_uuids_for_access_count:
                        node_data['access_count'] = node_data.get('access_count', 0) + 1
                        processed_uuids_for_access_count.add(uuid)
                        logger.debug(f"Incremented access count for {uuid[:8]} to {node_data['access_count']}")

                    node_info = node_data.copy()
                    node_info['final_activation'] = final_activation
                    node_info['guaranteed_inclusion'] = False # Mark as normally included
                    relevant_nodes_dict[uuid] = node_info

                    # --- Boost Saliency on Successful Recall (Threshold Pass) ---
                    if saliency_enabled:
                        boost_factor = saliency_cfg.get('recall_boost_factor', 0.05)
                        if boost_factor > 0:
                            current_saliency = node_data.get('saliency_score', 0.0)
                            new_saliency = min(1.0, current_saliency + boost_factor) # Additive boost, clamped
                            if new_saliency > current_saliency:
                                node_data['saliency_score'] = new_saliency
                                logger.debug(f"Boosted saliency (threshold recall) for {uuid[:8]} to {new_saliency:.3f}")

                    # --- Emotional Reconsolidation (Threshold Pass) ---
                    if emo_ctx_enabled and emo_ctx_cfg.get('reconsolidation_enable', False):
                        recon_threshold = emo_ctx_cfg.get('reconsolidation_threshold', 0.5)
                        recon_factor = emo_ctx_cfg.get('reconsolidation_factor', 0.05)
                        if recon_factor > 0:
                            try:
                                default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                                default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                                node_v = node_data.get('emotion_valence', default_v)
                                node_a = node_data.get('emotion_arousal', default_a)
                                mood_v, mood_a = effective_mood # Use potentially adjusted mood
                                dist_sq = (node_v - mood_v)**2 + (node_a - mood_a)**2
                                emo_dist = math.sqrt(dist_sq)

                                if emo_dist >= recon_threshold:
                                    # Nudge node emotion towards current mood
                                    new_v = node_v + (mood_v - node_v) * recon_factor
                                    new_a = node_a + (mood_a - node_a) * recon_factor
                                    # Clamp values
                                    new_v = max(-1.0, min(1.0, new_v))
                                    new_a = max(0.0, min(1.0, new_a))
                                    # Update graph node directly
                                    self.graph.nodes[uuid]['emotion_valence'] = new_v
                                    self.graph.nodes[uuid]['emotion_arousal'] = new_a
                                    logger.debug(f"  EmoRecon (Thresh): Node {uuid[:8]} V/A ({node_v:.2f},{node_a:.2f}) nudged towards mood ({mood_v:.2f},{mood_a:.2f}) -> ({new_v:.2f},{new_a:.2f}). Dist={emo_dist:.3f}")
                            except Exception as e:
                                 logger.warning(f"Error during emotional reconsolidation for {uuid[:8]}: {e}")

        logger.info(f"Found {len(relevant_nodes_dict)} active nodes above activation threshold ({activation_threshold}).")

        # Pass 2: Check for high-saliency nodes missed by activation threshold
        for uuid, final_activation in activation_levels.items():
            if uuid not in relevant_nodes_dict: # Only check nodes not already included
                node_data = self.graph.nodes.get(uuid)
                # No status check needed here
                if node_data:
                    current_saliency = node_data.get('saliency_score', 0.0)
                    if current_saliency >= guaranteed_saliency_threshold:
                        logger.info(f"Guaranteed inclusion for node {uuid[:8]} (Sal: {current_saliency:.3f} >= {guaranteed_saliency_threshold}, Act: {final_activation:.3f} < {activation_threshold})")
                        # Increment access count if not already done
                        if uuid not in processed_uuids_for_access_count:
                            node_data['access_count'] = node_data.get('access_count', 0) + 1
                            processed_uuids_for_access_count.add(uuid)
                            logger.debug(f"Incremented access count for guaranteed node {uuid[:8]} to {node_data['access_count']}")

                        node_info = node_data.copy()
                        # Store the actual activation, even if below threshold
                        node_info['final_activation'] = final_activation
                        node_info['guaranteed_inclusion'] = True # Mark as guaranteed
                        relevant_nodes_dict[uuid] = node_info
                        guaranteed_added_count += 1

                        # --- Boost Saliency on Successful Recall (Guarantee Pass) ---
                        if saliency_enabled:
                            boost_factor = saliency_cfg.get('recall_boost_factor', 0.05)
                            if boost_factor > 0:
                                current_saliency = node_data.get('saliency_score', 0.0)
                                new_saliency = min(1.0, current_saliency + boost_factor) # Additive boost, clamped
                                if new_saliency > current_saliency:
                                    node_data['saliency_score'] = new_saliency
                                    logger.debug(f"Boosted saliency (guaranteed recall) for {uuid[:8]} to {new_saliency:.3f}")

                        # --- Emotional Reconsolidation (Guarantee Pass) ---
                        if emo_ctx_enabled and emo_ctx_cfg.get('reconsolidation_enable', False):
                            recon_threshold = emo_ctx_cfg.get('reconsolidation_threshold', 0.5)
                            recon_factor = emo_ctx_cfg.get('reconsolidation_factor', 0.05)
                            if recon_factor > 0:
                                try:
                                    default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                                    default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                                    node_v = node_data.get('emotion_valence', default_v)
                                    node_a = node_data.get('emotion_arousal', default_a)
                                    mood_v, mood_a = effective_mood # Use potentially adjusted mood
                                    dist_sq = (node_v - mood_v)**2 + (node_a - mood_a)**2
                                    emo_dist = math.sqrt(dist_sq)

                                    if emo_dist >= recon_threshold:
                                        # Nudge node emotion towards current mood
                                        new_v = node_v + (mood_v - node_v) * recon_factor
                                        new_a = node_a + (mood_a - node_a) * recon_factor
                                        # Clamp values
                                        new_v = max(-1.0, min(1.0, new_v))
                                        new_a = max(0.0, min(1.0, new_a))
                                        # Update graph node directly
                                        self.graph.nodes[uuid]['emotion_valence'] = new_v
                                        self.graph.nodes[uuid]['emotion_arousal'] = new_a
                                        logger.debug(f"  EmoRecon (Guar): Node {uuid[:8]} V/A ({node_v:.2f},{node_a:.2f}) nudged towards mood ({mood_v:.2f},{mood_a:.2f}) -> ({new_v:.2f},{new_a:.2f}). Dist={emo_dist:.3f}")
                                except Exception as e:
                                     logger.warning(f"Error during emotional reconsolidation for guaranteed node {uuid[:8]}: {e}")


        if guaranteed_added_count > 0:
            logger.info(f"Added {guaranteed_added_count} additional nodes due to high saliency guarantee.")

        # Convert dict back to list and sort
        relevant_nodes = list(relevant_nodes_dict.values())
        relevant_nodes.sort(key=lambda x: (x.get('final_activation', 0.0), x.get('timestamp', '')), reverse=True) # Sort primarily by activation

        if relevant_nodes: logger.info(f"Final nodes ({len(relevant_nodes)} total): [{', '.join([n['uuid'][:8] + '({:.3f}{})'.format(n['final_activation'], '*' if n['guaranteed_inclusion'] else '') for n in relevant_nodes])}]")
        else: logger.info("No relevant nodes found above threshold.")

        # --- Corrected Debug Logging ---
        logger.debug("--- Retrieved Node Details (Top 5) ---")
        for i, node in enumerate(relevant_nodes[:5]):
            # Safely get and format saliency and strength scores
            saliency_val = node.get('saliency_score', '?')
            strength_val = node.get('memory_strength', '?')
            saliency_str = f"{saliency_val:.2f}" if isinstance(saliency_val, (int, float)) else str(saliency_val)
            strength_str = f"{strength_val:.2f}" if isinstance(strength_val, (int, float)) else str(strength_val)
            # Format the log message including strength
            logger.debug(f"  {i+1}. ({node['final_activation']:.3f}) UUID:{node['uuid'][:8]} Str:{strength_str} Count:{node.get('access_count','?')} Sal:{saliency_str} Text: '{node.get('text', 'N/A')[:80]}...'")
        logger.debug("------------------------------------")

        # --- Tuning Log: Retrieval Result ---
        final_retrieved_data = [{
            "uuid": n['uuid'],
            "type": n.get('node_type'),
            "final_activation": n.get('final_activation'),
            "saliency_score": n.get('saliency_score'),
            "guaranteed": n.get('guaranteed_inclusion'),
            "text_preview": n.get('text', '')[:50]
        } for n in relevant_nodes]

        log_tuning_event("RETRIEVAL_RESULT", {
            "personality": self.personality,
            "initial_node_uuids": initial_node_uuids, # Include for context
            "activation_threshold": activation_threshold,
            "guaranteed_saliency_threshold": guaranteed_saliency_threshold,
            "final_retrieved_count": len(relevant_nodes),
            "final_retrieved_nodes": final_retrieved_data,
        })

        return relevant_nodes


    # --- Saliency & Forgetting ---

    def update_node_saliency(self, node_uuid: str, direction: str): # <<< Added 'direction' parameter
        """
        Updates the saliency score of a node based on user feedback ('increase' or 'decrease').
        Uses `saliency.feedback_factor` from config for multiplicative adjustment.
        `direction` should be 'increase' or 'decrease'.
        """
        if not self.config.get('features', {}).get('enable_saliency', False):
            logger.warning("Saliency feedback received but feature is disabled.")
            return # Check feature flag

        if node_uuid in self.graph:
            try:
                current_saliency = float(self.graph.nodes[node_uuid].get('saliency_score', 0.0))
                feedback_factor = float(self.config.get('saliency', {}).get('feedback_factor', 0.15)) # Get factor from config, default 0.15

                if feedback_factor <= 0:
                     logger.warning("Saliency feedback factor must be positive. Using default 0.15.")
                     feedback_factor = 0.15

                adjustment_multiplier = 1.0 + feedback_factor # Multiplier for increase
                if direction == 'increase':
                    new_saliency = current_saliency * adjustment_multiplier
                    logger.info(f"Increasing saliency for {node_uuid[:8]} (Factor: {adjustment_multiplier:.2f})")
                elif direction == 'decrease':
                    # Use inverse multiplier for decrease to ensure symmetry around 1.0
                    decrease_multiplier = 1.0 / adjustment_multiplier
                    new_saliency = current_saliency * decrease_multiplier
                    logger.info(f"Decreasing saliency for {node_uuid[:8]} (Factor: {decrease_multiplier:.2f})")
                else:
                    logger.warning(f"Invalid direction '{direction}' for saliency update.")
                    return # Do nothing if direction is invalid

                new_saliency = max(0.0, min(1.0, new_saliency)) # Clamp 0-1

                if new_saliency != current_saliency:
                    self.graph.nodes[node_uuid]['saliency_score'] = new_saliency
                    logger.info(f"Updated saliency for {node_uuid[:8]} from {current_saliency:.3f} to {new_saliency:.3f}")
                    # --- Tuning Log: Saliency Update ---
                    log_tuning_event("SALIENCY_UPDATE", {
                        "personality": self.personality,
                        "node_uuid": node_uuid,
                        "direction": direction,
                        "feedback_factor": feedback_factor,
                        "old_saliency": current_saliency,
                        "new_saliency": new_saliency,
                    })
                    # Save memory after update? Could be frequent. Maybe defer saving?
                    # Let's save for now to ensure persistence.
                    self._save_memory()

                    # --- Apply Heuristic Drive Adjustment for Saliency Feedback ---
                    try:
                        drive_cfg = self.config.get('subconscious_drives', {})
                        if drive_cfg.get('enabled', False):
                            heuristics = drive_cfg.get('heuristic_adjustment_factors', {})
                            adjustment = 0.0
                            target_drive = "Connection" # Feedback relates to connection

                            if direction == 'increase':
                                adjustment = heuristics.get('saliency_increase_connection', 0.0)
                            elif direction == 'decrease':
                                adjustment = heuristics.get('saliency_decrease_connection', 0.0)

                            if abs(adjustment) > 1e-4 and target_drive in self.drive_state["short_term"]:
                                current_level = self.drive_state["short_term"][target_drive]
                                new_level = current_level + adjustment
                                self.drive_state["short_term"][target_drive] = new_level
                                logger.info(f"Applied heuristic drive adjustment to '{target_drive}' due to saliency feedback ({direction}): {current_level:.3f} -> {new_level:.3f} (Adj: {adjustment:.3f})")
                                # Saving happens above
                    except Exception as e:
                        logger.error(f"Error applying heuristic drive adjustment after saliency update: {e}", exc_info=True)

                else:
                     logger.debug(f"Saliency for {node_uuid[:8]} unchanged (already at limit or no effective change).")

            except Exception as e:
                 logger.error(f"Error updating saliency for node {node_uuid[:8]}: {e}", exc_info=True)
        else:
            logger.warning(f"update_node_saliency called for non-existent node: {node_uuid}")

    def apply_feedback(self, node_uuid: str, feedback_type: str):
        """
        Applies user feedback (thumbs up/down) to a node, primarily by adjusting saliency.
        """
        feedback_cfg = self.config.get('feedback_system', {})
        if not feedback_cfg.get('enable', False):
            logger.debug("Feedback system disabled. Ignoring feedback.")
            return

        if node_uuid not in self.graph:
            logger.warning(f"Cannot apply feedback to non-existent node: {node_uuid}")
            return

        logger.info(f"Applying feedback '{feedback_type}' to node {node_uuid[:8]}")
        node_data = self.graph.nodes[node_uuid]
        current_feedback_score = node_data.get('user_feedback_score', 0)
        current_saliency = node_data.get('saliency_score', 0.0)
        new_feedback_score = current_feedback_score
        saliency_adjustment = 0.0

        try:
            if feedback_type == 'up':
                new_feedback_score += 1
                saliency_adjustment = feedback_cfg.get('saliency_upvote_boost', 0.1)
            elif feedback_type == 'down':
                new_feedback_score -= 1
                saliency_adjustment = -feedback_cfg.get('saliency_downvote_penalty', 0.15) # Penalty is subtractive
            else:
                logger.warning(f"Invalid feedback type received: {feedback_type}")
                return

            # Update feedback score
            node_data['user_feedback_score'] = new_feedback_score
            logger.debug(f"  Updated feedback score: {current_feedback_score} -> {new_feedback_score}")

            # Update saliency score
            if abs(saliency_adjustment) > 1e-4:
                new_saliency = current_saliency + saliency_adjustment
                new_saliency = max(0.0, min(1.0, new_saliency)) # Clamp 0-1
                if new_saliency != current_saliency:
                    node_data['saliency_score'] = new_saliency
                    logger.info(f"  Adjusted saliency due to feedback: {current_saliency:.3f} -> {new_saliency:.3f} (Adj: {saliency_adjustment:.3f})")
                else:
                    logger.debug("  Saliency unchanged (at limit or no effective change).")

            # Save changes
            self._save_memory()

        except Exception as e:
            logger.error(f"Error applying feedback to node {node_uuid[:8]}: {e}", exc_info=True)


    # --- Forgetting Mechanism ---
    def run_memory_maintenance(self):
        """
        Runs the memory strength reduction process based on forgettability.
        Identifies candidate nodes, calculates forgettability scores, and reduces
        memory_strength based on the score and decay rate.
        Triggered periodically based on interaction count or other criteria.
        """
        if not self.config.get('features', {}).get('enable_forgetting', False):
            logger.debug("Memory strength reduction feature disabled. Skipping maintenance.")
            return

        logger.info("--- Running Memory Maintenance (Strength Reduction) ---")
        # --- Tuning Log: Maintenance Start ---
        log_tuning_event("MAINTENANCE_STRENGTH_START", {"personality": self.personality})

        # 1. Get config: weights, min_age, min_activation, strength decay rate
        forget_cfg = self.config.get('forgetting', {})
        weights = forget_cfg.get('weights', {})
        min_age_hr = forget_cfg.get('candidate_min_age_hours', 24)
        min_activation_threshold = forget_cfg.get('candidate_min_activation', 0.05)
        strength_cfg = self.config.get('memory_strength', {})
        strength_decay_rate = strength_cfg.get('decay_rate', 0.1)
        logger.debug(
            f"Strength Reduction Params: MinAgeHr={min_age_hr}, MinAct={min_activation_threshold}, DecayRate={strength_decay_rate}, Weights={weights}")

        # 2. Identify Candidate Nodes for strength reduction check:
        #    - age > min_age_hr (based on last_accessed_ts)
        #    - activation_level < min_activation (using stored 'activation_level')
        #    - memory_strength > purge_threshold (don't decay already very weak nodes?) - Optional optimization
        candidate_uuids = []
        current_time = time.time()
        min_age_sec = min_age_hr * 3600
        nodes_to_check = list(self.graph.nodes(data=True)) # Get a snapshot
        logger.debug(f"Checking {len(nodes_to_check)} nodes for strength reduction candidacy...")

        for uuid, data in nodes_to_check:
            # Filter 1: Age must be sufficient
            last_accessed = data.get('last_accessed_ts', 0)
            age_sec = current_time - last_accessed
            if age_sec < min_age_sec:
                # logger.debug(f"  Skip {uuid[:8]}: Too recent (Age: {age_sec/3600:.1f}h < {min_age_hr}h)")
                continue
            # Filter 2: Activation must be low enough
            # Note: 'activation_level' reflects the *last calculated* activation during retrieval.
            # An alternative could be to calculate decay *here* based on last_accessed_ts,
            # but that might be computationally heavier. Using stored value for now.
            current_activation = data.get('activation_level', 0.0)
            if current_activation >= min_activation_threshold:
                # logger.debug(f"  Skip {uuid[:8]}: Activation too high ({current_activation:.3f} >= {min_activation_threshold})")
                continue

            # Passed all filters
            # logger.debug(f"  Candidate {uuid[:8]}: Type={data.get('node_type')}, Age={age_sec/3600:.1f}h, Act={current_activation:.3f}")
            candidate_uuids.append(uuid)

        logger.info(f"Found {len(candidate_uuids)} candidate nodes for potential strength reduction.")
        # --- Tuning Log: Maintenance Candidates ---
        log_tuning_event("MAINTENANCE_STRENGTH_CANDIDATES", {
            "personality": self.personality,
            "candidate_count": len(candidate_uuids),
            "candidate_uuids": candidate_uuids, # Log all candidates
        })

        if not candidate_uuids:
            logger.info("--- Memory Maintenance Finished (No candidates) ---")
            # --- Tuning Log: Maintenance End ---
            log_tuning_event("MAINTENANCE_STRENGTH_END", {
                "personality": self.personality,
                "strength_reduced_count": 0,
                "nodes_changed": False,
            })
            return

        # 3. Calculate Forgettability Score and Reduce Strength for each candidate:
        strength_reduced_count = 0 # Initialize counter
        nodes_changed = False # Initialize flag
        for uuid in candidate_uuids:
            if uuid not in self.graph: continue # Node might have been deleted since snapshot
            node_data = self.graph.nodes[uuid]
            forget_score = self._calculate_forgettability(uuid, node_data, current_time, weights)
            logger.debug(f"  Node {uuid[:8]} ({node_data.get('node_type')}): Forgettability Score = {forget_score:.3f}")
            # --- Tuning Log: Forgettability Score ---
            log_tuning_event("MAINTENANCE_FORGETTABILITY_SCORE", {
                "personality": self.personality,
                "node_uuid": uuid,
                "node_type": node_data.get('node_type'),
                "forgettability_score": forget_score,
                # Optionally log contributing factors if needed, but might be too verbose
            })

            # 4. Reduce memory_strength based on score and decay rate
            current_strength = node_data.get('memory_strength', 1.0)
            # Strength reduction is proportional to forgettability score and decay rate
            strength_reduction = forget_score * strength_decay_rate
            new_strength = current_strength * (1.0 - strength_reduction)
            new_strength = max(0.0, new_strength) # Ensure strength doesn't go below 0

            if new_strength < current_strength:
                node_data['memory_strength'] = new_strength
                strength_reduced_count += 1
                nodes_changed = True
                logger.info(f"  Reduced strength for node {uuid[:8]} from {current_strength:.3f} to {new_strength:.3f} (ForgetScore: {forget_score:.3f}, Rate: {strength_decay_rate})")
                # --- Tuning Log: Strength Reduced ---
                log_tuning_event("MAINTENANCE_STRENGTH_REDUCED", {
                    "personality": self.personality,
                    "node_uuid": uuid,
                    "node_type": node_data.get('node_type'),
                    "forgettability_score": forget_score,
                    "old_strength": current_strength,
                    "new_strength": new_strength,
                    "decay_rate": strength_decay_rate,
                })
            # else:
                # logger.debug(f"  Strength for node {uuid[:8]} remains {current_strength:.3f} (Reduction: {strength_reduction:.3f})")

        # 5. Save memory if any strengths were changed
        if nodes_changed:
            logger.info(f"Reduced strength for {strength_reduced_count} nodes. Saving memory...")
            self._save_memory() # Save changes after maintenance
        else:
            logger.info("No node strengths were reduced in this maintenance cycle.")

        # --- Moved Logging Block ---
        logger.info(f"--- Memory Maintenance Finished ({strength_reduced_count} strengths reduced) ---")
        # --- Tuning Log: Maintenance End ---
        log_tuning_event("MAINTENANCE_STRENGTH_END", {
            "personality": self.personality,
            "strength_reduced_count": strength_reduced_count,
            "nodes_changed": nodes_changed,
        })
        # --- End Moved Logging Block ---


    def _calculate_forgettability(self, node_uuid: str, node_data: dict, current_time: float,
                                  weights: dict) -> float:
        """
        Calculates a score indicating how likely a node is to be forgotten (0-1).
        Higher score means more likely to be forgotten.
        Uses normalized factors based on node attributes and configured weights.
        """
        # --- Get Raw Factors ---
        # Recency: Time since last access (higher = more forgettable)
        last_accessed = node_data.get('last_accessed_ts', 0)
        recency_sec = max(0, current_time - last_accessed)

        # Activation: Current activation level (lower = more forgettable)
        activation = node_data.get('activation_level', 0.0) # Graph node's stored activation

        # Node Type: Some types intrinsically more forgettable
        node_type = node_data.get('node_type', 'default')

        # Saliency: Higher saliency resists forgetting
        saliency = node_data.get('saliency_score', 0.0)

        # Emotion: Higher arousal/valence magnitude resists forgetting
        valence = node_data.get('emotion_valence', 0.0)
        arousal = node_data.get('emotion_arousal', 0.1)
        # Use absolute values for magnitude calculation
        emotion_magnitude = math.sqrt(abs(valence) ** 2 + abs(arousal) ** 2)

        # Connectivity: Higher degree resists forgetting
        degree = self.graph.degree(node_uuid) if node_uuid in self.graph else 0
        # Consider in/out degree separately? For now, total degree.

        # Access Count: Higher count resists forgetting
        access_count = node_data.get('access_count', 0)

        # --- Normalize Factors (Example - needs tuning via config weights) ---
        # Normalize recency using exponential decay (Ebbinghaus-like curve)
        # Higher decay constant = faster forgetting
        decay_constant = weights.get('recency_decay_constant', 0.000005) # Default decay over seconds
        norm_recency = 1.0 - math.exp(-decay_constant * recency_sec) # Score approaches 1 as time increases

        # Normalize activation (already 0-1 theoretically, but use inverse)
        # Low activation -> high score component
        norm_inv_activation = 1.0 - min(1.0, max(0.0, activation))

        # Normalize node type factor (example mapping - higher value = more forgettable)
        type_map = {'turn': 1.0, 'summary': 0.4, 'concept': 0.1, 'default': 0.6}
        norm_type_forgettability = type_map.get(node_type, 0.6)

        # Normalize saliency (use inverse: low saliency -> high score component)
        norm_inv_saliency = 1.0 - min(1.0, max(0.0, saliency))

        # Normalize emotion (use inverse: low magnitude -> high score component)
        # Normalize magnitude based on potential range (e.g., 0 to sqrt(1^2+1^2) approx 1.414)
        norm_inv_emotion = 1.0 - min(1.0, max(0.0, emotion_magnitude / 1.414))

        # Normalize connectivity (use inverse, map degree to 0-1 range, e.g., log scale or capped)
        # Example: cap at 10 neighbors for normalization, inverse log scale might be better
        norm_inv_connectivity = 1.0 - min(1.0, math.log1p(degree) / math.log1p(10)) # Log scale, capped effect

        # Normalize access count (use inverse, map count to 0-1 range)
        # Example: cap at 20 accesses for normalization, inverse log scale
        norm_inv_access_count = 1.0 - min(1.0, math.log1p(access_count) / math.log1p(20))

        # --- Calculate Weighted Score ---
        # Factors increasing forgettability score (higher value = more forgettable)
        score = 0.0
        score += norm_recency * weights.get('recency_factor', 0.0)
        score += norm_inv_activation * weights.get('activation_factor', 0.0)
        score += norm_type_forgettability * weights.get('node_type_factor', 0.0)

        # Factors decreasing forgettability score (resistance factors)
        # These use inverse normalization, so apply positive weights from config
        # (Config weights represent importance of the factor)
        score += norm_inv_saliency * weights.get('saliency_factor', 0.0)
        score += norm_inv_emotion * weights.get('emotion_factor', 0.0)
        score += norm_inv_connectivity * weights.get('connectivity_factor', 0.0)
        score += norm_inv_access_count * weights.get('access_count_factor', 0.0) # Added access count factor

        # Clamp final score 0-1
        final_score = max(0.0, min(1.0, score))

        # logger.debug(f"    Forget Score Factors for {node_uuid[:8]}: Rec({norm_recency:.2f}), Act({norm_inv_activation:.2f}), Typ({norm_type_forgettability:.2f}), Sal({norm_inv_saliency:.2f}), Emo({norm_inv_emotion:.2f}), Con({norm_inv_connectivity:.2f}), Acc({norm_inv_access_count:.2f}) -> Initial Score: {final_score:.3f}") # Log initial score before adjustments

        # --- Apply Decay Resistance (Type-Based) ---
        type_resistance_factor = node_data.get('decay_resistance_factor', 1.0)
        score_after_type_resistance = final_score * type_resistance_factor
        logger.debug(f"    Node {node_uuid[:8]} Type Resistance Factor: {type_resistance_factor:.3f}. Score after type resist: {score_after_type_resistance:.4f}") # Corrected uuid -> node_uuid

        # Initialize final_adjusted_score
        final_adjusted_score = score_after_type_resistance

        # --- Apply Emotion Magnitude Resistance ---
        emotion_magnitude_resistance_factor = weights.get('emotion_magnitude_resistance_factor', 0.0)
        if emotion_magnitude_resistance_factor > 0:
            # Calculate emotion magnitude (already done above)
            # Reduce forgettability score based on magnitude (higher magnitude = lower score)
            # Ensure factor is clamped 0-1 to avoid negative scores
            clamped_emo_mag = min(1.0, max(0.0, emotion_magnitude / 1.414)) # Normalize approx 0-1
            emotion_resistance_multiplier = (1.0 - clamped_emo_mag * emotion_magnitude_resistance_factor)
            # Update final_adjusted_score
            final_adjusted_score *= emotion_resistance_multiplier
            logger.debug(f"    Node {node_uuid[:8]} Emotion Mag: {emotion_magnitude:.3f} (Norm: {clamped_emo_mag:.3f}), Emo Resist Factor: {emotion_resistance_multiplier:.3f}. Score updated to: {final_adjusted_score:.4f}") # Corrected uuid -> node_uuid
        # else: No emotion resistance applied, final_adjusted_score remains score_after_type_resistance

        # Log the final score being returned (moved outside the if/else)
        logger.debug(f"    Final Adjusted Forgettability Score for {node_uuid[:8]}: {final_adjusted_score:.4f}") # noqa: F821
        return final_adjusted_score

    def _get_relative_time_desc(self, timestamp_str: str) -> str:
        """Converts an ISO timestamp string into a human-readable relative time description."""
        if not timestamp_str: return "Timestamp Unknown"
        try:
            # Ensure timezone awareness
            target_dt_utc = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if target_dt_utc.tzinfo is None:
                target_dt_utc = target_dt_utc.replace(tzinfo=timezone.utc)

            now_utc = datetime.now(timezone.utc)
            delta = now_utc - target_dt_utc

            seconds = delta.total_seconds()
            days = delta.days

            if seconds < 60: return "just now"
            if seconds < 3600: return f"{int(seconds / 60)} minutes ago"
            if seconds < 86400: # Less than a day
                hours = int(seconds / 3600)
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            if days == 1: return "yesterday"
            if days < 7: return f"{days} days ago" # Or target_dt_utc.strftime('%A') for "last Tuesday"? Let's keep it simple.
            if days < 30: return f"{int(days / 7)} week{'s' if int(days / 7) > 1 else ''} ago"
            if days < 365: return f"{int(days / 30)} month{'s' if int(days / 30) > 1 else ''} ago"
            else: return f"{int(days / 365)} year{'s' if int(days / 365) > 1 else ''} ago" # Or return the date "on YYYY-MM-DD"?

        except ValueError:
            logger.warning(f"Could not parse timestamp for relative description: {timestamp_str}")
            return f"on {timestamp_str[:10]}" # Fallback to date part
        except Exception as e:
            logger.error(f"Error generating relative time description: {e}", exc_info=True)
            return "Timestamp Error"

    def purge_weak_nodes(self):
        """
        Permanently deletes nodes whose memory_strength is below a configured threshold
        and are older than a configured minimum age.
        """
        if not self.config.get('features', {}).get('enable_forgetting', False):
            logger.debug("Forgetting/Purging feature disabled. Skipping purge.")
            return

        strength_cfg = self.config.get('memory_strength', {})
        purge_threshold = strength_cfg.get('purge_threshold', 0.01)
        # min_age_days = strength_cfg.get('purge_min_age_days', 60) # REMOVED
        # min_age_seconds = min_age_days * 24 * 3600 # REMOVED

        logger.warning(f"--- Purging Weak Nodes (Strength < {purge_threshold}) ---") # Removed age from log message
        # --- Tuning Log: Purge Start ---
        log_tuning_event("PURGE_START", {
            "personality": self.personality,
            "strength_threshold": purge_threshold,
            # "min_age_days": min_age_days, # REMOVED
        })

        purge_count = 0
        current_time = time.time()
        nodes_to_purge = []
        nodes_snapshot = list(self.graph.nodes(data=True)) # Snapshot

        for uuid, data in nodes_snapshot:
            current_strength = data.get('memory_strength', 1.0)
            # --- Purge only based on strength ---
            if current_strength < purge_threshold:
                nodes_to_purge.append(uuid)
                logger.debug(f"Marked weak node {uuid[:8]} for purging (Strength: {current_strength:.3f})")
            # else: pass # Node strength is above threshold

        # Perform deletion
        if nodes_to_purge:
            logger.info(f"Attempting to permanently purge {len(nodes_to_purge)} weak nodes...")
            for uuid in list(nodes_to_purge): # Iterate over copy
                if self.delete_memory_entry(uuid): # delete_memory_entry handles graph, embed, index, map, rebuild
                    purge_count += 1
                else:
                    logger.error(f"Failed to purge weak node {uuid[:8]}. It might have been deleted already.")

            logger.info(f"--- Purge Complete: {purge_count} nodes permanently deleted. ---")
            # delete_memory_entry already rebuilds/saves if successful.
        else:
            logger.info("--- Purge Complete: No weak nodes met the criteria for purging. ---")

        # --- Tuning Log: Purge End ---
        log_tuning_event("PURGE_END", {
            "personality": self.personality,
            "purged_count": purge_count,
            "purged_uuids": nodes_to_purge, # Log UUIDs that were targeted
        })

    # --- Prompting and LLM Interaction ---
    # (Keep _construct_prompt and _call_kobold_api from previous version)
    def _construct_prompt(self, user_input: str, conversation_history: list, memory_chain: list, tokenizer, max_context_tokens: int) -> str:
        """Constructs the prompt for the LLM, incorporating time, memory, history."""
        # Use the correct logger name: 'logger'
        logger.debug(f"_construct_prompt received user_input: '{user_input}'") # Corrected logger name

        if tokenizer is None:
            logger.error("Tokenizer unavailable for prompt construction.") # Corrected logger name
            return f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

        # --- Time formatting ---
        time_str = "[Current time unavailable]"
        try:
            localtz = ZoneInfo("Europe/Berlin") if ZoneInfo else timezone.utc
            now = datetime.now(localtz)
            time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except Exception as e:
            logger.warning(f"Could not get/format local time: {e}") # Corrected logger name

        # --- Gemma Instruct format tags ---
        start_turn, end_turn = "<start_of_turn>", "<end_of_turn>"
        user_tag, model_tag = f"{start_turn}user\n", f"{start_turn}model\n"

        # --- Format CURRENT user input ---
        user_input_fmt = f"{user_tag}{user_input}{end_turn}\n"
        logger.debug(f"Formatted current user input (user_input_fmt): '{user_input_fmt[:150]}...'") # Corrected logger name

        final_model_tag = f"{model_tag}"
        time_info_block = f"{model_tag}Current time is {time_str}.{end_turn}\n"
        asm_block = "" # Initialize ASM block

        # --- Format Structured ASM Block ---
        if self.autobiographical_model:
            try:
                # Format the structured data into a readable block
                asm_parts = ["[My Self-Perception:]"]
                if self.autobiographical_model.get("summary_statement"):
                    asm_parts.append(f"- Summary: {self.autobiographical_model['summary_statement']}")
                if self.autobiographical_model.get("core_traits"):
                    asm_parts.append(f"- Traits: {', '.join(self.autobiographical_model['core_traits'])}")
                if self.autobiographical_model.get("recurring_themes"):
                    asm_parts.append(f"- Often Discuss: {', '.join(self.autobiographical_model['recurring_themes'])}")
                if self.autobiographical_model.get("values_beliefs"):
                    asm_parts.append(f"- Beliefs/Values: {', '.join(self.autobiographical_model['values_beliefs'])}")
                # Optionally add significant events if space allows or needed
                # if self.autobiographical_model.get("significant_events"):
                #     asm_parts.append(f"- Key Events: {'; '.join(self.autobiographical_model['significant_events'])}")

                if len(asm_parts) > 1: # Only add block if there's content beyond the header
                    asm_text = "\n".join(asm_parts)
                    asm_block = f"{model_tag}{asm_text}{end_turn}\n"
                    logger.debug("Structured ASM block created.")
                else:
                    logger.debug("ASM dictionary present but contained no usable fields.")
            except Exception as e:
                logger.error(f"Error formatting structured ASM for prompt: {e}", exc_info=True)
                asm_block = "" # Clear block on formatting error
        else:
            logger.debug("No ASM summary available to add to prompt.")

        # --- Token Budget Calculation ---
        prompt_cfg = self.config.get('prompting', {})
        context_headroom = prompt_cfg.get('context_headroom', 250)
        mem_budget_ratio = prompt_cfg.get('memory_budget_ratio', 0.45)
        hist_budget_ratio = prompt_cfg.get('history_budget_ratio', 0.55)
        # Add the system note to fixed parts
        system_note_block = f"{model_tag}[System Note: Pay close attention to the sequence and relative timing ('X minutes ago', 'yesterday', etc.) of the provided memories and conversation history to maintain context.]{end_turn}\n"
        try:
            fixed_tokens = (len(tokenizer.encode(time_info_block)) +
                            len(tokenizer.encode(system_note_block)) + # Add system note tokens
                            len(tokenizer.encode(asm_block)) + # Add ASM block tokens
                            len(tokenizer.encode(user_input_fmt)) +
                            len(tokenizer.encode(final_model_tag)))
        except Exception as e:
            logger.error(f"Tokenization error for fixed prompt parts (incl. ASM/System Note): {e}") # Corrected logger name
            fixed_tokens = len(time_info_block) + len(user_input_fmt) + len(final_model_tag)
            logger.warning("Using character count proxy for fixed tokens.") # Corrected logger name

        avail_budget = max_context_tokens - fixed_tokens - context_headroom
        logger.debug(f"Token counts: Max={max_context_tokens}, Fixed={fixed_tokens}, Headroom={context_headroom}, Available={avail_budget}") # Corrected logger name

        if avail_budget <= 0:
            logger.warning(f"Low token budget ({avail_budget}). Only including current input and time.") # Corrected logger name
            final_prompt = time_info_block + user_input_fmt + final_model_tag
            logger.debug(f"Final Prompt (Low Budget): '{final_prompt[:150]}...'") # Corrected logger name
            return final_prompt

        mem_budget = int(avail_budget * mem_budget_ratio)
        hist_budget = avail_budget - mem_budget
        logger.debug(f"Token Budget Allocation: Memory={mem_budget}, History={hist_budget}") # Corrected logger name
        # --- Tuning Log: Prompt Budgeting ---
        # Note: interaction_id not available here
        log_tuning_event("PROMPT_BUDGETING", {
            "personality": self.personality,
            "user_input_preview": user_input[:100],
            "max_context_tokens": max_context_tokens,
            "fixed_tokens": fixed_tokens,
            "headroom": context_headroom,
            "available_budget": avail_budget,
            "memory_budget": mem_budget,
            "history_budget": hist_budget,
        })

        # --- Memory Context Construction ---
        # (This part remains the same - assumes it uses 'logger' correctly if needed internally)
        mem_ctx_str = ""
        cur_mem_tokens = 0
        mem_header = "---\n[Relevant Past Information - Use this to recall facts (like names) and context]:\n"
        mem_footer = "\n---"
        mem_placeholder_no_mem = "[No relevant memories found or fit budget]"
        mem_placeholder_too_long = "[Memory Omitted Due To Length]"
        mem_placeholder_error = "[Memory Error Processing Context]"
        mem_content = mem_placeholder_no_mem
        included_mem_uuids = []

        if memory_chain and mem_budget > 0:
            # Sort by timestamp before processing for budget
            mem_chain_sorted = sorted(memory_chain, key=lambda x: x.get('timestamp', ''))
            mem_parts = []
            tmp_tokens = 0
            try:
                # Estimate tokens for header/footer/tags
                format_tokens = len(tokenizer.encode(f"{model_tag}{mem_header}{mem_footer}{end_turn}\n"))
            except Exception:
                 format_tokens = 50 # Rough estimate if tokenization fails
            effective_mem_budget = mem_budget - format_tokens

            for node in mem_chain_sorted:
                spk = node.get('speaker','?')
                txt = node.get('text','')
                ts = node.get('timestamp','')
                # --- Use new helper for relative time ---
                relative_time_desc = self._get_relative_time_desc(ts)

                fmt_mem = f"{spk} ({relative_time_desc}): {txt}\n"
                try:
                    mem_tok_len = len(tokenizer.encode(fmt_mem))
                except Exception as e:
                    logger.warning(f"Tokenization error for memory item: {e}. Skipping memory item.") # Corrected logger name
                    continue

                if tmp_tokens + mem_tok_len <= effective_mem_budget:
                    mem_parts.append(fmt_mem)
                    tmp_tokens += mem_tok_len
                    included_mem_uuids.append(node['uuid'][:8])
                else:
                    logger.debug("Memory budget reached during context construction.") # Corrected logger name
                    break # Stop adding memories

            if mem_parts:
                mem_content = mem_header + "".join(mem_parts) + mem_footer

        # Format the final memory block (or placeholder)
        if mem_content != mem_placeholder_no_mem:
             full_mem_block = f"{model_tag}{mem_content}{end_turn}\n"
        else:
             full_mem_block = f"{model_tag}{mem_placeholder_no_mem}{end_turn}\n"

        try:
            mem_block_tok_len = len(tokenizer.encode(full_mem_block))
            if mem_block_tok_len <= mem_budget:
                mem_ctx_str = full_mem_block
                cur_mem_tokens = mem_block_tok_len
                logger.debug(f"Included memory block ({cur_mem_tokens} tokens). UUIDs (chrono): {included_mem_uuids}") # Corrected logger name
            else:
                # This case should be less likely now due to pre-calculation, but handle anyway
                logger.warning(f"Formatted memory block ({mem_block_tok_len}) still exceeded budget ({mem_budget}). Using placeholder.") # Corrected logger name
                mem_ctx_str = f"{model_tag}{mem_placeholder_too_long}{end_turn}\n"
                cur_mem_tokens = len(tokenizer.encode(mem_ctx_str))
        except Exception as e:
            logger.error(f"Tokenization error for memory block: {e}. Using error placeholder.") # Corrected logger name
            mem_ctx_str = f"{model_tag}{mem_placeholder_error}{end_turn}\n"
            cur_mem_tokens = len(tokenizer.encode(mem_ctx_str))


        # --- History Context Construction ---
        hist_parts = []
        cur_hist_tokens = 0
        actual_hist_budget = avail_budget - cur_mem_tokens # History gets remaining budget
        logger.debug(f"Actual History Budget: {actual_hist_budget}") # Corrected logger name
        included_hist_count = 0

        history_to_process = conversation_history # Use history passed in

        if history_to_process and actual_hist_budget > 0:
            for turn in reversed(history_to_process):
                spk = turn.get('speaker', '?')
                txt = turn.get('text', '')
                logger.debug(f"Processing history turn: Speaker={spk}, Text='{txt[:80]}...'") # Corrected logger name

                if spk == 'User': fmt_turn = f"{user_tag}{txt}{end_turn}\n"
                elif spk in ['AI', 'System', 'Error']: fmt_turn = f"{model_tag}{txt}{end_turn}\n"
                else: logger.warning(f"Unknown speaker '{spk}' in history, skipping."); continue # Corrected logger name

                try: turn_tok_len = len(tokenizer.encode(fmt_turn))
                except Exception as e: logger.warning(f"Tokenization error for history turn: {e}. Skipping."); continue # Corrected logger name

                if cur_hist_tokens + turn_tok_len <= actual_hist_budget:
                    hist_parts.append(fmt_turn)
                    cur_hist_tokens += turn_tok_len
                    included_hist_count += 1
                else: logger.debug("History budget reached."); break # Corrected logger name

            hist_parts.reverse() # Chronological order
            logger.debug(f"Included history ({cur_hist_tokens} tokens / {included_hist_count} turns).") # Corrected logger name

        # --- Assemble Final Prompt ---
        final_parts = []
        final_parts.append(time_info_block)
        # Add instruction about temporal awareness AND action capability
        # --- System Instructions for AI ---
        system_instructions = [
            "[System Note: Pay close attention to the sequence and relative timing ('X minutes ago', 'yesterday', etc.) of the provided memories and conversation history to maintain context.]",
            # --- Action Capability Instructions ---
            "[System Note: You have the ability to manage files and calendar events.",
            "  To request an action, end your *entire* response with a special tag: `[ACTION: {\"action\": \"action_name\", \"args\": {\"arg1\": \"value1\", ...}}]`.",
            "  **Available Actions:** `create_file`, `append_file`, `list_files`, `read_file`, `delete_file`, `add_calendar_event`, `read_calendar`.",
            "  **CRITICAL: `edit_file` is NOT a valid action.**",
            "  **To Edit a File:** Use `read_file` then `create_file` (overwrites).",
            "  **Using Actions:**",
            "    - For `list_files`, `read_calendar`: Use `[ACTION: {\"action\": \"action_name\", \"args\": {}}]` (or add optional 'date' arg for read_calendar).",
            "    - For `read_file`, `delete_file`: Use `[ACTION: {\"action\": \"action_name\", \"args\": {\"filename\": \"target_file.txt\"}}]`.",
            "    - For `append_file`: Use `[ACTION: {\"action\": \"append_file\", \"args\": {\"filename\": \"target_file.txt\", \"content\": \"Text to append...\"}}]` (Generate the actual content to append).",
            "    - For `add_calendar_event`: Use `[ACTION: {\"action\": \"add_calendar_event\", \"args\": {\"date\": \"YYYY-MM-DD\", \"time\": \"HH:MM\", \"description\": \"Event details...\"}}]`.",
            "    - **For `create_file`:** Signal your *intent* by providing a brief description. The system will handle filename/content generation separately. Use `[ACTION: {\"action\": \"create_file\", \"args\": {\"description\": \"Brief description of what to save, e.g., 'List of project ideas'\"}}]`.",
            "  Only use the ACTION tag if you decide an action is necessary based on the context.]",
            # --- NEW: Instruction for handling retrieved intentions ---
            "[System Note: If you see a retrieved memory starting with 'Remember:', check if the trigger condition seems relevant to the current conversation. If so, incorporate the reminder into your response or perform the implied task if appropriate (potentially using the ACTION tag).]"
        ]
        # Add each instruction line as a separate system turn for clarity
        for instruction in system_instructions:
             final_parts.append(f"{model_tag}{instruction}{end_turn}\n")

        # --- Format Drive State Block ---
        drive_block = ""
        if self.config.get('subconscious_drives', {}).get('enabled', False) and self.drive_state:
            try:
                drive_parts = ["[Current Drive State (Internal Motivations):]"]
                # Format short-term drives (relative to baseline)
                st_drives = self.drive_state.get("short_term", {})
                lt_drives = self.drive_state.get("long_term", {})
                base_drives = self.config.get('subconscious_drives', {}).get('base_drives', {})
                lt_influence = self.config.get('subconscious_drives', {}).get('long_term_influence_on_baseline', 1.0)

                for drive, st_level in st_drives.items():
                    config_baseline = base_drives.get(drive, 0.0)
                    lt_level = lt_drives.get(drive, 0.0)
                    dynamic_baseline = config_baseline + (lt_level * lt_influence)
                    deviation = st_level - dynamic_baseline
                    # Describe the state qualitatively
                    state_desc = "Neutral"
                    if deviation > 0.2: state_desc = "High" # Need potentially met/overshot
                    elif deviation < -0.2: state_desc = "Low" # Need potentially unmet
                    drive_parts.append(f"- {drive}: {state_desc} (Level: {st_level:.2f}, Baseline: {dynamic_baseline:.2f})")

                # Optionally add long-term summary if needed/space allows
                # drive_parts.append("[Long-Term Tendencies:]")
                # for drive, lt_level in lt_drives.items():
                #     drive_parts.append(f"- {drive}: {lt_level:.2f}")

                if len(drive_parts) > 1:
                    drive_text = "\n".join(drive_parts)
                    drive_block = f"{model_tag}{drive_text}{end_turn}\n"
                    logger.debug("Formatted drive state block created.")
            except Exception as e:
                logger.error(f"Error formatting drive state for prompt: {e}", exc_info=True)
                drive_block = ""

        if asm_block: final_parts.append(asm_block) # Add ASM block after time/system note
        # --- Drive block is intentionally NOT added to final_parts to keep it subconscious ---
        # if drive_block: final_parts.append(drive_block) # Add Drive block after ASM
        if mem_ctx_str: final_parts.append(mem_ctx_str)
        final_parts.extend(hist_parts)
        final_parts.append(user_input_fmt) # Crucial: Adds current user input (potentially with tag)
        final_parts.append(final_model_tag)
        final_prompt = "".join(final_parts)

        # --- Final Logging and Checks ---
        logger.debug(f"--- Final Prompt Structure ---") # Corrected logger name
        if len(final_prompt) > 500: logger.debug(f"{final_prompt[:250]}...\n...\n...{final_prompt[-250:]}") # Corrected logger name
        else: logger.debug(final_prompt) # Corrected logger name
        logger.debug(f"--- End Final Prompt Structure ---") # Corrected logger name

        try:
            final_tok_count = len(tokenizer.encode(final_prompt))
            logger.info(f"Constructed prompt final token count: {final_tok_count} (Budget Available: {avail_budget})") # Corrected logger name
            if final_tok_count > max_context_tokens:
                 logger.error(f"CRITICAL: Final prompt ({final_tok_count} tokens) EXCEEDS max context ({max_context_tokens}).") # Corrected logger name
            elif final_tok_count > max_context_tokens - context_headroom:
                 logger.warning(f"Final prompt ({final_tok_count} tokens) close to max context ({max_context_tokens}). Less headroom ({max_context_tokens-final_tok_count}).") # Corrected logger name
        except Exception as e:
            logger.error(f"Tokenization error for final prompt: {e}") # Corrected logger name
            final_tok_count = -1 # Indicate error

        # --- Tuning Log: Prompt Construction Result ---
        log_tuning_event("PROMPT_CONSTRUCTION_RESULT", {
            "personality": self.personality,
            "user_input_preview": user_input[:100],
            "included_memory_uuids": included_mem_uuids, # From memory construction block
            "included_history_turns": included_hist_count, # From history construction block
            "final_token_count": final_tok_count,
            "max_context_tokens": max_context_tokens,
            "prompt_preview_start": final_prompt[:200],
            "prompt_preview_end": final_prompt[-200:],
        })

        return final_prompt


    # --- Updated signature to accept parameters ---
    def _call_kobold_api(self, prompt: str, model_name: str, max_length: int, temperature: float, top_p: float, top_k: int, min_p: float) -> str:
        """
        Sends prompt to KoboldCpp Generate API, returns generated text.
        Parameters are passed in, typically from _call_configured_llm.
        """
        logger.debug(f"_call_kobold_api received prompt ('{prompt[:80]}...'). Length: {len(prompt)}")
        logger.debug(f"  Params: max_len={max_length}, temp={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}")

        # Calculate max_context_length
        try:
            # Estimate tokens using the tokenizer if available
            prompt_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else len(prompt) // 3
        except Exception as e:
            logger.warning(f"Tokenizer error calculating prompt tokens: {e}. Estimating based on length.")
            prompt_tokens = len(prompt) // 3 # Fallback estimation

        # Get model's max context from config, default if not set
        model_max_ctx = self.config.get('prompting',{}).get('max_context_tokens', 4096)

        # Calculate desired total tokens and clamp to model max context
        # Add a small buffer (e.g., 50) for safety/overhead
        desired_total_tokens = prompt_tokens + max_length + 50
        max_ctx_len = min(model_max_ctx, desired_total_tokens)
        logger.debug(f"Prompt tokens: ~{prompt_tokens}. Max new: {max_length}. Max context length for API call: {max_ctx_len}")

        api_url = self.kobold_api_url
        if not api_url:
            logger.error("Kobold API URL is not configured.")
            return "Error: Kobold API URL not configured."

        # --- Construct Payload with New Defaults and Parameters ---
        payload = {
            'prompt': prompt,
            'max_context_length': max_ctx_len, # Ensure context doesn't exceed model limits
            'max_length': max_length, # Max tokens to generate
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k, # Added top_k
            'min_p': min_p, # Added min_p
            # Common Kobold parameters (adjust if needed for your backend)
            'rep_pen': 1.08, # Example repetition penalty, adjust as needed
            'rep_pen_range': 2048,
            'stop_sequence': ["<end_of_turn>", "<start_of_turn>user\n", "User:", "\nUser:", "\nUSER:"], # Added \nUSER:
            # Ensure these flags are set as desired (usually false for raw completion)
            'use_memory': False,
            'use_story': False,
            'use_authors_note': False,
            'use_world_info': False,
            # 'model': model_name, # KoboldCpp generate API often ignores this, but include if needed
        }

        # Log payload (masking potentially long prompt)
        log_payload = payload.copy()
        log_payload['prompt'] = log_payload['prompt'][:100] + ("..." if len(log_payload['prompt']) > 100 else "")
        logger.debug(f"Payload sent to Kobold API ({api_url}): {log_payload}")

        try:
            response = requests.post(api_url, json=payload, timeout=180) # 3-minute timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # Extract generated text (structure might vary slightly between Kobold versions)
            gen_txt = result.get('results', [{}])[0].get('text', '').strip()

            # Remove stop sequences from the end of the generation
            # Iterate carefully to avoid removing parts of valid sequences
            cleaned_txt = gen_txt
            for seq in sorted(payload['stop_sequence'], key=len, reverse=True): # Check longer sequences first
                 if cleaned_txt.endswith(seq):
                      cleaned_txt = cleaned_txt[:-len(seq)].rstrip()
                      break # Stop after removing the first found sequence

            if not cleaned_txt:
                 logger.warning("Kobold API returned empty text after stripping stop sequences.")
            else:
                 logger.debug(f"Kobold API cleaned response text: '{cleaned_txt[:100]}...'")

            return cleaned_txt

        except requests.exceptions.Timeout:
            logger.error(f"Kobold API call timed out after 180 seconds ({api_url}).")
            return f"Error: Kobold API call timed out."
        except requests.exceptions.RequestException as e:
            logger.error(f"Kobold API connection/request error: {e}", exc_info=True)
            # Provide more specific error if possible (e.g., connection error vs. HTTP error)
            status_code = getattr(e.response, 'status_code', 'N/A')
            return f"Error: Could not connect or communicate with Kobold API at {api_url}. Status: {status_code}. Details: {e}"
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON response from Kobold API: {e}. Response text: '{response.text[:500]}...'")
             return "Error: Failed to decode JSON response from Kobold API."
        except Exception as e:
            logger.error(f"Kobold API call unexpected error: {e}", exc_info=True)
            return f"Error: Unexpected issue during Kobold API call."

    # --- Memory Modification & Action Analysis ---

    # *** NEW: Analyze for File/Calendar/Other Actions ***
    def analyze_action_request(self, request_text: str) -> dict:
        """
        Uses LLM to detect non-memory action intents (e.g., file, calendar)
        and extract arguments. Returns structured action data or {"action": "none"}.
        (V4 Prompt: Same prompt, improved extraction and increased max_length)
        """
        logger.info(f"Analyzing for action request: '{request_text[:100]}...'")
        # Define available tools and their required arguments
        tools = {
            "create_file": ["filename", "content"],
            "append_file": ["filename", "content"],
            "list_files": [], # No arguments required
            "read_file": ["filename"],
            "delete_file": ["filename"],
            "add_calendar_event": ["date", "time", "description"],
            "read_calendar": [] # 'date' is optional here
        }
        # Tool descriptions are now loaded from the prompt file, no need to define here

        prompt_template = self._load_prompt("action_analysis_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load action analysis prompt template. Cannot analyze action.")
            return {'action': 'error', 'reason': 'Action analysis prompt template missing.'}

        # No need to format tool_descriptions here, prompt file handles it
        full_prompt = prompt_template.format(
            # tool_descriptions=tool_descriptions, # Removed
            request_text=request_text
        )

        logger.debug(f"Sending action analysis prompt (from file):\n{full_prompt}")
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('action_analysis', prompt=full_prompt)

        # --- Check for API call errors BEFORE parsing ---
        if not llm_response_str or llm_response_str.startswith("Error:"):
            error_reason = llm_response_str if llm_response_str else "LLM call failed (empty response)"
            return {'action': 'error', 'reason': 'Action analysis prompt template missing.'}

        # The prompt template now contains the descriptions directly.
        full_prompt = prompt_template.format(
            request_text=request_text
        )

        logger.debug(f"Sending action analysis prompt (from file):\n{full_prompt}")
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('action_analysis', prompt=full_prompt)

        # --- Check for API call errors BEFORE parsing ---
        if not llm_response_str or llm_response_str.startswith("Error:"):
            error_reason = llm_response_str if llm_response_str else "LLM call failed (empty response)"
            logger.error(f"Action analysis failed due to LLM API error: {error_reason}")
            return {'action': 'error', 'reason': error_reason}

        parsed_result = None
        json_str = "" # Initialize json_str for potential use in error logging
        try:
            logger.debug(f"Raw action analysis response: ```{llm_response_str}```")
            # --- Improved JSON Extraction ---
            # Attempt to find the outermost JSON object {} or list []
            # This handles cases where the LLM might add explanations before/after
            match = re.search(r'(\{.*\}|\[.*\])', llm_response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                logger.debug(f"Extracted potential JSON string using regex: {json_str}")
                parsed_result = json.loads(json_str)
            else:
                # Fallback: try cleaning markdown fences if regex fails
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

                start_brace = cleaned_response.find('{')
                end_brace = cleaned_response.rfind('}')
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    json_str = cleaned_response[start_brace:end_brace + 1]
                    logger.debug(f"Extracted JSON string using brace finding (fallback): {json_str}")
                    parsed_result = json.loads(json_str)
                else:
                    logger.error(f"Could not find valid JSON object in LLM response. Raw: '{llm_response_str}'")
                    return {'action': 'error', 'reason': 'Could not extract valid JSON object from LLM response.',
                            'raw_response': llm_response_str}

            # --- Validation ---
            if not isinstance(parsed_result, dict):
                 raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_result)}).")

            logger.info(f"LLM Parsed Action: {parsed_result}")
            action = parsed_result.get("action")
            if not action or not isinstance(action, str):
                raise ValueError("Missing or invalid 'action' key (must be a string) in JSON response.")

            valid_actions = ["none", "clarify", "error"] + list(tools.keys())
            if action not in valid_actions:
                logger.warning(f"LLM returned unknown action '{action}'. Treating as 'none'. Raw: {llm_response_str}")
                return {"action": "none"} # Treat unknown as none

            if action == "none": return {"action": "none"}
            if action == "error": return parsed_result # Pass through LLM-reported error

            # --- Argument Validation ---
            args = parsed_result.get("args", {})
            if not isinstance(args, dict):
                raise ValueError(f"Invalid 'args' format for action '{action}'. Expected a dictionary, got {type(args)}.")

            if action == "clarify":
                # Validate clarify structure
                missing_args = parsed_result.get("missing_args")
                original_action = parsed_result.get("original_action")
                if not isinstance(missing_args, list) or not all(isinstance(item, str) for item in missing_args):
                    raise ValueError("Clarify action missing or invalid 'missing_args' (must be a list of strings).")
                if not isinstance(original_action, str):
                    raise ValueError("Clarify action missing or invalid 'original_action' (must be a string).")
                if original_action not in tools:
                    raise ValueError(f"Clarify action refers to an invalid original_action '{original_action}'.")
                logger.info(f"Clarification requested for '{original_action}', missing: {missing_args}")
                return parsed_result # Return valid clarify request

            # --- Validate Required Args for Specific Actions ---
            required_args = tools.get(action, [])
            missing = []
            validated_args = {} # Store validated/sanitized args

            for req_arg in required_args:
                arg_value = args.get(req_arg)
                # Check if missing, None, or empty string
                if arg_value is None or (isinstance(arg_value, str) and not arg_value.strip()):
                    # Special case: 'date' is optional for 'read_calendar'
                    if not (action == "read_calendar" and req_arg == "date"):
                        missing.append(req_arg)
                else:
                    # Basic type validation/conversion (ensure strings)
                    if not isinstance(arg_value, str):
                         logger.warning(f"Argument '{req_arg}' for action '{action}' is not a string (type: {type(arg_value)}). Converting.")
                         validated_args[req_arg] = str(arg_value)
                    else:
                         validated_args[req_arg] = arg_value.strip() # Store stripped string

            if missing:
                logger.warning(f"Action '{action}' identified, but missing required args: {missing}. Requesting clarification.")
                # Return a well-formed clarify request
                return {"action": "clarify", "missing_args": missing, "original_action": action}

            # --- Sanitize Filename ---
            if "filename" in validated_args:
                original_filename = validated_args["filename"]
                safe_filename = os.path.basename(original_filename) # Removes path components
                # Additional checks for safety
                if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip() or '/' in safe_filename or '\\' in safe_filename:
                    logger.error(f"Invalid or unsafe filename extracted after sanitization: '{original_filename}' -> '{safe_filename}'")
                    return {'action': 'error', 'reason': f"Invalid or potentially unsafe filename provided: '{original_filename}'",
                            'raw_response': llm_response_str, 'parsed': parsed_result}
                validated_args["filename"] = safe_filename # Update with sanitized name
                logger.debug(f"Sanitized filename: '{original_filename}' -> '{safe_filename}'")

            # --- Success ---
            logger.info(f"Action analysis successful: Action='{action}', Args={validated_args}")
            return {"action": action, "args": validated_args} # Return validated args

        except json.JSONDecodeError as e:
            logger.error(f"LLM Action Parse Error (JSONDecodeError): {e}. Extracted String: '{json_str}'. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM response JSON parsing failed: {e}', 'raw_response': llm_response_str}
        except ValueError as e:
            # Catch validation errors raised above
            logger.error(f"LLM Action Validation Error: {e}. Parsed JSON: {parsed_result}. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM response validation failed: {e}', 'raw_response': llm_response_str, 'parsed': parsed_result}
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error parsing/validating action response: {e}", exc_info=True)
            return {'action': 'error', 'reason': f'Unexpected action parsing error: {e}', 'raw_response': llm_response_str}

    # (Keep Example Usage Block unchanged)
    #if __name__ == "__main__":
        # ...
        #logger.info("Basic test finished.")

    def analyze_memory_modification_request(self, request: str) -> dict:
        """Analyzes user request for **memory modification only** (delete, edit, forget)."""
        logger.info(f"Analyzing *memory modification* request: '{request[:100]}...'")
        uuid_pattern = r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'
        found_uuids = re.findall(uuid_pattern, request, re.IGNORECASE)
        target_uuid = found_uuids[0] if found_uuids else None
        request_lower = request.lower()
        detected_action = None

        # Direct keyword/UUID extraction (simple cases)
        mod_keywords_cfg = self.config.get('modification_keywords', [])
        if any(kw in request_lower for kw in mod_keywords_cfg):
            if any(kw in request_lower for kw in ['delete', 'remove', 'forget']):
                detected_action = 'delete' # Treat forget as delete if UUID present
            elif any(kw in request_lower for kw in ['edit', 'change', 'correct', 'update']):
                detected_action = 'edit'

        if detected_action and target_uuid:
            logger.info(f"Direct extract: Action={detected_action}, UUID={target_uuid}")
            result = {'action': detected_action, 'target_uuid': target_uuid}
            if detected_action == 'edit':
                # Try to extract text following the UUID
                parts = request.split(target_uuid)
                new_text = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                if new_text:
                    # Simple prefix removal
                    for prefix in ["to ", "say ", "is ", ":", "-"]:
                        if new_text.lower().startswith(prefix):
                            new_text = new_text[len(prefix):].strip()
                    result['new_text'] = new_text
                    logger.info(f"Extracted new text: {new_text[:50]}...")
                else:
                    logger.warning("Edit UUID found, but no new text extracted.")
                    result['new_text'] = None # Explicitly set to None
            return result # Return early if direct extraction successful

        # Fallback to LLM analysis
        logger.info("Falling back to LLM analysis for memory mod request.")
        prompt_template = self._load_prompt("memory_mod_prompt.txt")
        if not prompt_template:
             logger.error("Failed to load memory modification prompt template. Cannot analyze request.")
             return {'action': 'error', 'reason': 'Memory modification prompt template missing.'}

        full_prompt = prompt_template.format(request_text=request)

        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('memory_modification_analysis', prompt=full_prompt)
        if not llm_response_str or llm_response_str.startswith("Error:"): # Check for helper errors too
            return {'action': 'error', 'reason': llm_response_str or 'LLM call failed'}

        try:
            # Clean potential markdown fences before parsing
            cleaned_response = llm_response_str.strip()
            if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
            if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
            if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

            # Find first '{' and last '}'
            start_brace = cleaned_response.find('{')
            end_brace = cleaned_response.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str = cleaned_response[start_brace:end_brace + 1]
                parsed_response = json.loads(json_str)
            else:
                 raise ValueError("No valid JSON object found in LLM response.")

            logger.info(f"LLM Parsed Mod: {parsed_response}")

            # Basic validation
            if 'action' not in parsed_response or parsed_response['action'] not in ['delete','edit','forget','none','error']:
                raise ValueError("Missing/Invalid action in LLM response")

            # If regex found a UUID but LLM didn't, add it back for delete/edit
            if target_uuid and 'target_uuid' not in parsed_response and parsed_response['action'] in ['delete', 'edit']:
                logger.info(f"Adding regex UUID {target_uuid} to LLM result.")
                parsed_response['target_uuid'] = target_uuid
                parsed_response.pop('target', None) # Remove text target if UUID is present

            return parsed_response
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM Mod Parse/Validation Error: {e}. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM Parse/Validation Fail: {e}', 'raw_response': llm_response_str}
        except Exception as e:
             logger.error(f"Unexpected error parsing memory mod response: {e}", exc_info=True)
             return {'action': 'error', 'reason': f'Unexpected error: {e}', 'raw_response': llm_response_str}

    def _classify_query_type(self, query_text: str) -> str:
        """Uses LLM to classify query as 'episodic', 'semantic', or 'other'."""
        logger.debug(f"Classifying query type for: '{query_text[:100]}...'")
        prompt_template = self._load_prompt("query_type_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load query type prompt template. Defaulting to 'other'.")
            return "other"

        full_prompt = prompt_template.format(query_text=query_text)
        # --- Use configured LLM call ---
        llm_response = self._call_configured_llm('query_type_classification', prompt=full_prompt)
        classification = llm_response.strip().lower()

        if llm_response.startswith("Error:"):
             logger.error(f"Query classification failed: {llm_response}")
             return "other" # Default on error
        elif classification in ["episodic", "semantic", "other"]:
            logger.info(f"Query classified as: {classification}")
            return classification
        else:
            logger.warning(f"LLM returned unexpected query classification '{classification}'. Defaulting to 'other'.")
            return "other"

    def _analyze_intention_request(self, request_text: str) -> dict:
        """Uses LLM to detect if user wants AI to remember something for later."""
        logger.debug(f"Analyzing for intention request: '{request_text[:100]}...'")
        prompt_template = self._load_prompt("intention_analysis_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load intention analysis prompt template.")
            return {'action': 'error', 'reason': 'Intention analysis prompt template missing.'}

        full_prompt = prompt_template.format(request_text=request_text)
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('intention_analysis', prompt=full_prompt)

        if not llm_response_str or llm_response_str.startswith("Error:"):
            error_reason = llm_response_str or "LLM call failed (empty response)"
            logger.error(f"Intention analysis failed: {error_reason}")
            return {'action': 'error', 'reason': error_reason}

        parsed_result = None
        json_str = ""
        try:
            logger.debug(f"Raw intention analysis response: ```{llm_response_str}```")
            match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed_result = json.loads(json_str)
            else: # Fallback cleaning
                cleaned_response = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                start_brace = cleaned_response.find('{'); end_brace = cleaned_response.rfind('}')
                if start_brace != -1 and end_brace != -1: json_str = cleaned_response[start_brace:end_brace+1]; parsed_result = json.loads(json_str)
                else: raise ValueError("No JSON object found")

            if not isinstance(parsed_result, dict): raise ValueError("Parsed JSON is not a dictionary.")
            action = parsed_result.get("action")
            if action == "store_intention":
                content = parsed_result.get("content")
                trigger = parsed_result.get("trigger")
                if not content or not trigger or not isinstance(content, str) or not isinstance(trigger, str):
                    raise ValueError("Missing or invalid 'content' or 'trigger' for store_intention.")
                logger.info(f"Intention detected: Content='{content[:50]}...', Trigger='{trigger}'")
                return parsed_result # Return the valid intention data
            elif action == "none":
                logger.debug("No intention detected.")
                return {"action": "none"}
            else:
                logger.warning(f"LLM returned unexpected action '{action}' for intention analysis.")
                return {"action": "none"} # Treat unexpected as none

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Intention Parse/Validation Error: {e}. Extracted: '{json_str}'. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM response parse/validation failed: {e}', 'raw_response': llm_response_str}
        except Exception as e:
            logger.error(f"Unexpected error parsing intention response: {e}", exc_info=True)
            return {'action': 'error', 'reason': f'Unexpected intention parsing error: {e}', 'raw_response': llm_response_str}


    # *** NEW: Action Dispatcher ***
    def execute_action(self, action_data: dict) -> tuple[bool, str, str]:
        """Executes a validated action based on the action_data dictionary."""
        action = action_data.get("action")
        args = action_data.get("args", {})
        # *** ADDED: Log arguments before execution ***
        logger.debug(f"Attempting to execute action '{action}' with args: {args}")
        success = False; message = f"Action '{action}' failed."; action_suffix = f"{action}_fail"
        # --- Initialize variables for potential return values ---
        file_list = None
        file_content = None

        try:
            # --- File Actions ---
            if action == "create_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content is not None:
                    # --- Check if file exists BEFORE calling create/overwrite ---
                    workspace_path = file_manager.get_workspace_path(self.config, self.personality)
                    if workspace_path:
                        file_path = os.path.join(workspace_path, filename)
                        if os.path.exists(file_path):
                            logger.warning(f"File '{filename}' exists. Requesting overwrite confirmation.")
                            # Return dict to signal confirmation needed
                            return {"action": "confirm_overwrite", "args": {"filename": filename, "content": content}}
                        else:
                            # File doesn't exist, proceed with creation
                            logger.debug(f"File '{filename}' does not exist. Proceeding with creation.")
                            success, message = file_manager.create_or_overwrite_file(self.config, self.personality, filename, str(content))
                    else:
                        message = f"Error: Could not access workspace for personality '{self.personality}'."
                        success = False
                else:
                    message = "Error: Missing filename or content for create_file."
                    success = False
            elif action == "append_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content:
                    logger.debug(f"Calling file_manager.append_to_file(config, personality='{self.personality}', filename='{filename}', content='{str(content)[:50]}...')")
                    success = file_manager.append_to_file(self.config, self.personality, filename, str(content)) # Ensure content is string
                    message = f"Appended to '{filename}'." if success else f"Failed append to '{filename}'."
                else: message = "Missing filename or content for append_file."
            elif action == "add_calendar_event":
                date, time_str, desc = args.get("date"), args.get("time"), args.get("description")
                if date and time_str and desc:
                    logger.debug(f"Calling file_manager.add_calendar_event(config, personality='{self.personality}', date='{date}', time='{time_str}', desc='{desc}')")
                    # TODO: Date parsing
                    parsed_date_str = date # Use extracted date directly for now
                    success = file_manager.add_calendar_event(self.config, self.personality, parsed_date_str, time_str, desc)
                    message = f"Event '{desc}' added for {parsed_date_str} at {time_str}." if success else f"Failed adding event '{desc}'."
                else: message = "Missing date/time/description."
            elif action == "read_calendar":
                date = args.get("date") # Optional
                # file_manager.read_calendar_events now returns (list[dict], str)
                events, fm_message = file_manager.read_calendar_events(self.config, self.personality, date)
                # For reading, success is true if the read operation itself didn't fail,
                # even if no events were found. The message conveys the outcome.
                success = True # Assume read operation itself succeeded unless exception below
                action_suffix = "read_calendar_success" # Use success suffix
                date_str = f" for {date}" if date else " (all dates)"

                # Construct user-facing message based on results
                if fm_message.startswith("IO error") or fm_message.startswith("Permission denied") or fm_message.startswith("Unexpected error"):
                    success = False # Override success if file_manager reported read error
                    action_suffix = "read_calendar_fail"
                    message = f"Error reading calendar: {fm_message}" # Pass file_manager error message
                elif not events:
                    # Message could be "not found" or "found 0 events" - use fm_message
                    message = fm_message
                else:
                    event_lines = []
                    for e in sorted(events, key=lambda x: (x.get('event_date', ''), x.get('event_time', ''))): # Sort events
                        event_lines.append(f"- {e.get('event_time', '?')}: {e.get('description', '?')} ({e.get('event_date', '?')})")
                    message = f"Found {len(events)} event(s){date_str}:\n" + "\n".join(event_lines)

            # --- NEW File Actions ---
            elif action == "list_files":
                file_list, message = self.list_files_wrapper()
                if file_list is not None:
                    success = True
                    # Format the message for the user
                    if file_list: message = f"Files in workspace:\n- " + "\n- ".join(file_list)
                    else: message = "Workspace is empty."
                else: # Error occurred
                    success = False
                    # message already contains the error from file_manager

            elif action == "read_file":
                filename = args.get("filename")
                if filename:
                    file_content, message = self.read_file_wrapper(filename)
                    if file_content is not None:
                        success = True
                        # Truncate long content for the message?
                        content_preview = file_content[:500] + ('...' if len(file_content) > 500 else '')
                        message = f"Content of '{filename}':\n---\n{content_preview}\n---"
                    else: # Error occurred
                        success = False
                        # message already contains the error from file_manager
                else:
                    message = "Error: Missing filename for read_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            elif action == "delete_file":
                filename = args.get("filename")
                if filename:
                    # Add confirmation step? For now, execute directly.
                    # Consider adding a config flag for delete confirmation later.
                    logger.warning(f"Executing delete_file action for: {filename}")
                    success, message = self.delete_file_wrapper(filename)
                else:
                    message = "Error: Missing filename for delete_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            # --- Unknown Action ---
            else:
                message = f"Error: The action '{action}' is not recognized or supported."
                logger.error(message + f" Action data: {action_data}")
                success = False
                action_suffix = "unknown_action_fail"

            # --- Update Suffix ---
            # Ensure suffix reflects success/failure determined within this block
            if action != "unknown_action_fail": # Don't override specific unknown suffix
                action_suffix = f"{action}_success"

        except Exception as e:
             # *** ADDED: Log the specific exception during execution ***
             logger.error(f"Exception during execution of action '{action}': {e}", exc_info=True)
             message = f"An internal error occurred while trying to perform action: {action}."
             success = False
             action_suffix = f"{action}_exception" # Specific suffix for exception

        logger.info(f"Action '{action}' result: Success={success}, Msg='{message[:100]}...'")
        return success, message, action_suffix

    # --- Main Interaction Loop ---
    # (Keep process_interaction from previous version - returns memory chain)

    def process_interaction(self, user_input: str, conversation_history: list, attachment_data: dict | None = None) -> tuple[str, list]:
        """Processes user input (text and optional image attachment), calls appropriate LLM API, updates memory."""
        logger.debug(f"PROCESS_INTERACTION START: Has embedder? {hasattr(self, 'embedder')}")
        # --- Tuning Log: Interaction Start ---
        interaction_id = str(uuid.uuid4()) # Unique ID for this interaction
        log_tuning_event("INTERACTION_START", {
            "interaction_id": interaction_id,
            "personality": self.personality,
            "user_input_preview": user_input[:100],
            "has_attachment": bool(attachment_data),
            "attachment_type": attachment_data.get('type') if attachment_data else None,
        })

        if not hasattr(self, 'embedder') or self.embedder is None:
             logger.error("PROCESS_INTERACTION ERROR: Cannot proceed without embedder!")
             # --- Tuning Log: Interaction Error ---
             log_tuning_event("INTERACTION_ERROR", {
                 "interaction_id": interaction_id,
                 "personality": self.personality,
                 "stage": "embedder_check",
                 "error": "Embedder not initialized",
             })
             return "Error: Backend embedder not initialized correctly.", []

        logger.info(f"Processing interaction (ID: {interaction_id[:8]}): Input='{user_input[:50]}...' Has Attachment: {bool(attachment_data)}")
        if attachment_data: logger.debug(f"Attachment details: type={attachment_data.get('type')}, filename={attachment_data.get('filename')}")

        # Initialize variables that might not be assigned in all paths
        ai_response = "Error: Processing failed." # Default error message
        parsed_response = "Error: Processing failed." # Default for return
        memory_chain_data = []
        graph_user_input = user_input # Start with original input for graph
        action_result_message = None # Initialize here

        try:
            # --- Handle Multimodal Input ---
            if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('data_url'):
                logger.info("Image attachment detected. Using Chat Completions API.")
                logger.info("Skipping standard memory retrieval for image prompt.")
                memory_chain_data = [] # No memory chain for multimodal yet

                # Prepare messages for Chat API
                messages = []
                history_limit = 5 # Limit history for multimodal context
                relevant_history = conversation_history[-history_limit:]
                for turn in relevant_history:
                    role = "user" if turn.get("speaker") == "User" else "assistant"
                    # Remove potential image placeholders from history text
                    text_content = re.sub(r'\s*\[Image:\s*.*?\s*\]\s*', '', turn.get("text","")).strip()
                    if text_content: messages.append({"role": role, "content": text_content})

                # Construct current user message (text + image)
                user_content = []
                if user_input: user_content.append({"type": "text", "text": user_input})
                user_content.append({"type": "image_url", "image_url": {"url": attachment_data['data_url']}})
                messages.append({"role": "user", "content": user_content})

                # Call Multimodal API using configured helper
                ai_response = self._call_configured_llm('main_chat_multimodal', messages=messages)

                # Prepare user input for graph storage (add placeholder)
                if attachment_data.get('filename'):
                    placeholder = f" [Image Attached: {attachment_data['filename']}]"
                    separator = " " if graph_user_input else ""
                    graph_user_input += separator + placeholder

            # --- Handle Text-Only Input ---
            else:
                logger.info("No valid image attachment. Using standard Generate API.")
                memory_chain_data = [] # Reset just in case

                # Concept finding logic moved to AFTER node creation

                # Only retrieve memory if it's not explicitly an image placeholder input
                # (This check might be redundant if multimodal handles all image cases now)
                if not user_input.strip().startswith("[Image:"):
                    # --- Classify Query Type ---
                    query_type = self._classify_query_type(user_input) # Classify before search
                    # --- Tuning Log: Query Classification ---
                    log_tuning_event("QUERY_CLASSIFICATION", {
                        "interaction_id": interaction_id,
                        "personality": self.personality,
                        "query_preview": user_input[:100],
                        "classified_type": query_type,
                    })

                    logger.info(f"Searching initial nodes (Query Type: {query_type})...")
                    max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7)
                    # Pass query_type to search function
                    initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes, query_type=query_type)
                    initial_uuids = [uid for uid, score in initial_nodes]
                    logger.info(f"Initial UUIDs: {initial_uuids}")
                    # --- Tuning Log: Initial Search Results ---
                    log_tuning_event("INITIAL_SEARCH_RESULT", {
                        "interaction_id": interaction_id,
                        "personality": self.personality,
                        "query_preview": user_input[:100],
                        "query_type": query_type,
                        "initial_node_scores": initial_nodes, # List of (uuid, score)
                        "initial_uuids_selected": initial_uuids,
                    })


                    if initial_uuids:
                        # --- Use context from PREVIOUS interaction for retrieval bias ---
                        concepts_for_retrieval = self.last_interaction_concept_uuids
                        mood_for_retrieval = self.last_interaction_mood
                        logger.info(f"Using previous interaction context for retrieval: Concepts={len(concepts_for_retrieval)}, Mood={mood_for_retrieval}")

                        # --- Get UUIDs of immediately preceding turns for priming ---
                        last_turn_uuids_for_priming = []
                        if conversation_history:
                            # Get the last turn (which should be the user's input just added)
                            last_user_turn_uuid = conversation_history[-1].get("uuid") # Assuming UUID is added to history dict
                            if last_user_turn_uuid: last_turn_uuids_for_priming.append(last_user_turn_uuid)
                            # Get the turn before that (likely the AI's previous response)
                            if len(conversation_history) > 1:
                                 prev_ai_turn_uuid = conversation_history[-2].get("uuid")
                                 if prev_ai_turn_uuid: last_turn_uuids_for_priming.append(prev_ai_turn_uuid)
                        logger.debug(f"Priming UUIDs identified: {last_turn_uuids_for_priming}")


                        # --- Now call retrieval ---
                        logger.info("Retrieving memory chain...")
                        memory_chain_data = self.retrieve_memory_chain(
                            initial_node_uuids=initial_uuids,
                            recent_concept_uuids=list(concepts_for_retrieval), # Pass previous concepts
                            current_mood=mood_for_retrieval # Pass previous mood
                            # last_turn_uuids argument removed
                        )
                        logger.info(f"Retrieved memory chain size: {len(memory_chain_data)}")
                    else:
                        logger.info("No relevant initial nodes found.")
                else:
                    logger.warning("Input started with [Image:] but wasn't handled as attachment? Proceeding without memory.")

                # Construct and call standard API
                logger.info("Constructing prompt string...")
                max_tokens = self.config.get('prompting', {}).get('max_context_tokens', 4096)
                prompt = self._construct_prompt(user_input, conversation_history, memory_chain_data, self.tokenizer, max_tokens)

                logger.info("Calling standard LLM Generate API...")
                # Call standard Generate API using configured helper
                ai_response = self._call_configured_llm('main_chat_text', prompt=prompt)
                # graph_user_input is already set

            # --- Process Response, Check for AI Action Request, and Update Graph ---
            action_result_message = None # To store feedback about executed action
            if not ai_response or ai_response.startswith("Error:"):
                logger.error(f"LLM call failed or returned error: {ai_response}")
                parsed_response = ai_response if ai_response else "Error: Received empty response from language model."
            else:
                parsed_response = ai_response.strip()
                logger.debug(f"Raw LLM response received: '{parsed_response[:200]}...'")

                # --- Check for AI-requested action ---
                action_match = re.search(r'\[ACTION:\s*(\{.*?\})\s*\]$', parsed_response, re.DOTALL)
                if action_match:
                    action_json_str = action_match.group(1)
                    logger.info(f"AI requested action detected: {action_json_str}")
                    # Remove the tag from the conversational response
                    parsed_response = parsed_response[:action_match.start()].strip()
                    logger.debug(f"Conversational part of response: '{parsed_response[:100]}...'")

                    try:
                        action_data = json.loads(action_json_str)
                        # --- Handle Autonomous Create File Intent ---
                        if action_data.get("action") == "create_file" and "description" in action_data.get("args", {}):
                            ai_description = action_data["args"]["description"]
                            logger.info(f"AI signaled intent to create file: '{ai_description}'")

                            # Call the dedicated generation prompt
                            gen_prompt_template = self._load_prompt("generate_file_content_prompt.txt")
                            if gen_prompt_template:
                                # Use the original user input as context for generation
                                gen_prompt = gen_prompt_template.format(
                                    user_request_context=user_input, # Pass original user input
                                    ai_description=ai_description
                                )
                                logger.debug(f"Sending file content generation prompt:\n{gen_prompt}")
                                # --- Use configured LLM call ---
                                gen_response = self._call_configured_llm('file_content_generation', prompt=gen_prompt)

                                # Parse the generated filename and content
                                try:
                                    gen_match = re.search(r'(\{.*?\})', gen_response, re.DOTALL)
                                    if gen_match:
                                        gen_json_str = gen_match.group(0)
                                        gen_data = json.loads(gen_json_str)
                                        generated_filename = gen_data.get("filename")
                                        generated_content = gen_data.get("content")

                                        if generated_filename and generated_content is not None:
                                            # Ensure .txt extension (basic check)
                                            if not generated_filename.lower().endswith(".txt"):
                                                generated_filename += ".txt"
                                                logger.info(f"Appended .txt extension to generated filename: {generated_filename}")

                                            # Execute the create_file action with generated data
                                            logger.info(f"Executing create_file with generated data: Filename='{generated_filename}', Content Length={len(generated_content)}")
                                            create_action_data = {
                                                "action": "create_file",
                                                "args": {"filename": generated_filename, "content": generated_content}
                                            }
                                            action_success, action_message, _ = self.execute_action(create_action_data) # execute_action returns suffix, but we might override
                                            action_result_message = f"[System: Action 'create_file' ({generated_filename}) {'succeeded' if action_success else 'failed'}. {action_message}]"
                                            action_suffix_for_info = f"create_file_{'success' if action_success else 'fail'}" # <<< Set suffix here
                                        else:
                                            logger.error("Generated file content JSON missing filename or content.")
                                            action_suffix_for_info = "create_file_fail" # Set suffix for error case
                                            # More specific user feedback
                                            action_result_message = f"[System: I intended to create a file about '{ai_description}', but failed to generate a valid filename or content.]"
                                    else:
                                         logger.error(f"Could not extract JSON from file generation response: {gen_response}")
                                         action_suffix_for_info = "create_file_fail" # Set suffix for error case
                                         # More specific user feedback
                                         action_result_message = f"[System: I intended to create a file about '{ai_description}', but received an invalid format from the content generator.]"
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON from file generation response: {e}. String: '{gen_json_str if 'gen_json_str' in locals() else gen_response}'")
                                    action_suffix_for_info = "create_file_fail" # Set suffix for error case
                                    # More specific user feedback
                                    action_result_message = f"[System: I intended to create a file about '{ai_description}', but failed to parse the generated content.]"
                                except Exception as e:
                                     logger.error(f"Error during file content generation/execution: {e}", exc_info=True)
                                     action_suffix_for_info = "create_file_exception" # Set suffix for error case
                                     # More specific user feedback
                                     action_result_message = f"[System: I intended to create a file about '{ai_description}', but encountered an error: {e}]"
                            else:
                                logger.error("Failed to load generate_file_content_prompt.txt")
                                action_suffix_for_info = "create_file_fail" # Set suffix for error case
                                # More specific user feedback
                                action_result_message = f"[System: I intended to create a file about '{ai_description}', but the content generation prompt is missing.]"

                        # --- Handle Other AI-Requested Actions ---
                        elif isinstance(action_data, dict) and "action" in action_data and "args" in action_data:
                            # Execute other actions directly as before
                            logger.info(f"Executing AI-requested action: {action_data.get('action')} with args: {action_data.get('args')}")
                            action_success, action_message, returned_suffix = self.execute_action(action_data) # <<< Capture suffix
                            action_result_message = f"[System: Action '{action_data.get('action')}' {'succeeded' if action_success else 'failed'}. {action_message}]"
                            action_suffix_for_info = returned_suffix # <<< Use returned suffix
                            logger.info(f"AI Action Result: {action_result_message}")
                        else:
                            logger.error(f"Invalid JSON structure in AI action request: {action_json_str}")
                            action_suffix_for_info = "action_parse_fail" # Set suffix for error case
                            action_result_message = "[System: Error - Invalid action request format received from AI.]"
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from AI action request: {e}. String: '{action_json_str}'")
                        action_result_message = "[System: Error - Failed to parse action request from AI.]"
                    except json.JSONDecodeError as e:
                        # ... (handle JSON error) ...
                        action_result_message = "[System: Error - Failed to parse action request from AI.]" # Set message
                        action_suffix_for_info = "action_parse_fail" # Set suffix
                    except Exception as e:
                         # ... (handle execution error) ...
                         action_result_message = f"[System: Error executing action - {e}]" # Set message
                         action_suffix_for_info = f"{action_data.get('action', 'unknown')}_exception" # Try to get action name

                    # --- Construct action_result_info HERE, inside the if action_match block ---
                    if action_result_message: # Only create if a message was generated
                        action_executed = action_data.get("action", "unknown") if 'action_data' in locals() else "unknown"
                        args_executed = action_data.get("args", {}) if 'action_data' in locals() else {}
                        target_info_placeholder = str(args_executed)[:100]

                        action_result_info = {
                            "message": action_result_message,
                            "action_type": action_suffix_for_info, # Use the suffix determined above
                            "target_info": target_info_placeholder
                        }
                        logger.debug(f"Prepared action_result_info (inside action_match): {action_result_info}")
                    # --- End construction block ---
                else:
                     logger.debug("No AI action request tag found in response.")


            # Add nodes to graph regardless of LLM success/failure, using appropriate text
            logger.info("Adding user input node to graph...")
            logger.debug(f"Adding user node with text: '{graph_user_input[:100]}...'")
            user_node_uuid = self.add_memory_node(graph_user_input, "User")

            logger.info("Adding AI/System response node to graph...")
            # Use parsed_response (which could be success or error message)
            logger.debug(f"Adding AI node with text: '{parsed_response[:100]}...'")
            ai_node_uuid = self.add_memory_node(parsed_response, "AI") # Add AI response node

            # --- Calculate and Store context for NEXT interaction's retrieval bias ---
            current_turn_concept_uuids = set()
            nodes_to_check_for_concepts = [user_node_uuid, ai_node_uuid] # Use UUIDs just added
            for turn_uuid in nodes_to_check_for_concepts:
                if turn_uuid and turn_uuid in self.graph:
                    try:
                        # Check outgoing edges for MENTIONS_CONCEPT
                        for successor_uuid in self.graph.successors(turn_uuid):
                            edge_data = self.graph.get_edge_data(turn_uuid, successor_uuid)
                            if edge_data and edge_data.get('type') == 'MENTIONS_CONCEPT':
                                if successor_uuid in self.graph and self.graph.nodes[successor_uuid].get('node_type') == 'concept':
                                    current_turn_concept_uuids.add(successor_uuid)
                                    # logger.debug(f"Identified concept '{successor_uuid[:8]}' mentioned by turn '{turn_uuid[:8]}' for next bias")
                    except Exception as concept_find_e:
                         logger.warning(f"Error finding concepts linked from turn {turn_uuid[:8]} for next bias: {concept_find_e}")
            logger.info(f"Storing {len(current_turn_concept_uuids)} concepts for next interaction's bias.")
            self.last_interaction_concept_uuids = current_turn_concept_uuids # Update state

            current_turn_mood = (0.0, 0.1) # Default: Neutral valence, low arousal
            mood_nodes_found = 0
            total_valence = 0.0
            total_arousal = 0.0
            for node_uuid in [user_node_uuid, ai_node_uuid]:
                if node_uuid and node_uuid in self.graph:
                    node_data = self.graph.nodes[node_uuid]
                    default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                    default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                    total_valence += node_data.get('emotion_valence', default_v)
                    total_arousal += node_data.get('emotion_arousal', default_a)
                    mood_nodes_found += 1
            if mood_nodes_found > 0:
                current_turn_mood = (total_valence / mood_nodes_found, total_arousal / mood_nodes_found)
            logger.info(f"Storing mood (Avg V/A): {current_turn_mood[0]:.2f} / {current_turn_mood[1]:.2f} for next interaction's bias.")
            self.last_interaction_mood = current_turn_mood # Update state

            # --- Check for Intention Request (if not handled as action/mod) ---
            # This check happens *after* adding the user/AI nodes, using the original user_input
            # We might want to move this earlier if intention analysis should prevent normal response generation.
            # For V1, let's just store it alongside the normal interaction flow.
            try:
                intention_result = self._analyze_intention_request(user_input)
                if intention_result.get("action") == "store_intention":
                    intention_content = intention_result.get("content")
                    intention_trigger = intention_result.get("trigger", "later")
                    # Format node text to include both content and trigger
                    intention_text = f"Remember: {intention_content} (Trigger: {intention_trigger})"
                    # Add the intention node, linked temporally after the AI response
                    intention_node_uuid = self.add_memory_node(
                        text=intention_text,
                        speaker="System", # Or 'AI'? System seems better for internal intention
                        node_type='intention'
                        # Timestamp will be set automatically
                    )
                    if intention_node_uuid:
                        logger.info(f"Stored intention node {intention_node_uuid[:8]}")
                        # Optionally, add a system message to the *next* response context?
                        # For now, just store it. The user sees their request and the AI response.
                    else:
                        logger.error("Failed to add intention node to graph.")
            except Exception as intent_e:
                 logger.error(f"Error during intention analysis/storage: {intent_e}", exc_info=True)


        except Exception as e:
            # Catch errors during interaction processing (e.g., the ValueError)
            logger.error(f"Error during process_interaction (ID: {interaction_id[:8]}): {e}", exc_info=True)
            # Assign error message to both ai_response and parsed_response
            ai_response = f"Error during processing: {e}"
            parsed_response = ai_response # Ensure parsed_response has a value
            memory_chain_data = [] # Clear memory chain data on error
            # --- Tuning Log: Interaction Error ---
            log_tuning_event("INTERACTION_ERROR", {
                "interaction_id": interaction_id,
                "personality": self.personality,
                "stage": "main_processing_loop",
                "error": str(e),
            })

        # --- Tuning Log: Interaction End ---
        log_tuning_event("INTERACTION_END", {
            "interaction_id": interaction_id,
            "personality": self.personality,
            "final_response_preview": parsed_response[:100],
            "retrieved_memory_count": len(memory_chain_data),
            "user_node_added": user_node_uuid[:8] if 'user_node_uuid' in locals() and user_node_uuid else None,
            "ai_node_added": ai_node_uuid[:8] if 'ai_node_uuid' in locals() and ai_node_uuid else None,
            "action_executed": action_result_message is not None,
        })

        # Return the AI's conversational response, memory chain, AI node UUID, and action result info separately
        # action_result_info is now constructed inside the 'if action_match:' block or remains None
        return parsed_response, memory_chain_data, ai_node_uuid if 'ai_node_uuid' in locals() else None, action_result_info


    # --- Consolidation ---
    # (Keep _select_nodes_for_consolidation and run_consolidation from previous version)
    def _select_nodes_for_consolidation(self, count: int = None) -> list[str]:
        """Selects recent 'turn' nodes for consolidation."""
        # (Keep implementation from previous version)
        if count is None: count = self.config.get('consolidation', {}).get('turn_count', 10)
        turn_nodes = [(u, d['timestamp']) for u, d in self.graph.nodes(data=True) if d.get('node_type') == 'turn' and d.get('timestamp')]
        turn_nodes.sort(key=lambda x: x[1], reverse=True)
        return [uuid for uuid, ts in turn_nodes[:count]]

    # --- Reset Memory ---
    # (Keep implementation from previous version)
    def reset_memory(self):
        """Resets the entire memory state."""
        # (Keep previous version)
        logger.info("--- RESETTING MEMORY ---")
        try:
            self.graph = nx.DiGraph(); self.embeddings.clear(); self.faiss_id_to_uuid.clear(); self.uuid_to_faiss_id.clear(); self.last_added_node_uuid = None
            if self.index is not None: self.index.reset(); logger.debug("FAISS index reset.")
            else: self.index = faiss.IndexFlatL2(self.embedding_dim); logger.debug("FAISS index initialized.")
            logger.info("In-memory structures cleared.")
            files_to_delete = [self.graph_file, self.index_file, self.embeddings_file, self.mapping_file]
            files_to_delete.append(self.asm_file) # Also delete ASM file
            files_to_delete.append(self.drives_file) # Also delete drives file
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    try: os.remove(file_path); logger.info(f"Deleted: {file_path}")
                    except OSError as e: logger.error(f"Error deleting {file_path}: {e}")
                else: logger.debug(f"Not found, skip delete: {file_path}")
            # Re-initialize drive state to defaults after deleting file
            self._initialize_drive_state()
            logger.info("--- MEMORY RESET COMPLETE ---"); return True
        except Exception as e:
            logger.error(f"Error during memory reset: {e}", exc_info=True)
            # Ensure drive state is also reset in case of error
            self.graph = nx.DiGraph(); self.embeddings = {}; self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}; self.last_added_node_uuid = None; self.index = faiss.IndexFlatL2(self.embedding_dim); self._initialize_drive_state(); logger.warning("Reset failed, re-initialized empty state.")
            return False

    # --- Drive State Management ---
    def _initialize_drive_state(self):
        """Initializes both short-term and long-term drive states."""
        drive_cfg = self.config.get('subconscious_drives', {})
        base_drives = drive_cfg.get('base_drives', {}) # Use base_drives from config

        # Initialize short-term drives to base values
        self.drive_state["short_term"] = base_drives.copy()
        # Initialize long-term drives to zero (or potentially base values if desired?)
        # Let's start with zero, assuming they develop over time.
        self.drive_state["long_term"] = {drive_name: 0.0 for drive_name in base_drives}

        logger.info(f"Drive state initialized: ShortTerm={self.drive_state['short_term']}, LongTerm={self.drive_state['long_term']}")

    def _load_drive_state(self):
        """Loads combined drive state (short & long term) from JSON file or initializes it."""
        default_state = {"short_term": {}, "long_term": {}}
        if os.path.exists(self.drives_file):
            try:
                with open(self.drives_file, 'r') as f:
                    loaded_state = json.load(f)
                # Validate structure
                if isinstance(loaded_state, dict) and "short_term" in loaded_state and "long_term" in loaded_state:
                    self.drive_state = loaded_state
                    logger.info(f"Loaded drive state from {self.drives_file}: {self.drive_state}")
                    # Optional: Validate loaded drives against config drives? Ensure all expected drives exist?
                    base_drives = self.config.get('subconscious_drives', {}).get('base_drives', {})
                    for term in ["short_term", "long_term"]:
                        for drive_name in base_drives:
                            if drive_name not in self.drive_state[term]:
                                logger.warning(f"Drive '{drive_name}' missing from loaded '{term}' state. Initializing.")
                                self.drive_state[term][drive_name] = base_drives[drive_name] if term == "short_term" else 0.0
                else:
                    logger.error(f"Invalid structure in drive state file {self.drives_file}. Initializing defaults.")
                    self._initialize_drive_state()
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error decoding drive state file {self.drives_file}: {e}. Initializing defaults.")
                self._initialize_drive_state()
            except Exception as e:
                logger.error(f"Unexpected error loading drive state: {e}. Initializing defaults.", exc_info=True)
                self._initialize_drive_state()
        else:
            logger.info(f"Drive state file not found ({self.drives_file}). Initializing defaults.")
            self._initialize_drive_state()

    def _save_drive_state(self):
        """Saves the current drive state to JSON file."""
        if not self.drive_state: # Don't save if empty (e.g., during init error)
            logger.warning("Skipping drive state save because state is empty.")
            return
        try:
            with open(self.drives_file, 'w') as f:
                json.dump(self.drive_state, f, indent=4)
            logger.debug(f"Drive state saved to {self.drives_file}.")
        except IOError as e:
            logger.error(f"IO error saving drive state to {self.drives_file}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"Error serializing drive state to JSON: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving drive state: {e}", exc_info=True)

    def _update_drive_state(self, relevant_nodes: list = None):
        """
        Placeholder: Updates drive activation levels based on recent experience.
        This will eventually involve LLM analysis or heuristics.
        For now, it might just apply decay.
        """

        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False):
            return # Do nothing if disabled

        logger.debug("Running short-term drive state update (Decay + LLM)...")
        decay_rate = drive_cfg.get('short_term_decay_rate', 0.05)
        base_drives = drive_cfg.get('base_drives', {})
        long_term_influence = drive_cfg.get('long_term_influence_on_baseline', 1.0)
        changed = False

        # --- Decay Step (towards dynamic baseline) ---
        if decay_rate > 0:
            for drive_name, current_activation in list(self.drive_state["short_term"].items()):
                config_baseline = base_drives.get(drive_name, 0.0)
                long_term_level = self.drive_state["long_term"].get(drive_name, 0.0)
                # Calculate the dynamic baseline towards which the short-term drive decays
                dynamic_baseline = config_baseline + (long_term_level * long_term_influence)

                # Apply decay towards the dynamic baseline
                new_activation = current_activation + (dynamic_baseline - current_activation) * decay_rate
                # Prevent overshoot
                if (current_activation > dynamic_baseline and new_activation < dynamic_baseline) or \
                   (current_activation < dynamic_baseline and new_activation > dynamic_baseline):
                    new_activation = dynamic_baseline

                if abs(new_activation - current_activation) > 1e-4: # Check for significant change
                    self.drive_state["short_term"][drive_name] = new_activation
                    changed = True
                    logger.debug(f"  Drive '{drive_name}' decayed towards dynamic baseline {dynamic_baseline:.3f}: {current_activation:.3f} -> {new_activation:.3f}")

        # --- LLM Analysis for Short-Term Drive Satisfaction/Frustration ---
        update_interval = drive_cfg.get('short_term_update_interval_interactions', 0) # Get interval from config
        # Check if LLM update should run based on interval (can be combined with decay later)
        # For now, let's assume if relevant_nodes are provided (e.g., from consolidation), we analyze them.
        if relevant_nodes and update_interval > 0: # Basic check, could be more sophisticated
            logger.info("Attempting LLM analysis for drive state update...")
            try:
                # 1. Select Nodes & Format Context
                # Use the nodes passed in (e.g., from consolidation)
                context_nodes_data = []
                for node_uuid in relevant_nodes:
                    if node_uuid in self.graph:
                        context_nodes_data.append(self.graph.nodes[node_uuid])
                context_nodes_data.sort(key=lambda x: x.get('timestamp', '')) # Sort chronologically
                context_text = "\n".join([f"{d.get('speaker', '?')}: {d.get('text', '')}" for d in context_nodes_data])

                if not context_text.strip():
                     logger.warning("No text context available from relevant nodes for drive analysis.")
                else:
                    # 2. Load Prompt
                    prompt_template = self._load_prompt("drive_analysis_prompt.txt")
                    if not prompt_template:
                        logger.error("Failed to load drive analysis prompt template. Skipping LLM update.")
                    else:
                        # --- Format Current Drive State for Prompt ---
                        current_drive_state_str = "[Current Drive State (Relative to Baseline):]\n"
                        drive_state_parts = []
                        for drive_name, current_activation in self.drive_state["short_term"].items():
                            config_baseline = base_drives.get(drive_name, 0.0)
                            long_term_level = self.drive_state["long_term"].get(drive_name, 0.0)
                            dynamic_baseline = config_baseline + (long_term_level * long_term_influence)
                            deviation = current_activation - dynamic_baseline
                            state_desc = "Neutral"
                            if deviation > 0.2: state_desc = "High"
                            elif deviation < -0.2: state_desc = "Low"
                            drive_state_parts.append(f"- {drive_name}: {state_desc} (Deviation: {deviation:+.2f})")
                        current_drive_state_str += "\n".join(drive_state_parts) if drive_state_parts else "Neutral"

                        # 3. Call LLM
                        full_prompt = prompt_template.format(
                            context_text=context_text,
                            current_drive_state=current_drive_state_str # Pass formatted state
                        )
                        logger.debug(f"Sending drive analysis prompt:\n{full_prompt[:500]}...") # Log more context
                        # --- Use configured LLM call ---
                        llm_response_str = self._call_configured_llm('drive_analysis_short_term', prompt=full_prompt)

                        # 4. Parse Response
                        if llm_response_str and not llm_response_str.startswith("Error:"):
                            try:
                                # Extract JSON
                                match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
                                if match:
                                    json_str = match.group(0)
                                    drive_adjustments = json.loads(json_str)
                                    logger.debug(f"Parsed short-term drive adjustments from LLM: {drive_adjustments}")

                                    # 5. Adjust short_term drive_state based on satisfaction/frustration factors
                                    satisfaction_factor = drive_cfg.get('llm_satisfaction_factor', 0.1) # Use new config key
                                    frustration_factor = drive_cfg.get('llm_frustration_factor', 0.15) # Use new config key

                                    for drive_name, status in drive_adjustments.items():
                                        if drive_name in self.drive_state["short_term"]:
                                            current_activation = self.drive_state["short_term"][drive_name]
                                            # Calculate dynamic baseline for comparison (needed for satisfaction logic)
                                            config_baseline = base_drives.get(drive_name, 0.0)
                                            long_term_level = self.drive_state["long_term"].get(drive_name, 0.0)
                                            dynamic_baseline = config_baseline + (long_term_level * long_term_influence)
                                            adjustment = 0.0

                                            if status == "satisfied":
                                                # Reduce activation towards dynamic baseline (multiplicative adjustment on deviation)
                                                deviation = current_activation - dynamic_baseline
                                                if deviation > 0: # Only reduce if above baseline
                                                     adjustment = -deviation * satisfaction_factor # Adjustment is negative
                                                logger.debug(f"  Drive '{drive_name}' satisfied. Adjustment: {adjustment:.3f}")
                                            elif status == "frustrated":
                                                # Increase activation (additive adjustment)
                                                adjustment = frustration_factor # Adjustment is positive
                                                logger.debug(f"  Drive '{drive_name}' frustrated. Adjustment: {adjustment:.3f}")
                                            # else: status is neutral/unclear, no adjustment

                                            if abs(adjustment) > 1e-4:
                                                # Apply adjustment to short-term state
                                                new_level = current_activation + adjustment
                                                # Optional: Clamp short-term activation? (e.g., between -1 and 2?)
                                                # new_level = max(-1.0, min(2.0, new_level))
                                                self.drive_state["short_term"][drive_name] = new_level
                                                changed = True # Mark that state was changed by LLM analysis
                                                logger.info(f"Applied LLM drive adjustment to '{drive_name}' ({status}): {current_activation:.3f} -> {new_level:.3f} (Adj: {adjustment:.3f})")
                                        else:
                                            logger.warning(f"LLM returned adjustment for unknown drive '{drive_name}'.")

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON from drive analysis response: {e}. Raw: '{llm_response_str}'")
                            except Exception as e:
                                logger.error(f"Error processing drive analysis LLM response: {e}", exc_info=True)
                        else:
                            logger.error(f"LLM call failed or returned error for drive analysis: {llm_response_str}")

            except Exception as e:
                 logger.error(f"Unexpected error during LLM drive state analysis: {e}", exc_info=True)

        if changed:
            # Log final state after decay and potential LLM adjustments
            logger.info(f"Short-term drive state updated (Decay & LLM applied): {self.drive_state['short_term']}")
            # Saving happens in the calling function (e.g., run_consolidation or _save_memory)

    def _update_long_term_drives(self):
        """
        Updates long-term drive levels based on LLM analysis of the ASM or other
        long-term memory indicators. Called less frequently than short-term updates.
        """
        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False): return
        if not self.autobiographical_model:
            logger.warning("Skipping long-term drive update: Autobiographical Self-Model is empty.")
            return

        logger.info("Attempting LLM analysis for long-term drive state update...")
        try:
            # 1. Format Context (Using ASM)
            # Create a readable text summary from the structured ASM
            asm_parts = []
            if self.autobiographical_model.get("summary_statement"): asm_parts.append(f"Overall Summary: {self.autobiographical_model['summary_statement']}")
            if self.autobiographical_model.get("core_traits"): asm_parts.append(f"Core Traits: {', '.join(self.autobiographical_model['core_traits'])}")
            if self.autobiographical_model.get("recurring_themes"): asm_parts.append(f"Recurring Themes: {', '.join(self.autobiographical_model['recurring_themes'])}")
            if self.autobiographical_model.get("values_beliefs"): asm_parts.append(f"Values/Beliefs: {', '.join(self.autobiographical_model['values_beliefs'])}")
            if self.autobiographical_model.get("significant_events"): asm_parts.append(f"Significant Events: {'; '.join(self.autobiographical_model['significant_events'])}")
            asm_summary_text = "\n".join(asm_parts)

            if not asm_summary_text.strip():
                logger.warning("ASM exists but generated empty summary text for long-term drive analysis.")
                return

            # 2. Load Prompt
            prompt_template = self._load_prompt("long_term_drive_analysis_prompt.txt")
            if not prompt_template:
                logger.error("Failed to load long_term_drive_analysis_prompt.txt. Skipping update.")
                return

            # 3. Call LLM
            full_prompt = prompt_template.format(asm_summary_text=asm_summary_text)
            logger.debug(f"Sending long-term drive analysis prompt:\n{full_prompt[:300]}...")
            # --- Use configured LLM call ---
            llm_response_str = self._call_configured_llm('drive_analysis_long_term', prompt=full_prompt)

            # 4. Parse Response
            if llm_response_str and not llm_response_str.startswith("Error:"):
                try:
                    match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        long_term_assessment = json.loads(json_str)
                        logger.debug(f"Parsed long-term drive assessment from LLM: {long_term_assessment}")

                        # 5. Adjust long_term drive_state
                        adjustment_factor = drive_cfg.get('long_term_adjustment_factor', 0.05)
                        changed = False

                        for drive_name, assessment in long_term_assessment.items():
                            if drive_name in self.drive_state["long_term"]:
                                current_long_term_level = self.drive_state["long_term"][drive_name]
                                adjustment = 0.0

                                if assessment == "positive":
                                    # Nudge towards positive (e.g., +1 max)
                                    adjustment = (1.0 - current_long_term_level) * adjustment_factor
                                    logger.debug(f"  Long-term '{drive_name}' assessed positive. Adjustment: {adjustment:.3f}")
                                elif assessment == "negative":
                                    # Nudge towards negative (e.g., -1 min)
                                    adjustment = (-1.0 - current_long_term_level) * adjustment_factor
                                    logger.debug(f"  Long-term '{drive_name}' assessed negative. Adjustment: {adjustment:.3f}")
                                # else: neutral, no adjustment

                                if abs(adjustment) > 1e-4:
                                    new_level = current_long_term_level + adjustment
                                    # Clamp long-term levels (e.g., between -1 and 1)
                                    new_level = max(-1.0, min(1.0, new_level))
                                    self.drive_state["long_term"][drive_name] = new_level
                                    changed = True
                            else:
                                logger.warning(f"LLM returned assessment for unknown drive '{drive_name}'.")

                        if changed:
                            logger.info(f"Long-term drive state updated: {self.drive_state['long_term']}")
                            # Save immediately after long-term update? Yes, seems appropriate.
                            self._save_drive_state()
                            self._save_memory() # Also save graph/etc. which includes drive state file saving

                    else:
                        logger.error(f"Could not extract JSON from long-term drive analysis response. Raw: '{llm_response_str}'")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from long-term drive analysis response: {e}. Raw: '{llm_response_str}'")
                except Exception as e:
                    logger.error(f"Error processing long-term drive analysis LLM response: {e}", exc_info=True)
            else:
                logger.error(f"LLM call failed or returned error for long-term drive analysis: {llm_response_str}")

        except Exception as e:
            logger.error(f"Unexpected error during long-term drive state update: {e}", exc_info=True)


    # *** ADDED: Wrapper methods for file operations ***
    def create_workspace_file(self, filename: str, content: str) -> bool:
        """Wrapper to create/overwrite a file in the workspace."""
        logger.info(f"Client request to create/overwrite workspace file: {filename}")
        # Pass personality to the file manager function
        return file_manager.create_or_overwrite_file(self.config, self.personality, filename, content)

    def append_to_workspace_file(self, filename: str, content_to_append: str) -> bool:
        """Wrapper to append content to a file in the workspace."""
        logger.info(f"Client request to append to workspace file: {filename}")
        # Pass personality to the file manager function
        return file_manager.append_to_file(self.config, self.personality, filename, content_to_append)

    def add_calendar_event_wrapper(self, event_date: str, event_time: str, description: str) -> bool:
        """Wrapper to add an event to the calendar file."""
        logger.info(f"Client request to add calendar event: {event_date} {event_time} - {description}")
        # Pass personality to the file manager function
        return file_manager.add_calendar_event(self.config, self.personality, event_date, event_time, description)

    def read_calendar_events_wrapper(self, target_date: str = None) -> list[dict]:
        """Wrapper to read events from the calendar file."""
        logger.info(f"Client request to read calendar events (Date: {target_date or 'All'}).")
        # Pass personality to the file manager function
        success, message = file_manager.read_calendar_events(self.config, self.personality, target_date)
        # Return only the list of events for internal use, message is discarded here
        return success if isinstance(success, list) else [] # Ensure list return

    def list_files_wrapper(self) -> tuple[list[str] | None, str]:
        """Wrapper to list files in the workspace."""
        logger.info("Client request to list workspace files.")
        return file_manager.list_files(self.config, self.personality)

    def read_file_wrapper(self, filename: str) -> tuple[str | None, str]:
        """Wrapper to read a file from the workspace."""
        logger.info(f"Client request to read workspace file: {filename}")
        return file_manager.read_file(self.config, self.personality, filename)

    def delete_file_wrapper(self, filename: str) -> tuple[bool, str]:
        """Wrapper to delete a file from the workspace."""
        logger.info(f"Client request to delete workspace file: {filename}")
        return file_manager.delete_file(self.config, self.personality, filename)

    # --- Updated signature to accept parameters ---
    def _call_kobold_multimodal_api(self, messages: list, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """
        Sends prompt to KoboldCpp OpenAI-compatible Chat API, handles multimodal messages.
        Parameters are passed in, typically from _call_configured_llm.
        """
        logger.debug(f"Calling Kobold Chat Completions API ({self.kobold_chat_api_url})")
        logger.debug(f"  Params: model={model_name}, max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        # Maybe get model name from config? Kobold might ignore it anyway.
        model_name = self.config.get('kobold_model_name', 'gemma-3-27b-it')  # Optional

        payload = {
            "model": model_name,
            "messages": messages,  # List of {"role": "user/assistant", "content": ...} dicts
            "max_tokens": max_tokens,
            # Parameter name might be different for Kobold's OpenAI API? Check Kobold docs if needed. Often 'max_tokens'.
            "temperature": temperature,
            "top_p": top_p,
            # Add other parameters if needed and supported by Kobold's implementation
        }

        # Log payload carefully (messages can be large)
        log_payload_summary = {k: v for k, v in payload.items() if k != 'messages'}
        log_payload_summary['messages_count'] = len(payload['messages'])
        # Log first/last message content briefly
        if payload['messages']:
            first_msg = payload['messages'][0]
            last_msg = payload['messages'][-1]
            log_payload_summary['first_message_role'] = first_msg.get('role')
            log_payload_summary['last_message_role'] = last_msg.get('role')
            # Log content type of last message (potentially multimodal)
            if isinstance(last_msg.get('content'), list):
                log_payload_summary['last_message_content_types'] = [item.get('type') for item in last_msg['content']]
            elif isinstance(last_msg.get('content'), str):
                log_payload_summary['last_message_content_preview'] = last_msg['content'][:50] + "..."

        logger.debug(f"Payload for Chat Completions API: {log_payload_summary}")
        # logger.debug(f"Full Payload (use with caution): {json.dumps(payload, indent=2)}") # Uncomment for deep debug

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.kobold_chat_api_url, headers=headers, json=payload,
                                     timeout=180)  # Increased timeout
            response.raise_for_status()  # Check for HTTP errors

            result = response.json()
            logger.debug(f"Raw Chat Completions API response JSON: {result}")  # Log raw response

            # Parse the response according to OpenAI format
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                content = message.get('content', '').strip()
                if content:
                    logger.debug(f"Extracted content: '{content[:100]}...'")
                    return content
                else:
                    logger.warning("API response had choice but no message content.")
                    return "Error: Received empty content from AI."
            else:
                logger.warning(f"API response did not contain expected 'choices': {result}")
                return "Error: Received unexpected response format from AI."

        except requests.exceptions.RequestException as e:
            logger.error(f"Kobold Chat API connection/request error: {e}", exc_info=True)
            return f"Error: Could not connect to Kobold Chat API at {self.kobold_chat_api_url}."
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from Kobold Chat API: {response.text}", exc_info=True)
            return "Error: Failed to decode AI response."
        except Exception as e:
            logger.error(f"Kobold Chat API call unexpected error: {e}", exc_info=True)
            return f"Error: Unexpected issue during Kobold Chat API call."

    def _call_configured_llm(self, task_name: str, prompt: str = None, messages: list = None, **overrides) -> str:
        """
        Calls the appropriate LLM API based on configuration for the given task.

        Args:
            task_name: The key for the task in config['llm_models'].
            prompt: The prompt string (for 'generate' API type).
            messages: The list of messages (for 'chat_completions' API type).
            **overrides: Keyword arguments to override default parameters from config.

        Returns:
            The generated text response string, or an error message string.
        """
        logger.debug(f"Calling configured LLM for task: '{task_name}'")
        task_config = self.config.get('llm_models', {}).get(task_name)

        if not task_config:
            err_msg = f"Error: LLM configuration for task '{task_name}' not found in config.yaml."
            logger.error(err_msg)
            return err_msg

        api_type = task_config.get('api_type')
        model_name = task_config.get('model_name', 'koboldcpp-default')

        # --- Get default parameters from config ---
        default_params = {
            'max_length': task_config.get('max_length', 512),
            'max_tokens': task_config.get('max_tokens', 512), # For chat API
            'temperature': task_config.get('temperature', 0.7),
            'top_p': task_config.get('top_p', 0.95),
            'top_k': task_config.get('top_k', 60),
            'min_p': task_config.get('min_p', 0.0),
            # Add other potential parameters here if needed
        }

        # --- Merge overrides ---
        final_params = default_params.copy()
        final_params.update(overrides)
        logger.debug(f"  Task Config: {task_config}")
        logger.debug(f"  Final Params: {final_params}")


        # --- Call appropriate API ---
        if api_type == 'generate':
            if prompt is None:
                err_msg = f"Error: Prompt is required for 'generate' API type (task: {task_name})."
                logger.error(err_msg)
                return err_msg
            # Pass parameters explicitly to _call_kobold_api
            return self._call_kobold_api(
                prompt=prompt,
                model_name=model_name, # Pass model name
                max_length=final_params['max_length'],
                temperature=final_params['temperature'],
                top_p=final_params['top_p'],
                top_k=final_params['top_k'],
                min_p=final_params['min_p']
            )
        elif api_type == 'chat_completions':
            if messages is None:
                err_msg = f"Error: Messages list is required for 'chat_completions' API type (task: {task_name})."
                logger.error(err_msg)
                return err_msg
            # Pass parameters explicitly to _call_kobold_multimodal_api
            return self._call_kobold_multimodal_api(
                messages=messages,
                model_name=model_name, # Pass model name
                max_tokens=final_params['max_tokens'],
                temperature=final_params['temperature'],
                top_p=final_params['top_p']
                # Add top_k, min_p if supported by chat API later
            )
        else:
            err_msg = f"Error: Unknown api_type '{api_type}' configured for task '{task_name}'."
            logger.error(err_msg)
            return err_msg


    # --- Forgetting Mechanism ---
    def run_memory_maintenance(self):
        """
        Placeholder: Runs the nuanced forgetting process.
        Identifies candidate nodes, calculates forgettability scores, and archives nodes
        exceeding the threshold.
        Triggered periodically based on interaction count or other criteria.
        """
        if not self.config.get('features', {}).get('enable_forgetting', False):
            logger.debug("Nuanced forgetting feature disabled. Skipping maintenance.")
            return

        logger.info("--- Running Memory Maintenance (Nuanced Forgetting - Placeholder) ---")
        # 1. Get config: threshold, weights, min_age, min_activation, protected types
        forget_cfg = self.config.get('forgetting', {})
        score_threshold = forget_cfg.get('score_threshold', 0.7)
        weights = forget_cfg.get('weights', {})
        min_age_hr = forget_cfg.get('candidate_min_age_hours', 24)
        min_activation = forget_cfg.get('candidate_min_activation', 0.05)
        protected_types = forget_cfg.get('protected_node_types', [])
        logger.debug(
            f"Forgetting Params: Threshold={score_threshold}, MinAgeHr={min_age_hr}, MinAct={min_activation}, Protected={protected_types}, Weights={weights}")

        # 2. Identify Candidate Nodes:
        #    - status == 'active'
        #    - node_type NOT IN protected_types
        #    - age > min_age_hr
        #    - activation_level < min_activation (using 'activation_level' attribute)
        candidate_uuids = []
        current_time = time.time()
        min_age_sec = min_age_hr * 3600
        for uuid, data in self.graph.nodes(data=True):
            if data.get('status', 'active') != 'active': continue
            if data.get('node_type') in protected_types: continue
            last_accessed = data.get('last_accessed_ts', 0)
            age_sec = current_time - last_accessed
            if age_sec < min_age_sec: continue
            # Note: 'activation_level' might not be updated frequently unless retrieval runs often.
            # Consider using decay calculation based on last_accessed_ts instead?
            # For now, use stored 'activation_level' if available.
            current_activation = data.get('activation_level', 0.0)  # This is the graph node's stored activation
            if current_activation >= min_activation: continue

            candidate_uuids.append(uuid)

        logger.info(f"Found {len(candidate_uuids)} candidate nodes for potential forgetting.")
        if not candidate_uuids: return

        # 3. Calculate Forgettability Score for each candidate:
        archived_count = 0
        for uuid in candidate_uuids:
            node_data = self.graph.nodes[uuid]
            score = self._calculate_forgettability(uuid, node_data, current_time, weights)
            logger.debug(f"  Node {uuid[:8]} ({node_data.get('node_type')}): Forgettability Score = {score:.3f}")

            # 4. Archive if score exceeds threshold (Soft Delete):
            if score >= score_threshold:
                node_data['status'] = 'archived'
                archived_count += 1
                logger.info(f"  Archiving node {uuid[:8]} (Score: {score:.3f} >= {score_threshold})")
                # Remove from FAISS index (requires rebuild or selective removal if supported)
                # Deferring index modification until after loop or using rebuild.
            # else:
            # logger.debug(f"  Keeping node {uuid[:8]} (Score: {score:.3f} < {score_threshold})")

        # 5. Rebuild FAISS index if nodes were archived
        if archived_count > 0:
            logger.info(f"Archived {archived_count} nodes. Rebuilding FAISS index to remove them...")
            # Make sure rebuild respects status='active'
            self._rebuild_index_from_graph_embeddings()  # Rebuild should only include 'active' nodes now
            self._save_memory()  # Save changes after maintenance
        # --- Logging block moved from here ---


    def _calculate_forgettability(self, node_uuid: str, node_data: dict, current_time: float,
                                  weights: dict) -> float:
        """
        Placeholder: Calculates a score indicating how likely a node is to be forgotten.
        Score range should ideally be normalized (e.g., 0-1).
        """
        # --- Get Raw Factors ---
        # Recency: Time since last access (higher = more forgettable)
        last_accessed = node_data.get('last_accessed_ts', 0)
        recency_sec = max(0, current_time - last_accessed)

        # Activation: Current activation level (lower = more forgettable)
        activation = node_data.get('activation_level', 0.0)  # Graph activation level

        # Node Type: Some types intrinsically more forgettable
        node_type = node_data.get('node_type', 'default')
        # (Assign numeric value based on type - e.g., turn=1.0, summary=0.5, concept=0.2?)

        # Saliency: Higher saliency resists forgetting
        saliency = node_data.get('saliency_score', 0.0)

        # Emotion: Higher arousal/valence magnitude resists forgetting
        valence = node_data.get('emotion_valence', 0.0)
        arousal = node_data.get('emotion_arousal', 0.1)
        emotion_magnitude = math.sqrt(valence ** 2 + arousal ** 2)  # Simple magnitude

        # Connectivity: Higher degree resists forgetting
        degree = self.graph.degree(node_uuid) if node_uuid in self.graph else 0

        # --- Normalize Factors (Example - needs tuning) ---
        # Normalize recency (e.g., using exponential decay or mapping to 0-1 over a time range)
        # Example: Normalize over a week (604800 seconds)
        norm_recency = min(1.0, recency_sec / 604800.0)

        # Normalize activation (already 0-1 theoretically, but use inverse)
        norm_inv_activation = 1.0 - min(1.0, max(0.0, activation))  # Low activation -> high score component

        # Normalize node type factor (example mapping)
        type_map = {'turn': 1.0, 'summary': 0.4, 'concept': 0.1, 'default': 0.6}
        norm_type = type_map.get(node_type, 0.6)

        # Normalize saliency (use inverse: low saliency -> high score component)
        norm_inv_saliency = 1.0 - min(1.0, max(0.0, saliency))

        # Normalize emotion (use inverse: low magnitude -> high score component)
        norm_inv_emotion = 1.0 - min(1.0, max(0.0, emotion_magnitude))

        # Normalize connectivity (use inverse, map degree to 0-1 range, e.g., log scale or capped)
        # Example: cap at 10 neighbors for normalization
        norm_inv_connectivity = 1.0 - min(1.0, degree / 10.0)

        # --- Calculate Weighted Score ---
        score = 0.0
        score += norm_recency * weights.get('recency_factor', 0.0)
        score += norm_inv_activation * weights.get('activation_factor', 0.0)
        score += norm_type * weights.get('node_type_factor', 0.0)
        # Resistance factors (higher values decrease forgettability)
        # We used inverse normalization above, so apply positive weights here.
        score += norm_inv_saliency * abs(
            weights.get('saliency_factor', 0.0))  # Use abs() in case config uses negative
        score += norm_inv_emotion * abs(weights.get('emotion_factor', 0.0))
        score += norm_inv_connectivity * abs(weights.get('connectivity_factor', 0.0))

        # Clamp final score 0-1
        final_score = max(0.0, min(1.0, score))

        # logger.debug(f"    Forget Score Factors for {node_uuid[:8]}: Rec({norm_recency:.2f}), Act({norm_inv_activation:.2f}), Typ({norm_type:.2f}), Sal({norm_inv_saliency:.2f}), Emo({norm_inv_emotion:.2f}), Con({norm_inv_connectivity:.2f}) -> Score: {final_score:.3f}")

        # Removed duplicate return statement that was here

    def purge_weak_nodes(self):
        """
        Permanently deletes nodes whose memory_strength is below a configured threshold
        and are older than a configured minimum age.
        """
        if not self.config.get('features', {}).get('enable_forgetting', False):
            logger.debug("Forgetting/Purging feature disabled. Skipping purge.")
            return

        strength_cfg = self.config.get('memory_strength', {})
        purge_threshold = strength_cfg.get('purge_threshold', 0.01)
        # min_age_days = strength_cfg.get('purge_min_age_days', 60) # REMOVED
        # min_age_seconds = min_age_days * 24 * 3600 # REMOVED

        logger.warning(f"--- Purging Weak Nodes (Strength < {purge_threshold}) ---") # Removed age from log message
        # --- Tuning Log: Purge Start ---
        log_tuning_event("PURGE_START", {
            "personality": self.personality,
            "strength_threshold": purge_threshold,
            # "min_age_days": min_age_days, # REMOVED
        })

        purge_count = 0
        current_time = time.time()
        nodes_to_purge = []
        nodes_snapshot = list(self.graph.nodes(data=True)) # Snapshot

        for uuid, data in nodes_snapshot:
            current_strength = data.get('memory_strength', 1.0)
            # --- Purge only based on strength ---
            if current_strength < purge_threshold:
                nodes_to_purge.append(uuid)
                logger.debug(f"Marked weak node {uuid[:8]} for purging (Strength: {current_strength:.3f})")
            # else: pass # Node strength is above threshold

        # Perform deletion
        if nodes_to_purge:
            logger.info(f"Attempting to permanently purge {len(nodes_to_purge)} weak nodes...")
            for uuid in list(nodes_to_purge): # Iterate over copy
                if self.delete_memory_entry(uuid): # delete_memory_entry handles graph, embed, index, map, rebuild
                    purge_count += 1
                else:
                    logger.error(f"Failed to purge weak node {uuid[:8]}. It might have been deleted already.")

            logger.info(f"--- Purge Complete: {purge_count} nodes permanently deleted. ---")
            # delete_memory_entry already rebuilds/saves if successful.
        else:
            logger.info("--- Purge Complete: No weak nodes met the criteria for purging. ---")

        # --- Tuning Log: Purge End ---
        log_tuning_event("PURGE_END", {
            "personality": self.personality,
            "purged_count": purge_count,
            "purged_uuids": nodes_to_purge, # Log UUIDs that were targeted
        })

    def execute_action(self, action_data: dict) -> tuple[bool, str, str]:
        """
        Executes a validated action based on the action_data dictionary.

        Args:
            action_data: Dictionary containing 'action' and 'args'.

        Returns:
             A tuple (success: bool, message: str, action_suffix: str).
             'message' is user-facing status.
             'action_suffix' is for internal GUI state/styling (e.g., 'create_file_success').
        """
        action = action_data.get("action", "unknown")
        args = action_data.get("args", {})
        logger.debug(f"Attempting to execute action '{action}' with args: {args}")

        success = False
        message = f"Action '{action}' could not be completed."  # Default failure message
        action_suffix = f"{action}_fail"  # Default suffix

        try:
            # --- File Actions ---
            if action == "create_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content is not None:  # Content can be empty string
                    # Call file_manager function which now returns (bool, str)
                    success, message = file_manager.create_or_overwrite_file(self.config, self.personality, filename,
                                                                             str(content))
                else:
                    message = "Error: Missing filename or content for create_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            elif action == "append_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content is not None:  # Content can be empty string
                    success, message = file_manager.append_to_file(self.config, self.personality, filename,
                                                                   str(content))
                else:
                    message = "Error: Missing filename or content for append_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            # --- Calendar Actions ---
            elif action == "add_calendar_event":
                date, time_str, desc = args.get("date"), args.get("time"), args.get("description")
                if date and time_str and desc:
                    # TODO: Date/Time parsing/validation could happen here or in file_manager
                    # For now, pass strings directly
                    success, message = file_manager.add_calendar_event(self.config, self.personality, date, time_str,
                                                                       desc)
                else:
                    message = "Error: Missing date, time, or description for add_calendar_event."
                    logger.error(message + f" Args received: {args}")
                    success = False

            elif action == "read_calendar":
                date = args.get("date")  # Optional
                # file_manager.read_calendar_events now returns (list[dict], str)
                events, fm_message = file_manager.read_calendar_events(self.config, self.personality, date)
                # For reading, success is true if the read operation itself didn't fail,
                # even if no events were found. The message conveys the outcome.
                success = True  # Assume read operation itself succeeded unless exception below
                action_suffix = "read_calendar_success"  # Use success suffix
                date_str = f" for {date}" if date else " (all dates)"

                # Construct user-facing message based on results
                if fm_message.startswith("IO error") or fm_message.startswith(
                        "Permission denied") or fm_message.startswith("Unexpected error"):
                    success = False  # Override success if file_manager reported read error
                    action_suffix = "read_calendar_fail"
                    message = f"Error reading calendar: {fm_message}"  # Pass file_manager error message
                elif not events:
                    # Message could be "not found" or "found 0 events" - use fm_message
                    message = fm_message
                else:
                    event_lines = []
                    for e in sorted(events,
                                    key=lambda x: (x.get('event_date', ''), x.get('event_time', ''))):  # Sort events
                        event_lines.append(
                            f"- {e.get('event_time', '?')}: {e.get('description', '?')} ({e.get('event_date', '?')})")
                    message = f"Found {len(events)} event(s){date_str}:\n" + "\n".join(event_lines)

            # --- NEW File Actions ---
            elif action == "list_files":
                file_list, message = self.list_files_wrapper()
                if file_list is not None:
                    success = True
                    # Format the message for the user
                    if file_list: message = f"Files in workspace:\n- " + "\n- ".join(file_list)
                    else: message = "Workspace is empty."
                else: # Error occurred
                    success = False
                    # message already contains the error from file_manager

            elif action == "read_file":
                filename = args.get("filename")
                if filename:
                    file_content, message = self.read_file_wrapper(filename)
                    if file_content is not None:
                        success = True
                        # Truncate long content for the message?
                        content_preview = file_content[:500] + ('...' if len(file_content) > 500 else '')
                        message = f"Content of '{filename}':\n---\n{content_preview}\n---"
                    else: # Error occurred
                        success = False
                        # message already contains the error from file_manager
                else:
                    message = "Error: Missing filename for read_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            elif action == "delete_file":
                filename = args.get("filename")
                if filename:
                    # Add confirmation step? For now, execute directly.
                    # Consider adding a config flag for delete confirmation later.
                    logger.warning(f"Executing delete_file action for: {filename}")
                    success, message = self.delete_file_wrapper(filename)
                else:
                    message = "Error: Missing filename for delete_file."
                    logger.error(message + f" Args received: {args}")
                    success = False

            # --- Unknown Action ---
            else:
                message = f"Error: The action '{action}' is not recognized or supported."
                logger.error(message + f" Action data: {action_data}")
                success = False
                action_suffix = "unknown_action_fail"

            # --- Update Suffix ---
            # Ensure suffix reflects success/failure determined within this block
            if action != "unknown_action_fail":  # Don't override specific unknown suffix
                action_suffix = f"{action}_{'success' if success else 'fail'}"

        except Exception as e:
            # Catch any unexpected errors during the dispatch/execution logic itself
            logger.error(f"Unexpected exception during execution of action '{action}': {e}", exc_info=True)
            message = f"An internal error occurred while trying to perform action '{action}'. Please check logs."
            success = False
            action_suffix = f"{action}_exception"  # Specific suffix for exceptions here

        # --- Final Logging and Return ---
        log_level = logging.INFO if success else logging.ERROR
        # Log the full message if it's short, otherwise truncate
        log_message_detail = message if len(message) < 150 else message[:150] + '...'
        logger.log(log_level, f"Action '{action}' execution result: Success={success}, Suffix='{action_suffix}', Msg='{log_message_detail}'")

        # --- Apply Heuristic Drive Adjustment ---
        try:
            drive_cfg = self.config.get('subconscious_drives', {})
            if drive_cfg.get('enabled', False):
                heuristics = drive_cfg.get('heuristic_adjustment_factors', {})
                adjustment = 0.0
                target_drive = "Control" # Default target drive for actions

                if success:
                    adjustment = heuristics.get('action_success_control', 0.0)
                else:
                    adjustment = heuristics.get('action_fail_control', 0.0)

                if abs(adjustment) > 1e-4 and target_drive in self.drive_state["short_term"]:
                    current_level = self.drive_state["short_term"][target_drive]
                    new_level = current_level + adjustment
                    # Optional: Clamp adjustment range?
                    self.drive_state["short_term"][target_drive] = new_level
                    logger.info(f"Applied heuristic drive adjustment to '{target_drive}' due to action result ({'Success' if success else 'Fail'}): {current_level:.3f} -> {new_level:.3f} (Adj: {adjustment:.3f})")
                    # No need to save here, saving happens elsewhere (e.g., end of interaction)
        except Exception as e:
            logger.error(f"Error applying heuristic drive adjustment after action execution: {e}", exc_info=True)

        return success, message, action_suffix

    def _consolidate_summarize(self, context_text: str, nodes_data: list, active_nodes_to_process: list) -> tuple[str | None, bool]:
        """Helper to generate and store the summary node."""
        prompt_template = self._load_prompt("summary_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load summary prompt template. Skipping summarization.")
            return None, False

        summary_prompt = prompt_template.format(context_text=context_text)

        logger.info("Requesting summary (from file prompt)...")
        # --- Use configured LLM call ---
        summary_text = self._call_configured_llm('consolidation_summary', prompt=summary_prompt)
        summary_node_uuid = None
        summary_created = False
        if summary_text and len(summary_text) > 10:
            logger.info(f"Generated Summary: '{summary_text[:100]}...'")
            summary_ts = nodes_data[-1].get('timestamp') if nodes_data else datetime.now(timezone.utc).isoformat()
            summary_node_uuid = self.add_memory_node(summary_text, "System", 'summary', summary_ts, 0.7)
            if summary_node_uuid:
                summary_created = True
                logger.info(f"Added summary node {summary_node_uuid[:8]}. Adding 'SUMMARY_OF' edges...")
                current_time = time.time()
                for orig_uuid in active_nodes_to_process: # Use the correct parameter name
                    if orig_uuid in self.graph:
                        try:
                            self.graph.add_edge(summary_node_uuid, orig_uuid, type='SUMMARY_OF', base_strength=0.9,
                                                last_traversed_ts=current_time)
                        except Exception as e:
                            logger.error(
                                f"Error adding SUMMARY_OF edge from {summary_node_uuid[:8]} to {orig_uuid[:8]}: {e}")
            else:
                logger.error("Failed to add summary node.")
        else:
            logger.warning(f"No valid summary generated ('{summary_text}').")
        return summary_node_uuid, summary_created

    def _consolidate_extract_concepts(self, context_text: str) -> list[str]:
        """Helper to extract concepts via LLM."""
        prompt_template = self._load_prompt("concept_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load concept prompt template. Skipping concept extraction.")
            return []

        concept_prompt = prompt_template.format(context_text=context_text)

        logger.info("Requesting concepts (from file prompt)...")
        # --- Use configured LLM call ---
        concepts_text = self._call_configured_llm('consolidation_concept', prompt=concept_prompt)
        llm_extracted_concepts = []
        if concepts_text and not concepts_text.startswith("Error:"): # Check for errors
            potential_concepts = concepts_text.split(',')
            for concept in potential_concepts:
                cleaned = concept.strip().strip('"').strip("'").strip()
                if cleaned and len(cleaned) > 2 and len(cleaned) < 80:
                    llm_extracted_concepts.append(cleaned)
                elif cleaned:
                    logger.debug(f"Filtered out potential LLM concept due to length/format: '{cleaned}'")
        if not llm_extracted_concepts:
             logger.warning("LLM returned no valid concepts after filtering.")
        return llm_extracted_concepts

    def _consolidate_extract_rich_relations(self, context_text: str, concept_node_map: dict, spacy_doc: object | None):
        """Helper to extract typed relationships between concepts using LLM and/or spaCy."""
        logger.info("Attempting Relationship Extraction...")
        current_time = time.time()
        added_llm_edge_count = 0
        added_spacy_edge_count = 0

        # --- Load Target Relation Types from Config ---
        consolidation_cfg = self.config.get('consolidation', {})
        target_relations = consolidation_cfg.get('target_relation_types',
                                                 ["CAUSES", "PART_OF", "HAS_PROPERTY", "RELATED_TO", "IS_A"]) # Default list
        if not target_relations:
             logger.warning("No target_relation_types defined in config. Using default set.")
             target_relations = ["CAUSES", "PART_OF", "HAS_PROPERTY", "RELATED_TO", "IS_A"]

        # --- spaCy Relation Extraction (if enabled and model loaded) ---
        if self.nlp and spacy_doc:
            logger.info("Extracting relations using spaCy dependencies...")
            # Invert concept_node_map for quick UUID lookup by text
            uuid_lookup = {uuid: text for text, uuid in concept_node_map.items()}

            for token in spacy_doc:
                # Look for Subject-Verb-Object patterns involving known concepts
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subj_text = token.text
                    verb_text = token.head.lemma_ # Use lemma for verb
                    obj_text = None
                    # Find direct object or prepositional object or attribute
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                            # Find the noun chunk the object belongs to, map to concept
                            obj_chunk = next((chunk for chunk in spacy_doc.noun_chunks if child in chunk), None)
                            obj_text = obj_chunk.text if obj_chunk else child.text
                            break # Take the first likely object

                    if subj_text and obj_text:
                        # Check if subject and object text match known concepts
                        subj_uuid = concept_node_map.get(subj_text)
                        obj_uuid = concept_node_map.get(obj_text)

                        # If direct match fails, check if token is *part* of a known concept text
                        if not subj_uuid:
                             subj_uuid = next((uuid for txt, uuid in concept_node_map.items() if subj_text in txt), None)
                        if not obj_uuid:
                             obj_uuid = next((uuid for txt, uuid in concept_node_map.items() if obj_text in txt), None)


                        if subj_uuid and obj_uuid and subj_uuid != obj_uuid:
                            # No status check needed before adding edge
                            try:
                                # Add edge if it doesn't exist or update timestamp
                                edge_type = f"SPACY_{verb_text.upper()}" # e.g., SPACY_USE, SPACY_BE
                                if not self.graph.has_edge(subj_uuid, obj_uuid) or self.graph.edges[subj_uuid, obj_uuid].get("type") != edge_type:
                                    base_strength = 0.5 # Lower base strength for spaCy relations?
                                    self.graph.add_edge(subj_uuid, obj_uuid, type=edge_type, base_strength=base_strength, last_traversed_ts=current_time)
                                    logger.info(f"Added spaCy Edge: {subj_uuid[:8]} --[{edge_type}]--> {obj_uuid[:8]} ('{subj_text}' -> '{obj_text}')")
                                    added_spacy_edge_count += 1
                                else:
                                    self.graph.edges[subj_uuid, obj_uuid]['last_traversed_ts'] = current_time
                            except Exception as e:
                                logger.error(f"Error adding spaCy edge {subj_uuid[:8]} -> {obj_uuid[:8]}: {e}")
                        # else: logger.warning(f"Skipping spaCy relation involving inactive node: {subj_uuid} or {obj_uuid}") # No longer needed

            logger.info(f"Added {added_spacy_edge_count} new spaCy-derived relationship edges.")
        elif self.nlp is None:
             logger.debug("spaCy model not loaded, skipping spaCy relation extraction.")


        # --- LLM Rich Relation Extraction (if enabled) ---
        if not self.config.get('features', {}).get('enable_rich_associations', False):
             logger.info("LLM Rich Relationship Extraction disabled by config.")
             return # Skip LLM part if disabled

        logger.info("Attempting Rich Relationship Extraction (LLM)...")
        # Use the target_relations list loaded from config
        target_relations_str = ", ".join([f"'{r}'" for r in target_relations])
        concept_list_str = "\n".join([f"- \"{c}\"" for c in concept_node_map.keys()])

        prompt_template = self._load_prompt("rich_relation_prompt.txt")
        if not prompt_template:
             logger.error("Failed to load rich relation prompt template. Skipping rich relation extraction.")
             return

        # Format the prompt template with current data
        rich_relation_prompt = prompt_template.format(
            target_relations_str=target_relations_str,
            concept_list_str=concept_list_str,
            context_text=context_text
        )

        logger.debug(f"Sending Rich Relation prompt (from file):\n{rich_relation_prompt}")
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('consolidation_relation', prompt=rich_relation_prompt)

        # ... (Rest of the parsing and edge adding logic remains the same) ...
        extracted_relations = []
        if llm_response_str:
            try:
                logger.debug(f"Raw Rich Relation response: ```{llm_response_str}```")
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

                start_bracket = cleaned_response.find('[')
                end_bracket = cleaned_response.rfind(']')

                if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                    json_str = cleaned_response[start_bracket:end_bracket + 1]
                    logger.debug(f"Extracted Rich Relation JSON string: {json_str}")
                    parsed_list = json.loads(json_str)
                    if isinstance(parsed_list, list):
                        extracted_relations = parsed_list
                        logger.info(f"Successfully parsed {len(extracted_relations)} relations from LLM.")
                    else:
                        logger.warning(f"LLM response was valid JSON but not a list. Raw: {llm_response_str}")
                else:
                    logger.warning(
                        f"Could not find valid JSON list '[]' in cleaned response. Cleaned: '{cleaned_response}' Raw: '{llm_response_str}'"
                    )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for rich relations: {e}. Raw: '{llm_response_str}'")
            except Exception as e:
                logger.error(f"Unexpected error processing rich relations response: {e}", exc_info=True)

        added_edge_count = 0
        if extracted_relations:
            current_time = time.time()
            for rel in extracted_relations:
                if isinstance(rel, dict) and all(k in rel for k in ["subject", "relation", "object"]):
                    subj_text = rel["subject"]
                    rel_type = rel["relation"]
                    obj_text = rel["object"]

                    # Validate against the loaded target_relations list
                    if rel_type not in target_relations:
                        logger.warning(f"Skipping relation with invalid type '{rel_type}' (not in configured list): {rel}")
                        continue

                    subj_uuid = concept_node_map.get(subj_text)
                    obj_uuid = concept_node_map.get(obj_text)

                    if subj_uuid and obj_uuid and subj_uuid in self.graph and obj_uuid in self.graph:
                        # No status check needed before adding edge
                        try:
                            if not self.graph.has_edge(subj_uuid, obj_uuid) or self.graph.edges[subj_uuid, obj_uuid].get("type") != rel_type:
                                base_strength = 0.7
                                self.graph.add_edge(
                                    subj_uuid,
                                    obj_uuid,
                                    type=rel_type,
                                    base_strength=base_strength,
                                    last_traversed_ts=current_time,
                                )
                                logger.info(
                                    f"Added Edge: {subj_uuid[:8]} --[{rel_type}]--> {obj_uuid[:8]} ('{subj_text}' -> '{obj_text}')"
                                )
                                added_llm_edge_count += 1
                            else:
                                # Update timestamp even if edge exists, but maybe don't overwrite type?
                                # For now, update timestamp regardless.
                                    self.graph.edges[subj_uuid, obj_uuid]["last_traversed_ts"] = current_time
                                    logger.debug(
                                        f"LLM Edge {subj_uuid[:8]} --[{rel_type}]--> {obj_uuid[:8]} already exists. Updated timestamp."
                                    )
                        except Exception as e:
                            logger.error(f"Error adding LLM typed edge {subj_uuid[:8]} -> {obj_uuid[:8]}: {e}")
                        # else: logger.warning(f"Skipping LLM relation '{rel}' because one or both nodes are not active.") # No longer needed
                    else: # One or both concepts not found in map or graph
                        logger.warning(
                            f"Could not find nodes in map/graph for LLM relation: Subject='{subj_text}' ({subj_uuid}), Object='{obj_text}' ({obj_uuid})"
                        )
                else:
                    logger.warning(f"Skipping invalid LLM relation object in list: {rel}")
            logger.info(f"Added {added_llm_edge_count} new LLM-derived typed relationship edges.")

    def _load_prompt(self, filename: str) -> str:
        """Loads a prompt template from the prompts directory."""
        # Assumes a 'prompts' subdirectory relative to this script's location
        # Or adjust path logic as needed (e.g., relative to config?)
        # For simplicity, let's assume prompts are relative to the main script dir
        # If base_memory_path is reliable, maybe put prompts there?
        # Let's assume relative to the script for now.
        script_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(script_dir, "prompts", filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            return ""  # Return empty string or raise error? Empty string is safer.
        except Exception as e:
            logger.error(f"Error loading prompt file {prompt_path}: {e}", exc_info=True)
            return ""

    def _consolidate_extract_v1_associative(self, concept_node_map: dict):
        """Helper to extract simple 'ASSOCIATIVE' relations via LLM."""
        concept_list_str = "\n".join([f"- \"{c}\"" for c in concept_node_map.keys()])

        prompt_template = self._load_prompt("v1_assoc_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load V1 associative prompt template. Skipping associative link extraction.")
            return

        relation_prompt = prompt_template.format(concept_list_str=concept_list_str)

        logger.info("Requesting concept relationships (V1 - Associative, from file prompt)...")
        # --- Use configured LLM call ---
        relations_text = self._call_configured_llm('consolidation_associative_v1', prompt=relation_prompt)
        # ... (Rest of the parsing and edge adding logic remains the same) ...
        if relations_text and relations_text.strip().upper() != "NONE":
            logger.info(f"Found potential associative relationships:\n{relations_text}")
            current_time = time.time()
            lines = relations_text.strip().split('\n')
            for line in lines:
                match = re.match(r'^\s*-\s*"(.+?)"\s*->\s*"(.+?)"\s*$', line)
                if match:
                    c1_txt, c2_txt = match.group(1), match.group(2)
                    uuid1, uuid2 = concept_node_map.get(c1_txt), concept_node_map.get(c2_txt)
                    if (
                            uuid1 and uuid2
                            and uuid1 in self.graph
                            # and self.graph.nodes[uuid1].get('status', 'active') == 'active' # Status removed
                            and uuid2 in self.graph
                            # and self.graph.nodes[uuid2].get('status', 'active') == 'active' # Status removed
                    ):
                        try:
                            if not self.graph.has_edge(uuid1, uuid2):
                                self.graph.add_edge(
                                    uuid1, uuid2, type='ASSOCIATIVE', base_strength=0.6, last_traversed_ts=current_time
                                )
                                logger.info(f"Added ASSOC edge: {uuid1[:8]} ('{c1_txt}') -> {uuid2[:8]} ('{c2_txt}')")
                            else:
                                self.graph.edges[uuid1, uuid2]['last_traversed_ts'] = current_time
                                logger.debug(f"Assoc edge {uuid1[:8]}->{uuid2[:8]} exists. Updated timestamp.")
                        except Exception as e:
                            logger.error(f"Error adding/updating assoc edge: {e}")
                    else:
                        logger.warning(
                            f"Could not find nodes in graph for relation: '{c1_txt}' -> '{c2_txt}' (UUIDs: {uuid1}, {uuid2})"
                        )
                else:
                    logger.debug(f"Line did not match V1 associative format: '{line}'")
        else:
            logger.info("LLM reported no direct associative relationships.")

    def _consolidate_extract_hierarchy(self, concept_node_map: dict):
        """Helper to extract hierarchical relationships via LLM."""
        concept_list_str = "\n".join([f"- \"{c}\"" for c in concept_node_map.keys()])

        prompt_template = self._load_prompt("hierarchy_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load hierarchy prompt template. Skipping hierarchy extraction.")
            return

        hierarchy_prompt = prompt_template.format(concept_list_str=concept_list_str)

        logger.info("Requesting concept hierarchy (from file prompt)...")
        # --- Use configured LLM call ---
        hierarchy_text = self._call_configured_llm('consolidation_hierarchy_v1', prompt=hierarchy_prompt)
        # ... (Rest of the parsing and edge adding logic remains the same, including the regex fix) ...
        if hierarchy_text and hierarchy_text.strip().upper() != "NONE":
            logger.info(f"Found potential hierarchies:\n{hierarchy_text}")
            current_time = time.time()
            lines = hierarchy_text.strip().split('\n')
            for line in lines:
                match = re.match(
                    r'^\s*"(.+?)"\s+(is(?: |_)?a(?: |_type(?: |_of)?)?|is(?: |_)?part(?: |_of)?)\s+"(.+?)"\s*$',
                    line, re.IGNORECASE)
                if match:
                    child_text = match.group(1)
                    relation_type_str = match.group(2).lower()
                    parent_text = match.group(3)
                    child_uuid = concept_node_map.get(child_text)
                    parent_uuid = concept_node_map.get(parent_text)

                    if not parent_uuid and len(parent_text) < 80:
                        logger.info(f"Identified potential new parent concept: '{parent_text}'. Checking existing...")
                        concept_sim_threshold = self.config.get('consolidation', {}).get('concept_similarity_threshold',
                                                                                         0.3)
                        similar_parents = self._search_similar_nodes(parent_text, k=1, node_type_filter='concept')
                        if similar_parents and similar_parents[0][1] <= concept_sim_threshold:
                            parent_uuid = similar_parents[0][0]
                            logger.debug(f"Found existing node for parent: {parent_uuid[:8]}")
                            if parent_text not in concept_node_map:
                                concept_node_map[parent_text] = parent_uuid
                        else:
                            logger.info(f"Adding new node for parent '{parent_text}'")
                            parent_uuid = self.add_memory_node(parent_text, "System", 'concept', base_strength=0.85)
                            if parent_uuid:
                                if parent_text not in concept_node_map:
                                    concept_node_map[parent_text] = parent_uuid
                            else:
                                logger.warning(
                                    f"Failed to add new parent node '{parent_text}'. Skipping hierarchy link.")
                                parent_uuid = None

                    if (
                            child_uuid and parent_uuid
                            and child_uuid in self.graph
                            and parent_uuid in self.graph
                    ):
                        # No status check needed before adding edge
                        try:
                            if not self.graph.has_edge(parent_uuid, child_uuid):
                                self.graph.add_edge(parent_uuid, child_uuid, type='HIERARCHICAL', base_strength=0.85,
                                                    last_traversed_ts=current_time)
                                logger.info(
                                    f"Added HIERARCHICAL edge: {parent_uuid[:8]} ('{parent_text}') -> {child_uuid[:8]} ('{child_text}')")
                            else:
                                self.graph.edges[parent_uuid, child_uuid]['last_traversed_ts'] = current_time
                                logger.debug(
                                    f"Hierarchical edge {parent_uuid[:8]}->{child_uuid[:8]} exists. Updated timestamp.")
                        except Exception as e:
                            logger.error(f"Error adding/updating hierarchical edge: {e}")
                    else:
                        logger.warning(
                            f"Could not find nodes in graph for hierarchy: Child='{child_text}' ({child_uuid}), Parent='{parent_text}' ({parent_uuid})")
                else:
                    logger.debug(f"Line did not match hierarchy format: '{line}'")
        else:
            logger.info("LLM reported no direct hierarchical relationships.")

    def _consolidate_extract_causal_chains(self, context_text: str, concept_node_map: dict):
        """Helper to extract causal chains (A causes B causes C) via LLM."""
        if not self.config.get('consolidation', {}).get('enable_causal_chains', False):
            logger.info("Causal chain extraction disabled by config.")
            return 0 # Return 0 added edges

        logger.info("Attempting Causal Chain Extraction (LLM)...")
        concept_list_str = "\n".join([f"- \"{c}\"" for c in concept_node_map.keys()])
        prompt_template = self._load_prompt("causal_chain_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load causal chain prompt template.")
            return 0

        causal_chain_prompt = prompt_template.format(
            concept_list_str=concept_list_str,
            context_text=context_text
        )

        logger.debug(f"Sending Causal Chain prompt:\n{causal_chain_prompt}")
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('consolidation_causal', prompt=causal_chain_prompt)

        extracted_chains = []
        if llm_response_str:
            try:
                logger.debug(f"Raw Causal Chain response: ```{llm_response_str}```")
                # --- Improved JSON Extraction ---
                cleaned_response = llm_response_str.strip()
                # Remove potential markdown fences first
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

                # Find the first '[' and the last ']'
                start_bracket = cleaned_response.find('[')
                end_bracket = cleaned_response.rfind(']')

                if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                    json_str = cleaned_response[start_bracket:end_bracket + 1]
                    logger.debug(f"Extracted potential JSON list string: {json_str}")
                    parsed_list = json.loads(json_str) # Attempt to parse the extracted string

                    if isinstance(parsed_list, list):
                        # Validate inner lists (Keep existing validation)
                        valid_chains = []
                        for chain in parsed_list:
                            # Validate format: list of strings, length >= 2
                            if isinstance(chain, list) and len(chain) >= 2 and all(isinstance(item, str) for item in chain):
                                # CRITICAL VALIDATION: Check if ALL concepts in the chain exist in the provided map
                                if all(item in concept_node_map for item in chain):
                                    valid_chains.append(chain)
                                else:
                                    unknown_concepts = [item for item in chain if item not in concept_node_map]
                                    logger.warning(f"Skipping chain with unknown/unprovided concepts: {chain} (Unknown: {unknown_concepts})")
                            else:
                                logger.warning(f"Skipping invalid chain format: {chain}")
                        extracted_chains = valid_chains
                        logger.info(f"Successfully parsed {len(extracted_chains)} valid causal chains from LLM.")
                    else:
                        logger.warning(f"LLM response was valid JSON but not a list. Raw: {llm_response_str}")
                else:
                    logger.warning(f"Could not extract valid JSON list '[]' from causal chain response. Raw: '{llm_response_str}'")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for causal chains: {e}. Raw: '{llm_response_str}'")
            except Exception as e:
                logger.error(f"Unexpected error processing causal chains response: {e}", exc_info=True)

        added_edge_count = 0
        if extracted_chains:
            current_time = time.time()
            for chain in extracted_chains:
                # Add CAUSES edges between consecutive elements in the chain
                for i in range(len(chain) - 1):
                    cause_text = chain[i]
                    effect_text = chain[i+1]
                    cause_uuid = concept_node_map.get(cause_text)
                    effect_uuid = concept_node_map.get(effect_text)

                    # Should always have UUIDs due to validation above, but check anyway
                    if cause_uuid and effect_uuid and cause_uuid in self.graph and effect_uuid in self.graph:
                        try:
                            # Add edge if it doesn't exist, or update timestamp if it does
                            edge_type = "CAUSES"
                            if not self.graph.has_edge(cause_uuid, effect_uuid) or self.graph.edges[cause_uuid, effect_uuid].get("type") != edge_type:
                                base_strength = 0.75 # Slightly higher strength for causal links?
                                self.graph.add_edge(
                                    cause_uuid, effect_uuid,
                                    type=edge_type,
                                    base_strength=base_strength,
                                    last_traversed_ts=current_time
                                )
                                logger.info(f"Added Causal Edge: {cause_uuid[:8]} --[{edge_type}]--> {effect_uuid[:8]} ('{cause_text}' -> '{effect_text}')")
                                added_edge_count += 1
                            else:
                                self.graph.edges[cause_uuid, effect_uuid]["last_traversed_ts"] = current_time
                                logger.debug(f"Causal edge {cause_uuid[:8]}->{effect_uuid[:8]} exists. Updated timestamp.")
                        except Exception as e:
                            logger.error(f"Error adding causal edge {cause_uuid[:8]} -> {effect_uuid[:8]}: {e}")
                    else:
                        # This shouldn't happen if validation worked
                        logger.error(f"Could not find nodes for validated causal link: '{cause_text}' -> '{effect_text}'")

            logger.info(f"Added {added_edge_count} new CAUSES edges from causal chains.")
        return added_edge_count

    def _consolidate_extract_analogies(self, context_text: str, concept_node_map: dict):
        """Helper to extract analogies (A is like B) between concepts via LLM."""
        if not self.config.get('consolidation', {}).get('enable_analogies', False):
            logger.info("Analogy extraction disabled by config.")
            return 0 # Return 0 added edges

        logger.info("Attempting Analogy Extraction (LLM)...")
        concept_list_str = "\n".join([f"- \"{c}\"" for c in concept_node_map.keys()])
        prompt_template = self._load_prompt("analogy_extraction_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load analogy extraction prompt template.")
            return 0

        analogy_prompt = prompt_template.format(
            concept_list_str=concept_list_str,
            context_text=context_text
        )

        logger.debug(f"Sending Analogy prompt:\n{analogy_prompt}")
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('consolidation_analogy', prompt=analogy_prompt)

        extracted_analogies = []
        if llm_response_str:
            try:
                logger.debug(f"Raw Analogy response: ```{llm_response_str}```")
                # Extract JSON list
                match = re.search(r'(\[.*?\])', llm_response_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    parsed_list = json.loads(json_str)
                    if isinstance(parsed_list, list):
                        # Validate inner lists
                        valid_analogies = []
                        for pair in parsed_list:
                            if isinstance(pair, list) and len(pair) == 2 and all(isinstance(item, str) for item in pair):
                                # Check if both concepts in the pair are known
                                if all(item in concept_node_map for item in pair):
                                    valid_analogies.append(pair)
                                else:
                                    logger.warning(f"Skipping analogy with unknown concepts: {pair}")
                            else:
                                logger.warning(f"Skipping invalid analogy format (must be list of 2 strings): {pair}")
                        extracted_analogies = valid_analogies
                        logger.info(f"Successfully parsed {len(extracted_analogies)} valid analogies from LLM.")
                    else:
                        logger.warning(f"LLM response was valid JSON but not a list. Raw: {llm_response_str}")
                else:
                    logger.warning(f"Could not extract valid JSON list '[]' from analogy response. Raw: '{llm_response_str}'")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for analogies: {e}. Raw: '{llm_response_str}'")
            except Exception as e:
                logger.error(f"Unexpected error processing analogies response: {e}", exc_info=True)

        added_edge_count = 0
        if extracted_analogies:
            current_time = time.time()
            for analogy_pair in extracted_analogies:
                concept_a_text = analogy_pair[0]
                concept_b_text = analogy_pair[1]
                uuid_a = concept_node_map.get(concept_a_text)
                uuid_b = concept_node_map.get(concept_b_text)

                # Should always have UUIDs due to validation, but check
                if uuid_a and uuid_b and uuid_a in self.graph and uuid_b in self.graph:
                    try:
                        # Add edge if it doesn't exist (direction doesn't strictly matter for analogy)
                        # Let's add A->B for consistency
                        edge_type = "ANALOGY"
                        if not self.graph.has_edge(uuid_a, uuid_b): # Check only one direction for adding
                            base_strength = 0.65 # Moderate strength for analogy
                            self.graph.add_edge(
                                uuid_a, uuid_b,
                                type=edge_type,
                                base_strength=base_strength,
                                last_traversed_ts=current_time
                            )
                            logger.info(f"Added Analogy Edge: {uuid_a[:8]} --[{edge_type}]--> {uuid_b[:8]} ('{concept_a_text}' like '{concept_b_text}')")
                            added_edge_count += 1
                        # Update timestamp if edge exists? Maybe not necessary for analogy.
                        # else:
                        #     self.graph.edges[uuid_a, uuid_b]["last_traversed_ts"] = current_time
                        #     logger.debug(f"Analogy edge {uuid_a[:8]}->{uuid_b[:8]} exists. Updated timestamp.")
                    except Exception as e:
                        logger.error(f"Error adding analogy edge {uuid_a[:8]} -> {uuid_b[:8]}: {e}")
                else:
                    logger.error(f"Could not find nodes for validated analogy: '{concept_a_text}' -> '{concept_b_text}'")

            logger.info(f"Added {added_edge_count} new ANALOGY edges.")
        return added_edge_count


    def _generate_autobiographical_model(self):
        """Analyzes key memories and updates the ASM via LLM."""
        logger.info("--- Generating Autobiographical Self-Model (ASM) ---")
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("ASM generation skipped: Graph is empty.")
            return

        # --- Configuration ---
        asm_cfg = self.config.get('autobiographical_model', {}) # Add section to config later if needed
        num_salient_nodes = asm_cfg.get('num_salient_nodes', 10)
        num_emotional_nodes = asm_cfg.get('num_emotional_nodes', 10)
        max_context_nodes = asm_cfg.get('max_context_nodes', 15) # Limit total nodes for prompt

        # --- Select Key Nodes ---
        key_nodes_data = []
        node_uuids_added = set()

        # 1. Get Top N Salient Nodes
        try:
            salient_nodes = sorted(
                [(uuid, data.get('saliency_score', 0.0)) for uuid, data in self.graph.nodes(data=True)],
                key=lambda item: item[1], reverse=True
            )
            for uuid, score in salient_nodes[:num_salient_nodes]:
                if uuid not in node_uuids_added:
                    node_data = self.graph.nodes[uuid]
                    key_nodes_data.append(node_data)
                    node_uuids_added.add(uuid)
                    # logger.debug(f"ASM Candidate (Salient): {uuid[:8]} (Score: {score:.3f})")
        except Exception as e:
            logger.error(f"Error selecting salient nodes for ASM: {e}", exc_info=True)

        # 2. Get Top N Emotional Nodes (by magnitude)
        try:
            emotional_nodes = sorted(
                [(uuid, math.sqrt(data.get('emotion_valence', 0.0)**2 + data.get('emotion_arousal', 0.1)**2))
                 for uuid, data in self.graph.nodes(data=True)],
                key=lambda item: item[1], reverse=True
            )
            for uuid, magnitude in emotional_nodes[:num_emotional_nodes]:
                if uuid not in node_uuids_added:
                    node_data = self.graph.nodes[uuid]
                    key_nodes_data.append(node_data)
                    node_uuids_added.add(uuid)
                    # logger.debug(f"ASM Candidate (Emotional): {uuid[:8]} (Mag: {magnitude:.3f})")
        except Exception as e:
            logger.error(f"Error selecting emotional nodes for ASM: {e}", exc_info=True)

        if not key_nodes_data:
            logger.warning("ASM generation skipped: No key nodes identified.")
            return

        # --- Prepare Context ---
        # Sort by timestamp and limit total nodes
        key_nodes_data.sort(key=lambda x: x.get('timestamp', ''))
        context_nodes = key_nodes_data[:max_context_nodes]
        context_text = "\n".join([f"{d.get('speaker', '?')} ({self._get_relative_time_desc(d.get('timestamp',''))}): {d.get('text', '')}" for d in context_nodes])
        logger.debug(f"ASM generation context (from {len(context_nodes)} nodes):\n{context_text[:300]}...")

        # --- Call LLM ---
        prompt_template = self._load_prompt("asm_generation_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load ASM generation prompt template.")
            return

        full_prompt = prompt_template.format(context_text=context_text)
        # --- Use configured LLM call ---
        llm_response_str = self._call_configured_llm('asm_generation', prompt=full_prompt)

        # --- Parse Response and Update Model ---
        if llm_response_str:
            try:
                logger.debug(f"Raw ASM response: ```{llm_response_str}```")
                # --- Improved JSON Extraction ---
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

                start_brace = cleaned_response.find('{')
                end_brace = cleaned_response.rfind('}')

                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    json_str = cleaned_response[start_brace:end_brace + 1]
                    logger.debug(f"Extracted potential JSON object string: {json_str}")
                    parsed_data = json.loads(json_str)

                    # --- Validate Structured ASM ---
                    if isinstance(parsed_data, dict):
                        required_keys = ["core_traits", "recurring_themes", "significant_events", "values_beliefs", "summary_statement"]
                        if all(key in parsed_data for key in required_keys):
                            # Basic type validation (can be expanded)
                            if (isinstance(parsed_data["core_traits"], list) and
                                isinstance(parsed_data["recurring_themes"], list) and
                                isinstance(parsed_data["significant_events"], list) and
                                isinstance(parsed_data["values_beliefs"], list) and
                                isinstance(parsed_data["summary_statement"], str)):

                                # Update the entire model
                                self.autobiographical_model = parsed_data
                                self.autobiographical_model["last_updated"] = datetime.now(timezone.utc).isoformat()
                                logger.info(f"Structured Autobiographical Self-Model updated. Summary: '{self.autobiographical_model.get('summary_statement', '')[:100]}...'")
                                self._save_memory() # Save the updated model
                            else:
                                logger.warning("LLM response JSON had correct keys but incorrect value types for ASM.")
                        else:
                            logger.warning(f"LLM response JSON missing required keys for structured ASM. Keys found: {list(parsed_data.keys())}")
                    else:
                        logger.warning("LLM response was not a valid JSON dictionary for structured ASM.")
                else:
                    logger.warning(f"Could not extract valid JSON object from ASM response. Raw: '{llm_response_str}'")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for ASM: {e}. Raw: '{llm_response_str}'")
            except Exception as e:
                logger.error(f"Unexpected error processing ASM response: {e}", exc_info=True)
        else:
            logger.error("LLM call failed for ASM generation (empty response).")

        logger.info("--- ASM Generation Finished ---")

    def _infer_second_order_relations(self):
        """Infers generic 'INFERRED_RELATED_TO' edges based on paths of length 2."""
        inference_cfg = self.config.get('consolidation', {}).get('inference', {})
        if not inference_cfg.get('enable', False):
            logger.info("Second-order inference disabled by config.")
            return

        logger.info("--- Inferring Second-Order Relationships ---")
        strength_factor = inference_cfg.get('strength_factor', 0.3)
        max_depth = 2 # Fixed for V1
        inferred_edge_count = 0
        current_time = time.time()

        # Consider only concept and summary nodes as start/end/intermediate points for V1
        candidate_nodes = [
            uuid for uuid, data in self.graph.nodes(data=True)
            if data.get('node_type') in ['concept', 'summary']
        ]

        if len(candidate_nodes) < 3:
            logger.info("Skipping inference: Not enough concept/summary nodes.")
            return

        logger.debug(f"Checking {len(candidate_nodes)} candidate nodes for inference...")

        # Iterate through all pairs of candidate nodes (A, C)
        for node_a_uuid in candidate_nodes:
            for node_c_uuid in candidate_nodes:
                if node_a_uuid == node_c_uuid: continue

                # Check if a direct edge already exists (A->C or C->A)
                if self.graph.has_edge(node_a_uuid, node_c_uuid) or self.graph.has_edge(node_c_uuid, node_a_uuid):
                    continue

                # Find paths of length 2 (A -> B -> C) where B is also a candidate node
                found_path = False
                path_strength_sum = 0.0
                path_count = 0

                # Check A -> B edges
                for _, node_b_uuid, edge_ab_data in self.graph.out_edges(node_a_uuid, data=True):
                    if node_b_uuid in candidate_nodes: # Check if intermediate is concept/summary
                        # Check B -> C edges
                        if self.graph.has_edge(node_b_uuid, node_c_uuid):
                            edge_bc_data = self.graph.get_edge_data(node_b_uuid, node_c_uuid)
                            if edge_bc_data:
                                found_path = True
                                # Calculate path strength (e.g., product of edge strengths * factor)
                                strength_ab = edge_ab_data.get('base_strength', 0.5) # Use base_strength for calculation
                                strength_bc = edge_bc_data.get('base_strength', 0.5)
                                path_strength = strength_ab * strength_bc * strength_factor
                                path_strength_sum += path_strength
                                path_count += 1
                                logger.debug(f"  Found path: {node_a_uuid[:4]}->{node_b_uuid[:4]}->{node_c_uuid[:4]} (Strength: {path_strength:.3f})")

                if found_path:
                    # Calculate average strength if multiple paths exist? Or max? Let's use average.
                    avg_strength = path_strength_sum / path_count if path_count > 0 else 0.0
                    clamped_strength = max(0.01, min(1.0, avg_strength)) # Ensure minimum strength, clamp max

                    # Add the inferred edge
                    try:
                        self.graph.add_edge(
                            node_a_uuid, node_c_uuid,
                            type='INFERRED_RELATED_TO',
                            base_strength=clamped_strength,
                            last_traversed_ts=current_time # Set timestamp to now
                        )
                        inferred_edge_count += 1
                        logger.info(f"Added inferred edge: {node_a_uuid[:8]} --[INFERRED_RELATED_TO ({clamped_strength:.3f})]--> {node_c_uuid[:8]}")
                    except Exception as e:
                        logger.error(f"Error adding inferred edge {node_a_uuid[:8]} -> {node_c_uuid[:8]}: {e}")

        if inferred_edge_count > 0:
            logger.info(f"Added {inferred_edge_count} new inferred relationship edges.")
            self._save_memory() # Save graph if changes were made
        else:
            logger.info("No new second-order relationships were inferred.")


    def run_consolidation(self, active_nodes_to_process=None):
        """
        Orchestrates the memory consolidation process: summarization, concept extraction,
        relation extraction, and pruning.
        """
        logger.info("--- Running Consolidation ---")
        # --- Tuning Log: Consolidation Start ---
        log_tuning_event("CONSOLIDATION_START", {"personality": self.personality})

        consolidation_cfg = self.config.get('consolidation', {})
        min_nodes_for_consolidation = consolidation_cfg.get('min_nodes', 5)
        turn_count_for_consolidation = consolidation_cfg.get('turn_count', 10)
        prune_summarized = consolidation_cfg.get('prune_summarized_turns', True)
        concept_sim_threshold = consolidation_cfg.get('concept_similarity_threshold', 0.3)
        features_cfg = self.config.get('features', {})
        rich_assoc_enabled = features_cfg.get('enable_rich_associations', False)
        emotion_analysis_enabled = features_cfg.get('enable_emotion_analysis',
                                                    False)

        # --- 1. Select Nodes ---
        nodes_to_consolidate = self._select_nodes_for_consolidation(count=turn_count_for_consolidation)
        # Filter out nodes that are already summarized or not 'turn' nodes
        nodes_to_process = []
        nodes_data = []
        for uuid in nodes_to_consolidate:
            if uuid in self.graph:
                node_data = self.graph.nodes[uuid]
                # Check if it's a 'turn' node and not already linked FROM a summary
                is_summarized = any(
                    True for pred, _, data in self.graph.in_edges(uuid, data=True) if data.get('type') == 'SUMMARY_OF')
                if node_data.get('node_type') == 'turn' and not is_summarized:
                    nodes_to_process.append(uuid)
                    nodes_data.append(node_data)
                else:
                    logger.debug(
                        f"Skipping node {uuid[:8]} for consolidation (Type: {node_data.get('node_type')}, Summarized: {is_summarized})")

        if len(nodes_to_process) < min_nodes_for_consolidation:
            logger.info(
                f"Consolidation skipped: Only {len(nodes_to_process)} suitable nodes found (min: {min_nodes_for_consolidation}).")
            # --- Tuning Log: Consolidation End (Skipped) ---
            log_tuning_event("CONSOLIDATION_END", {
                "personality": self.personality,
                "status": "skipped_min_nodes",
                "nodes_processed_count": len(nodes_to_process),
                "min_nodes_required": min_nodes_for_consolidation,
            })
            return

        logger.info(f"Consolidating {len(nodes_to_process)} nodes: {nodes_to_process}")
        # --- Tuning Log: Consolidation Nodes Selected ---
        log_tuning_event("CONSOLIDATION_NODES_SELECTED", {
            "personality": self.personality,
            "selected_node_uuids": nodes_to_process,
        })


        # --- 2. Prepare Context ---
        nodes_data.sort(key=lambda x: x.get('timestamp', ''))  # Ensure chronological order for context
        context_text = "\n".join([f"{d.get('speaker', '?')}: {d.get('text', '')}" for d in nodes_data])
        logger.debug(f"Consolidation context text (first 200 chars):\n{context_text[:200]}...")

        # --- 3. Summarization ---
        summary_node_uuid, summary_created = self._consolidate_summarize(context_text=context_text,
                                                                         nodes_data=nodes_data,
                                                                         processed_node_uuids=nodes_to_process) # Pass the list of UUIDs
        # --- Tuning Log: Consolidation Summary ---
        log_tuning_event("CONSOLIDATION_SUMMARY", {
            "personality": self.personality,
            "summary_created": summary_created,
            "summary_node_uuid": summary_node_uuid,
            "source_node_uuids": nodes_to_process,
        })

        # --- 4. Concept Extraction (LLM) ---
        llm_concepts = self._consolidate_extract_concepts(context_text)
        # --- Tuning Log: Consolidation Concepts ---
        log_tuning_event("CONSOLIDATION_CONCEPTS_EXTRACTED", {
            "personality": self.personality,
            "llm_extracted_concepts": llm_concepts,
        })

        # --- 5. Concept Deduplication & Node Management ---
        concept_node_map = {}  # Map: concept_text -> concept_node_uuid
        newly_added_concepts = []
        processed_llm_concepts = set()  # Track concepts processed from LLM list

        if llm_concepts:
            logger.info(f"LLM Concepts to process: {llm_concepts}")
            for concept_text in llm_concepts:
                if concept_text in processed_llm_concepts: continue  # Avoid processing duplicates from LLM list itself
                processed_llm_concepts.add(concept_text)
                logger.debug(f"Processing LLM concept: '{concept_text}'")
                # Search for existing similar 'concept' nodes
                similar_concepts = self._search_similar_nodes(concept_text, k=1, node_type_filter='concept')

                existing_uuid = None
                if similar_concepts:
                    best_match_uuid, best_match_score = similar_concepts[0]
                    # No status check needed here, just check score
                    if best_match_score <= concept_sim_threshold:
                        existing_uuid = best_match_uuid
                        logger.info(
                            f"Concept '{concept_text}' matches existing node {existing_uuid[:8]} (Score: {best_match_score:.3f})")
                    else:
                        logger.debug(
                            f"Found similar concept {best_match_uuid[:8]}, but score ({best_match_score:.3f}) > threshold ({concept_sim_threshold}).")

                if existing_uuid:
                    concept_node_map[concept_text] = existing_uuid
                    # Update access time of existing concept
                    self.graph.nodes[existing_uuid]['last_accessed_ts'] = time.time()
                    # Optionally boost activation/saliency? Not implemented here.
                else:
                    logger.info(f"Adding new concept node for: '{concept_text}'")
                    new_concept_uuid = self.add_memory_node(concept_text, "System", 'concept', base_strength=0.85)
                    if new_concept_uuid:
                        concept_node_map[concept_text] = new_concept_uuid
                        newly_added_concepts.append(new_concept_uuid)
                    else:
                        logger.error(f"Failed to add new concept node for '{concept_text}'")
            logger.info(
                f"Processed LLM concepts. Map size: {len(concept_node_map)}. New concepts added: {len(newly_added_concepts)}")
            # --- Tuning Log: Consolidation Concepts Processed ---
            log_tuning_event("CONSOLIDATION_CONCEPTS_PROCESSED", {
                "personality": self.personality,
                "final_concept_map": concept_node_map, # text -> uuid
                "newly_added_concept_uuids": newly_added_concepts,
            })
        else:
            logger.info("No concepts extracted by LLM.")

        # --- 6. Link Concepts to Source Nodes ---
        if concept_node_map:
            current_time = time.time()
            for concept_text, concept_uuid in concept_node_map.items():
                if concept_uuid not in self.graph: continue  # Should not happen, but safety check
                # Link concept to the summary node if created
                if summary_node_uuid and summary_node_uuid in self.graph:
                    try:
                        if not self.graph.has_edge(summary_node_uuid, concept_uuid):
                            self.graph.add_edge(summary_node_uuid, concept_uuid, type='MENTIONS_CONCEPT',
                                                base_strength=0.7, last_traversed_ts=current_time)
                    except Exception as e:
                        logger.error(
                            f"Error adding MENTIONS_CONCEPT edge from summary {summary_node_uuid[:8]} to {concept_uuid[:8]}: {e}")

                # Link concept to original turn nodes where it might appear (less precise)
                for node_uuid in nodes_to_process:
                    if node_uuid in self.graph and concept_text.lower() in self.graph.nodes[node_uuid].get('text',
                                                                                                           '').lower():
                        try:
                            if not self.graph.has_edge(node_uuid, concept_uuid):
                                self.graph.add_edge(node_uuid, concept_uuid, type='MENTIONS_CONCEPT', base_strength=0.7,
                                                    last_traversed_ts=current_time)
                                logger.debug(f"Added MENTIONS_CONCEPT edge: {node_uuid[:8]} -> {concept_uuid[:8]}")
                            else:  # Update timestamp if edge exists
                                self.graph.edges[node_uuid, concept_uuid]['last_traversed_ts'] = current_time
                                logger.debug(f"Updated MENTIONS_CONCEPT edge timestamp: {node_uuid[:8]} -> {concept_uuid[:8]}")
                        except Exception as e:
                            logger.error(
                                f"Error adding/updating MENTIONS_CONCEPT edge from turn {node_uuid[:8]} to {concept_uuid[:8]}: {e}")

        # --- 7. Relation Extraction ---
        # (Only run if we actually identified/created concepts)
        if concept_node_map:
            spacy_doc = None  # Initialize spacy_doc
            if rich_assoc_enabled and self.nlp:
                try:
                    logger.info("Using spaCy for potential pre-processing of context...")
                    spacy_doc = self.nlp(context_text)
                    # Placeholder: Could extract entities/dependencies here if needed by rich relation prompt later
                except Exception as spacy_err:
                    logger.error(f"Error processing context with spaCy: {spacy_err}. Falling back.", exc_info=True)
                    # Fallback to LLM-only methods if spaCy fails
                    self._consolidate_extract_v1_associative(concept_node_map)
                    self._consolidate_extract_hierarchy(concept_node_map)
                    # Attempt causal chain extraction even if spaCy failed
                    self._consolidate_extract_causal_chains(context_text, concept_node_map)
                else:
                    # If spaCy succeeded, proceed with rich relation extraction
                    self._consolidate_extract_rich_relations(context_text, concept_node_map, spacy_doc)
                    # Also attempt causal chain extraction if spaCy was used
                    self._consolidate_extract_causal_chains(context_text, concept_node_map)
                    # Attempt analogy extraction
                    self._consolidate_extract_analogies(context_text, concept_node_map)

            else:  # Rich associations disabled or spaCy unavailable/failed earlier
                if not self.nlp and rich_assoc_enabled:
                    logger.warning(
                        "Rich associations enabled but spaCy model not loaded. Falling back to V1 relation extraction.")
                logger.info("Running V1 Associative and Hierarchy extraction (LLM only).")
                self._consolidate_extract_v1_associative(concept_node_map)
                self._consolidate_extract_hierarchy(concept_node_map)
                # Attempt causal chain extraction even if rich associations are off
                self._consolidate_extract_causal_chains(context_text, concept_node_map)
                # Attempt analogy extraction
                self._consolidate_extract_analogies(context_text, concept_node_map)
        else:
            logger.info("Skipping relation extraction as no concepts were identified.")
            # --- Tuning Log: Consolidation Relations Skipped ---
            log_tuning_event("CONSOLIDATION_RELATIONS_SKIPPED", {
                "personality": self.personality,
                "reason": "no_concepts_identified",
            })

        # --- 7b. Emotion Analysis (Optional) ---
        if emotion_analysis_enabled and te:
            logger.info("Running V1 Emotion Analysis on consolidated nodes...")
            nodes_for_emotion = set(nodes_to_process)
            if summary_node_uuid: nodes_for_emotion.add(summary_node_uuid)
            nodes_for_emotion.update(concept_node_map.values())

            emotion_analyzed_count = 0
            for node_uuid in nodes_for_emotion:
                 if node_uuid in self.graph:
                      self._analyze_and_update_emotion(node_uuid)
                      emotion_analyzed_count += 1
            logger.info(f"Emotion analysis attempted for {emotion_analyzed_count} nodes.")
        elif emotion_analysis_enabled and not te:
             logger.warning("Emotion analysis enabled but text2emotion library not loaded.")


        # --- 8. Pruning Summarized Nodes (Optional) ---
        if prune_summarized and summary_created:
            logger.info("Pruning original turn nodes that were summarized...")
            pruned_count = 0
            for uuid_to_prune in nodes_to_process:
                # Use delete_memory_entry for permanent removal
                if self.delete_memory_entry(uuid_to_prune):
                    pruned_count += 1
                else:
                    logger.warning(f"Failed to prune summarized node {uuid_to_prune[:8]} (might have been deleted already).")

            logger.info(f"Pruned {pruned_count} summarized turn nodes.")
            # --- Tuning Log: Consolidation Pruning ---
            log_tuning_event("CONSOLIDATION_PRUNING", {
                "personality": self.personality,
                "pruning_enabled": prune_summarized,
                "summary_created": summary_created,
                "pruned_node_count": pruned_count,
                "pruned_node_uuids": nodes_to_process[:pruned_count], # Assuming they were pruned in order
            })
            # Note: delete_memory_entry already rebuilds the index and saves memory,
            # so no explicit rebuild/save needed here if pruning happened.
            # If no pruning happened, we still need to save other consolidation changes.
            if pruned_count == 0:
                 self._save_memory() # Save if no pruning occurred but other changes did
        else:
             # Save memory if pruning is disabled or no summary was created,
             # but other changes (concepts, relations) might have occurred.
             self._save_memory()

        logger.info("--- Consolidation Finished ---")

        # --- Update Short-Term Drive State ---
        self._update_drive_state(relevant_nodes=nodes_to_process)

        # --- Update Long-Term Drive State (Less Frequently) ---
        drive_cfg = self.config.get('subconscious_drives', {})
        lt_update_interval = drive_cfg.get('long_term_update_interval_consolidations', 0)
        # Need a way to track consolidation count - add an instance variable?
        if not hasattr(self, '_consolidation_counter'): self._consolidation_counter = 0
        self._consolidation_counter += 1
        logger.debug(f"Consolidation counter: {self._consolidation_counter}")

        if lt_update_interval > 0 and self._consolidation_counter >= lt_update_interval:
            logger.info(f"Long-term drive update interval ({lt_update_interval}) reached. Triggering update.")
            self._update_long_term_drives()
            self._consolidation_counter = 0 # Reset counter
        else:
            logger.debug(f"Skipping long-term drive update (Interval: {lt_update_interval}, Count: {self._consolidation_counter}).")


        # --- Update Autobiographical Model after consolidation ---
        self._generate_autobiographical_model()

        # --- Infer Second-Order Relationships ---
        self._infer_second_order_relations()

        # --- Final Save (if not already saved by pruning) ---
        saved_in_consolidation = False
        if not (prune_summarized and summary_created and pruned_count > 0):
            self._save_memory() # Save all changes (summary, concepts, relations, drives, ASM, inference)
            saved_in_consolidation = True

        # --- Tuning Log: Consolidation End ---
        log_tuning_event("CONSOLIDATION_END", {
            "personality": self.personality,
            "status": "completed",
            "nodes_processed_count": len(nodes_to_process),
            "summary_created": summary_created,
            "concepts_added_count": len(newly_added_concepts) if 'newly_added_concepts' in locals() else 0,
            # Add counts for relations if available from helpers
            "pruned_node_count": pruned_count if 'pruned_count' in locals() else 0,
            "memory_saved": saved_in_consolidation,
        })

#Function call
if __name__ == "__main__":
    # Basic test execution when the script is run directly
    logger.info("Running basic test of GraphMemoryClient...")

    # Initialize the client (using default personality from config)
    try:
        client = GraphMemoryClient()
        print("\n--- Initial State ---")
        print(f"Personality: '{client.personality}'")
        print(f"Nodes: {client.graph.number_of_nodes()}, Edges: {client.graph.number_of_edges()}")
        print(f"Embeddings: {len(client.embeddings)}, FAISS Index Vectors: {client.index.ntotal if client.index else 'N/A'}")
        print(f"Last added node: {client.last_added_node_uuid}")
        print("-" * 20)

        # --- Add a few nodes if memory is small ---
        min_consolidation_nodes = client.config.get('consolidation', {}).get('min_nodes', 5)
        if client.graph.number_of_nodes() < min_consolidation_nodes:
            print(f"Adding initial nodes (less than {min_consolidation_nodes} found)...")
            ts1 = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
            uuid1 = client.add_memory_node("The quick brown fox", "User", timestamp=ts1)
            ts2 = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
            uuid2 = client.add_memory_node("jumped over the lazy dog", "AI", timestamp=ts2)
            uuid3 = client.add_memory_node("Both are animals.", "User")
            print(f" Added nodes: {uuid1[:8] if uuid1 else 'Fail'}, {uuid2[:8] if uuid2 else 'Fail'}, {uuid3[:8] if uuid3 else 'Fail'}")
            client._save_memory() # Save after adding
        else:
            print("Sufficient nodes exist, skipping initial node addition.")
            # Get some existing node UUIDs for testing retrieval if needed
            try:
                 uuids = sorted(list(client.graph.nodes()), key=lambda u: client.graph.nodes[u].get('timestamp',''))
                 uuid1=uuids[-3] if len(uuids)>=3 else None
                 uuid2=uuids[-2] if len(uuids)>=2 else None
                 uuid3=uuids[-1] if len(uuids)>=1 else None
                 print(f"Using latest existing nodes for potential testing: {uuid1}, {uuid2}, {uuid3}")
            except Exception as e:
                 print(f"Could not get existing node UUIDs: {e}")


        # --- Test Retrieval ---
        print("\n--- Testing Retrieval ---")
        if client.graph.number_of_nodes() > 0 and client.index and client.index.ntotal > 0:
            query = "animal sounds" # Example query
            print(f"Query: '{query}'")
            initial_nodes = client._search_similar_nodes(query)
            initial_uuids = [uid for uid, score in initial_nodes]
            print(f" Initial nodes found by similarity search: {initial_uuids}")
            if initial_uuids:
                 chain = client.retrieve_memory_chain(initial_uuids)
                 print(f" Retrieved memory chain ({len(chain)} nodes):")
                 # Print details of retrieved nodes (limited)
                 for node in chain[:10]: # Limit output
                      print(f"  - UUID: {node['uuid'][:8]}, Act: {node.get('final_activation', 0.0):.3f}, Type: {node.get('node_type', '?')}, Text: '{node.get('text', '')[:40]}...'")
            else:
                 print(" No initial nodes found for retrieval.")
        else:
            print("Graph or FAISS index empty/unavailable, skipping retrieval test.")

        # --- Test Consolidation ---
        print("\n--- Testing Consolidation ---")
        # Check if consolidation should run based on node count
        if client.graph.number_of_nodes() >= min_consolidation_nodes:
            print("Running consolidation (requires sufficient nodes)...")
            try:
                # Ensure rich associations are enabled in config for testing that path
                # Or set client.config['features']['enable_rich_associations'] = True here for testing
                client.run_consolidation() # Run the main consolidation logic
                print("--- State After Consolidation ---")
                print(f"Nodes: {client.graph.number_of_nodes()}, Edges: {client.graph.number_of_edges()}")
                # Optionally save memory after consolidation test
                client._save_memory()
            except Exception as e:
                print(f"Error during consolidation test: {e}")
                logger.error("Error during consolidation test", exc_info=True)
        else:
            print(f"Skipping consolidation test (requires >= {min_consolidation_nodes} nodes).")

        # --- Test Action Analysis (Example) ---
        print("\n--- Testing Action Analysis ---")
        test_action_request = "add meeting for tomorrow 10am project discussion to calendar"
        print(f"Analyzing request: '{test_action_request}'")
        action_result = client.analyze_action_request(test_action_request)
        print(f" Analysis result: {action_result}")
        if action_result.get("action") not in ["none", "clarify", "error"]:
             print("  (Would normally execute this action now)")
             # success, message, suffix = client.execute_action(action_result)
             # print(f"  Execution result: Success={success}, Message='{message}'")

        test_none_request = "that is interesting"
        print(f"Analyzing request: '{test_none_request}'")
        action_result_none = client.analyze_action_request(test_none_request)
        print(f" Analysis result: {action_result_none}")


        logger.info("Basic test finished.")
        print("\nBasic test finished.")

    except Exception as main_e:
        print(f"\nAn error occurred during the main test execution: {main_e}")
        logger.critical("Error during main test execution", exc_info=True)
