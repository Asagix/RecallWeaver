# persistent_backend_graph.py
import math
import os
import re # <<< Add import re
import subprocess
import sys
import dataclasses
import spacy
import json
import logging
import time
import uuid
# import re # Removed duplicate import
import networkx as nx
import numpy as np
import faiss
import requests
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from networkx.readwrite import json_graph
from datetime import datetime, timezone, timedelta

from workspace_agent import WorkspaceAgent
from emotional_core import EmotionalCore # <<< NEW IMPORT

# Keywords that might indicate a need for workspace planning
WORKSPACE_KEYWORDS = [
    "file", "save", "create", "write", "append", "read", "open", "list", "delete", "remove",
    "calendar", "event", "schedule", "meeting", "appointment", "remind", "task",
    "note", "document", "report", "summary", "code", # Keywords related to file content
    "workspace", "directory", # Keywords related to the workspace itself
    # Add date/time related words often used with calendar
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "next week", "last week",
]


# --- Emoji Stripping Helper ---
# Regex to match most common emojis
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002B50"            # Star
    "\U0000FE0F"            # Variation selector
    "\U0001F004"            # Mahjong tile red dragon
    "\U0001F0CF"            # Playing card joker
    "]+", flags=re.UNICODE)

# Import zoneinfo safely
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    logging.warning("zoneinfo module not found. Using UTC. Consider `pip install tzdata`.") # Use logging directly here
    ZoneInfo = None # type: ignore
    ZoneInfoNotFoundError = Exception # Placeholder
from collections import defaultdict

# *** Removed text2emotion import ***

# *** Import file manager ***
import file_manager # Assuming file_manager.py exists in the same directory
# WorkspaceAgent will be imported later if needed by plan_and_execute

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


def strip_emojis(text: str) -> str:
    """Removes common emoji characters from a string."""
    if not isinstance(text, str):
        return text # Return non-strings as-is
    try:
        # Attempt to encode/decode to handle potential broken surrogates before regex
        cleaned_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogatepass')
        return EMOJI_PATTERN.sub(r'', cleaned_text)
    except Exception:
        # Fallback to regex only if encoding/decoding fails
        return EMOJI_PATTERN.sub(r'', text)


@dataclasses.dataclass
class InteractionResult:
    """Holds the results of processing a single user interaction."""
    final_response_text: str
    inner_thoughts: str | None = None
    memories_used: list = dataclasses.field(default_factory=list)
    user_node_uuid: str | None = None
    ai_node_uuid: str | None = None
    needs_planning: bool = False


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
        self.drives_file = os.path.join(self.data_dir, "drives.json")
        self.last_conversation_file = os.path.join(self.data_dir, "last_conversation.json") # NEW: Last conversation file path

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
        self.initial_history_turns = [] # DEPRECATED: Will use last_conversation_turns instead
        self.last_conversation_turns = [] # NEW: Store actual last N turns separately
        self.time_since_last_interaction_hours = 0.0 # Store time gap
        self.pending_re_greeting = None # Store generated re-greeting message
        # --- Track nodes with high emotional impact within a single interaction ---
        self.high_impact_nodes_this_interaction = {} # uuid -> magnitude
        self.emotional_core = None # Initialize EmotionalCore attribute

        os.makedirs(self.data_dir, exist_ok=True)
        embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        # --- Load tokenizer name/path from config ---
        tokenizer_name = self.config.get('tokenizer_name') # Get the path from config
        if not tokenizer_name:
            logger.error("Tokenizer name/path ('tokenizer_name') not found in config. Cannot load tokenizer.")
            # Handle error appropriately - maybe raise an exception or set tokenizer to None
            self.tokenizer = None
        else:
            logger.info(f"Using tokenizer path from config: {tokenizer_name}")
        # --- End tokenizer name loading ---
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

        # Load Tokenizer (only if tokenizer_name was successfully loaded from config)
        if tokenizer_name:
            try:
                logger.info(f"Loading tokenizer from: {tokenizer_name}") # Removed mention of slow implementation
                # Use the path loaded from config (removed use_fast=False)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                logger.info("Tokenizer loaded.") # Removed mention of slow implementation
            except Exception as e:
                logger.error(f"Failed loading tokenizer from '{tokenizer_name}': {e}", exc_info=True) # Removed (use_fast=False) from log
                self.tokenizer = None # Ensure tokenizer is None if loading fails
                # Decide if this is fatal
        else:
            # Tokenizer name was missing from config, already logged error
            self.tokenizer = None

        self._load_memory() # Loads data from self.data_dir

        # --- Initialize Emotional Core ---
        # Must happen after config is loaded but before memory load? No, after memory load is fine.
        try:
            if self.config.get('features', {}).get('enable_emotional_core', False):
                logger.info("Instantiating EmotionalCore...")
                self.emotional_core = EmotionalCore(self, self.config) # Pass self (client) and config
                if not self.emotional_core.is_enabled:
                    logger.warning("EmotionalCore instantiated but is disabled internally.")
                    self.emotional_core = None # Set back to None if disabled
                else:
                    logger.info("EmotionalCore instantiated successfully.")
            else:
                logger.info("EmotionalCore feature is disabled in main config.")
                self.emotional_core = None
        except Exception as e:
            logger.error(f"Failed to instantiate EmotionalCore: {e}", exc_info=True)
            self.emotional_core = None # Ensure it's None on error

        self._load_initial_history() # Load initial history after main memory load
        self._check_and_generate_re_greeting() # Check and generate re-greeting after history load

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

        # --- Load Last Conversation Turns ---
        self._load_last_conversation() # Load before calculating time gap

        # --- Calculate Time Gap (using last_conversation_turns) ---
        self._calculate_time_since_last_interaction() # Use helper

        # --- Check for Re-Greeting (using last_conversation_turns) ---
        self._check_and_generate_re_greeting() # Check and generate re-greeting

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

    def get_current_mood(self) -> tuple[float, float]:
        """Returns the last calculated interaction mood (Valence, Arousal)."""
        # Returns the mood calculated *after* the last interaction, used for the *next* retrieval bias.
        logger.debug(f"get_current_mood() returning: {self.last_interaction_mood}") # Add logging
        return self.last_interaction_mood

    def get_drive_state(self) -> dict:
        """Returns the current drive state dictionary."""
        # Return a copy to prevent external modification? Deep copy might be safer if nested.
        return self.drive_state.copy() if self.drive_state else {}

    # --- Emotion Analysis Helper (REMOVED) ---
    # def _analyze_and_update_emotion(self, node_uuid: str): ... (Removed - Handled by EmotionalCore)

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
            # --- Saving Last Conversation Turns moved to add_memory_node ---

            logger.info(f"Memory saving done ({time.time() - start_time:.2f}s).")
        except Exception as e: logger.error(f"Unexpected save error: {e}", exc_info=True)

    def _load_last_conversation(self, max_turns=6):
        """Loads the last N conversation turns from a separate JSON file."""
        self.last_conversation_turns = [] # Reset before loading
        if os.path.exists(self.last_conversation_file):
            try:
                with open(self.last_conversation_file, 'r', encoding='utf-8') as f:
                    loaded_turns = json.load(f)
                if isinstance(loaded_turns, list):
                    # Basic validation of structure (optional but recommended)
                    validated_turns = []
                    for turn in loaded_turns:
                        if isinstance(turn, dict) and all(k in turn for k in ["speaker", "text", "timestamp"]):
                            validated_turns.append(turn)
                        else:
                            logger.warning(f"Skipping invalid turn structure in {self.last_conversation_file}: {turn}")
                    # Ensure we only keep the last max_turns
                    self.last_conversation_turns = validated_turns[-max_turns:]
                    logger.info(f"Loaded {len(self.last_conversation_turns)} turns from {self.last_conversation_file}.")
                else:
                    logger.error(f"Invalid format in {self.last_conversation_file} (expected list). Initializing empty.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {self.last_conversation_file}: {e}. Initializing empty.")
            except Exception as e:
                logger.error(f"Unexpected error loading {self.last_conversation_file}: {e}. Initializing empty.", exc_info=True)
        else:
            logger.info(f"Last conversation file not found ({self.last_conversation_file}). Starting fresh.")

    def _save_last_conversation(self):
        """Saves the current self.last_conversation_turns list to its JSON file."""
        try:
            # Ensure directory exists (should already, but safety check)
            os.makedirs(os.path.dirname(self.last_conversation_file), exist_ok=True)
            with open(self.last_conversation_file, 'w', encoding='utf-8') as f:
                json.dump(self.last_conversation_turns, f, indent=2)
            logger.debug(f"Saved {len(self.last_conversation_turns)} turns to {self.last_conversation_file}")
        except Exception as e:
            logger.error(f"Failed saving last conversation turns immediately: {e}", exc_info=True)

    # --- Memory Node Management ---
    # (Keep add_memory_node, _rollback_add, delete_memory_entry, _find_latest_node_uuid, edit_memory_entry, forget_topic)
    # ... (methods unchanged) ...
    def add_memory_node(self, text: str, speaker: str, node_type: str = 'turn', timestamp: str = None, base_strength:
    float = 0.5, emotion_valence: float | None = None, emotion_arousal: float | None = None) -> str | None:
        """
        Adds a new memory node with enhanced attributes to the graph and index.
        Accepts optional emotion values.
        """
        logger.debug(f"ADD_MEMORY_NODE START: Has embedder? {hasattr(self, 'embedder')}")

        if not text: logger.warning("Skip adding empty node."); return None
        log_text = text[:80] + '...' if len(text) > 80 else text
        logger.info(f"Adding node: Spk={speaker}, Typ={node_type}, Txt='{strip_emojis(log_text)}'") # Strip emojis
        current_time = time.time()
        node_uuid = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        logger.debug(f"Assigning timestamp to node {node_uuid[:8]}: {timestamp}") # Add logging

        # --- Get config values safely ---
        features_cfg = self.config.get('features', {})
        saliency_enabled = features_cfg.get('enable_saliency', False)
        emotion_cfg = self.config.get('emotion_analysis', {})
        # Use provided emotions or fall back to defaults from config
        final_valence = emotion_valence if emotion_valence is not None else emotion_cfg.get('default_valence', 0.0)
        final_arousal = emotion_arousal if emotion_arousal is not None else emotion_cfg.get('default_arousal', 0.1)
        logger.debug(f"Node {node_uuid[:8]} Emotion: V={final_valence:.2f}, A={final_arousal:.2f} (Provided:V={emotion_valence}, A={emotion_arousal})")

        saliency_cfg = self.config.get('saliency', {})
        initial_scores = saliency_cfg.get('initial_scores', {})
        emotion_influence = saliency_cfg.get('emotion_influence_factor', 0.0)
        importance_keywords = saliency_cfg.get('importance_keywords', [])
        importance_boost = saliency_cfg.get('importance_saliency_boost', 0.0)
        flag_important_as_core = saliency_cfg.get('flag_important_as_core', False)


        # --- Calculate Initial Saliency ---
        initial_saliency = 0.0 # Default if disabled or error
        is_important_keyword_match = False # Flag for keyword match
        if saliency_enabled:
            # --- Base Saliency by Node Type ---
            if node_type == 'intention':
                # Give intention nodes higher base saliency
                base_saliency = initial_scores.get('intention', 0.75) # Use specific score or default
            else:
                base_saliency = initial_scores.get(node_type, initial_scores.get('default', 0.5))

                # Influence initial saliency by FINAL arousal (passed in or default)
                initial_saliency = base_saliency + (final_arousal * emotion_influence)

                # --- Check for Importance Keywords ---
            if importance_keywords and importance_boost > 0:
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in importance_keywords):
                    is_important_keyword_match = True
                    initial_saliency += importance_boost
                    logger.info(
                        f"Importance keyword match found in node {node_uuid[:8]}. "
                        f"Boosting initial saliency by {importance_boost}."
                    )

                    initial_saliency = max(0.0, min(1.0, initial_saliency))  # Clamp between 0 and 1
                    logger.debug(
                        f"Calculated initial saliency for {node_uuid[:8]} ({node_type}): {initial_saliency:.3f} "
                        f"(Base: {base_saliency}, ArousalInf: {final_arousal * emotion_influence:.3f}, "
                        f"ImportanceBoost: {importance_boost if is_important_keyword_match else 0.0})"
                    )
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
                emotion_valence=final_valence, # Use final (potentially provided) emotion
                emotion_arousal=final_arousal, # Use final (potentially provided) emotion
                saliency_score=initial_saliency, # NEW: Calculated initial saliency
                # --- Existing attributes ---
                base_strength=float(base_strength),
                activation_level=0.0, # Initial activation, updated during retrieval
                last_accessed_ts=current_time, # Timestamp of last access/creation
                # --- NEW: Decay Resistance ---
                decay_resistance_factor=self.config.get('forgetting', {}).get('decay_resistance', {}).get(node_type, 1.0),
                # --- NEW: Feedback Score ---
                user_feedback_score=0, # Initialize feedback score
                # --- NEW: Core Memory Flag (potentially set by importance keywords) ---
                is_core_memory= (is_important_keyword_match and flag_important_as_core)
            )
            # Access node data *after* adding it to the graph
            node_data = self.graph.nodes[node_uuid]
            if node_data.get('is_core_memory'):
                logger.info(f"Node {node_uuid[:8]} flagged as CORE MEMORY due to importance keyword match.")
                log_tuning_event("CORE_MEMORY_FLAGGED", {
                    "personality": self.personality,
                    "node_uuid": node_uuid,
                    "reason": "importance_keyword",
                })
            logger.debug(f"Node {node_uuid[:8]} added with decay_resistance: {node_data.get('decay_resistance_factor')}, is_core: {node_data.get('is_core_memory')}")
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

        # --- Update Last Conversation Turns (if it's a turn node) ---
        if node_type == 'turn':
            try:
                turn_data = {
                    "speaker": speaker,
                    "text": text,
                    "timestamp": timestamp,
                    "uuid": node_uuid # Include UUID for potential future use
                }
                self.last_conversation_turns.append(turn_data)
                # Keep only the last N turns (e.g., 6)
                max_turns_to_keep = 6
                if len(self.last_conversation_turns) > max_turns_to_keep:
                    self.last_conversation_turns = self.last_conversation_turns[-max_turns_to_keep:]
                logger.debug(f"Updated last_conversation_turns. Current count: {len(self.last_conversation_turns)}")
                # --- Save immediately after updating the list ---
                self._save_last_conversation()
            except Exception as e:
                logger.error(f"Error updating/saving last_conversation_turns: {e}", exc_info=True)

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
        logger.info(f"Editing: UUID={node_uuid[:8]}, Text='{strip_emojis(new_text[:50])}...'"); # Strip emojis
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
        logger.info(f"Forget topic request: '{strip_emojis(topic[:50])}...'") # Strip emojis
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
        if deleted_count > 0: message = f"Forgot topic '{strip_emojis(topic)}'. Deleted {deleted_count} entries." + (f" ({failed_count} fails)." if failed_count else ""); logger.info(message + f" UUIDs: {deleted_uuids}"); return True, message # Strip emojis in message
        else: message = f"Could not delete memories for '{strip_emojis(topic)}'."; logger.warning(message); return False, message # Strip emojis in message


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
        act_cfg = self.config.get('activation', {})
        if k is None: k = act_cfg.get('max_initial_nodes', 7)

        if not query_text or self.index is None or self.index.ntotal == 0: return []
        try:
            q_embed = self._get_embedding(query_text)
            if q_embed is None or q_embed.shape != (self.embedding_dim,): return []

            q_embed_np = np.array([q_embed], dtype='float32')
            search_multiplier = 3
            if query_type == 'episodic': search_multiplier = 5
            elif query_type == 'semantic': search_multiplier = 4

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
                            node_type = node_data.get('node_type')

                            if node_type_filter and node_type != node_type_filter:
                                logger.debug(f"    -> Filtered (Explicit): UUID={node_uuid[:8]} (Type {node_type} != {node_type_filter})")
                                continue

                            adjusted_dist = dist
                            penalty_applied = False
                            if query_type == 'episodic' and node_type != 'turn':
                                # --- MODIFIED Penalty Factor ---
                                penalty_factor = 1.1 # Reduced from 1.5
                                # --- END MODIFICATION ---
                                adjusted_dist *= penalty_factor
                                penalty_applied = True
                                logger.debug(f"    -> Penalized (Episodic Bias): UUID={node_uuid[:8]} (Type {node_type} != 'turn'). Dist {dist:.3f} -> {adjusted_dist:.3f}")
                            elif query_type == 'semantic' and node_type not in ['summary', 'concept']:
                                penalty_factor = 1.2
                                adjusted_dist *= penalty_factor
                                penalty_applied = True
                                logger.debug(f"    -> Penalized (Semantic Bias): UUID={node_uuid[:8]} (Type {node_type} not summary/concept). Dist {dist:.3f} -> {adjusted_dist:.3f}")

                            results.append((node_uuid, adjusted_dist))
                            logger.debug(f"    -> Added Candidate: UUID={node_uuid[:8]} (Type: {node_type}, AdjDist: {adjusted_dist:.3f})")

                        else: logger.debug(f"    -> UUID for FAISS ID {fid_int} not in graph/map.")
                    else: logger.debug(f"    -> Invalid FAISS ID -1 encountered.")

                    if len(results) >= k:
                        logger.debug(f"    -> Reached target k={k} results. Stopping search.")
                        break

            results.sort(key=lambda item: item[1])
            final_results = [(uuid, dist) for uuid, dist in results[:k]] # Return only top k
            logger.info(f"Found {len(final_results)} potentially relevant nodes (type='{node_type_filter or 'any'}', query_type='{query_type}') for query '{strip_emojis(query_text[:30])}...'")
            logger.debug(f" Final top {k} nodes after sorting by adjusted distance: {final_results}")
            return final_results

        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            return []

    def retrieve_memory_chain(self, initial_node_uuids: list[str],
                              recent_concept_uuids: list[str] | None = None,
                              current_mood: tuple[float, float] | None = None) -> tuple[list[dict], tuple[float, float] | None]:
        """
        Retrieves relevant memories using activation spreading.
        Considers memory strength, saliency, edge types, optionally boosts recently mentioned concepts,
        and optionally biases based on emotional context similarity to current_mood.
        Also applies dynamic edge weighting based on current drive state.

        Returns:
            A tuple containing:
            - list[dict]: The list of retrieved memory node data.
            - tuple[float, float] | None: The effective mood (Valence, Arousal) used during retrieval (after drive influence).
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

        guaranteed_saliency_threshold = act_cfg.get('guaranteed_saliency_threshold', 0.88) # Use updated default
        priming_boost_factor = act_cfg.get('priming_boost_factor', 1.0) # Get priming boost factor
        intention_boost_factor = act_cfg.get('intention_boost_factor', 1.2) # NEW: Boost for intention nodes
        always_retrieve_core = act_cfg.get('always_retrieve_core', True) # NEW: Flag for core retrieval

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

        # --- Drive State Influence on Mood & Edge Weighting ---
        drive_cfg = self.config.get('subconscious_drives', {})
        mood_influence_cfg = drive_cfg.get('mood_influence', {})
        drives_enabled = drive_cfg.get('enabled', False)
        mood_before_drive_influence = current_mood if current_mood else (0.0, 0.1) # Store initial mood
        effective_mood = mood_before_drive_influence # Start with initial mood
        drive_state_snapshot = self.drive_state.copy() # Get snapshot for this retrieval

        # Check if drives are enabled and we have state data
        if drives_enabled and mood_influence_cfg and drive_state_snapshot.get("short_term"): # Use snapshot
            logger.info(f"Calculating mood adjustment based on drive state: ShortTerm={drive_state_snapshot.get('short_term')}, LongTerm={drive_state_snapshot.get('long_term')}") # Changed level to INFO
            base_valence, base_arousal = effective_mood
            valence_adjustment = 0.0
            arousal_adjustment = 0.0
            valence_factors = mood_influence_cfg.get('valence_factors', {})
            arousal_factors = mood_influence_cfg.get('arousal_factors', {})
            base_drives = drive_cfg.get('base_drives', {}) # Use base_drives config
            long_term_influence = drive_cfg.get('long_term_influence_on_baseline', 1.0)

            for drive_name, current_activation in drive_state_snapshot["short_term"].items(): # Use snapshot
                # Calculate the dynamic baseline for comparison
                config_baseline = base_drives.get(drive_name, 0.0)
                long_term_level = drive_state_snapshot["long_term"].get(drive_name, 0.0) # Use snapshot
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
                # Simplified logic: factor sign determines direction. Review config factors.
                arousal_adj = arousal_factor * deviation
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
            # --- Log mood adjustment from drives ---
            log_tuning_event("RETRIEVAL_MOOD_DRIVE_ADJUSTMENT", {
                "personality": self.personality,
                "mood_before": mood_before_drive_influence,
                "valence_adjustment": valence_adjustment,
                "arousal_adjustment": arousal_adjustment,
                "mood_after": effective_mood,
                "drive_state_short_term": self.drive_state.get("short_term"), # Log state that caused adjustment
            })
        # --- Apply EmotionalCore Mood Hints (NEW) - This block was moved outside the else ---
        # Note: This applies hints *after* potential drive adjustment, or to the original mood if drives were disabled.
        elif self.emotional_core and self.emotional_core.is_enabled: # Check if it should run even if drives are disabled
            valence_hint = self.emotional_core.derived_mood_hints.get("valence", 0.0)
            arousal_hint = self.emotional_core.derived_mood_hints.get("arousal", 0.0)
            valence_factor = self.emotional_core.config.get("mood_valence_factor", 0.3)
            arousal_factor = self.emotional_core.config.get("mood_arousal_factor", 0.2)

            if abs(valence_hint) > 1e-4 or abs(arousal_hint) > 1e-4:
                logger.info(f"Applying EmotionalCore mood hints: V_hint={valence_hint:.2f} (Factor:{valence_factor:.2f}), A_hint={arousal_hint:.2f} (Factor:{arousal_factor:.2f})")
                current_v, current_a = effective_mood # Mood potentially after drive influence
                # Apply hints additively, scaled by factors
                new_v = current_v + (valence_hint * valence_factor)
                new_a = current_a + (arousal_hint * arousal_factor)
                # Clamp final mood
                effective_mood = (max(-1.0, min(1.0, new_v)), max(0.0, min(1.0, new_a)))
                logger.info(f"Mood after EmotionalCore hints: ({effective_mood[0]:.2f}, {effective_mood[1]:.2f})")
                log_tuning_event("RETRIEVAL_MOOD_EMOCORE_ADJUSTMENT", {
                    "personality": self.personality,
                    "mood_after_drives": (current_v, current_a), # Log mood *before* hint application
                    "valence_hint": valence_hint,
                    "arousal_hint": arousal_hint,
                    "valence_factor": valence_factor,
                    "arousal_factor": arousal_factor,
                    "mood_after_emocore": effective_mood,
                })

        else: # Drives were disabled or no state/config found
            logger.debug("Subconscious drives disabled or no config/state found, using original mood.")
            # --- Log that no drive adjustment was made ---
            log_tuning_event("RETRIEVAL_MOOD_DRIVE_ADJUSTMENT", {
                "personality": self.personality,
                "mood_before": mood_before_drive_influence,
                "valence_adjustment": 0.0,
                "arousal_adjustment": 0.0,
                "mood_after": effective_mood, # Will be same as mood_before here
                "reason": "Drives disabled or state missing",
            })
            # --- Apply EmotionalCore Mood Hints (NEW) even if drives disabled ---
            if self.emotional_core and self.emotional_core.is_enabled:
                valence_hint = self.emotional_core.derived_mood_hints.get("valence", 0.0)
                arousal_hint = self.emotional_core.derived_mood_hints.get("arousal", 0.0)
                valence_factor = self.emotional_core.config.get("mood_valence_factor", 0.3)
                arousal_factor = self.emotional_core.config.get("mood_arousal_factor", 0.2)

                if abs(valence_hint) > 1e-4 or abs(arousal_hint) > 1e-4:
                    logger.info(f"Applying EmotionalCore mood hints (drives disabled): V_hint={valence_hint:.2f} (Factor:{valence_factor:.2f}), A_hint={arousal_hint:.2f} (Factor:{arousal_factor:.2f})")
                    current_v, current_a = effective_mood # Original mood
                    # Apply hints additively, scaled by factors
                    new_v = current_v + (valence_hint * valence_factor)
                    new_a = current_a + (arousal_hint * arousal_factor)
                    # Clamp final mood
                    effective_mood = (max(-1.0, min(1.0, new_v)), max(0.0, min(1.0, new_a)))
                    logger.info(f"Mood after EmotionalCore hints: ({effective_mood[0]:.2f}, {effective_mood[1]:.2f})")
                    log_tuning_event("RETRIEVAL_MOOD_EMOCORE_ADJUSTMENT", {
                        "personality": self.personality,
                        "mood_after_drives": (current_v, current_a), # Log mood *before* hint application
                        "valence_hint": valence_hint,
                        "arousal_hint": arousal_hint,
                        "valence_factor": valence_factor,
                        "arousal_factor": arousal_factor,
                        "mood_after_emocore": effective_mood,
                    })


        # --- Emotional Context Config (Uses effective_mood) ---
        emo_ctx_cfg = act_cfg.get('emotional_context', {})
        # Enable emotional context bias if the feature is on AND we have a valid mood (original or adjusted)
        emo_ctx_enabled = emo_ctx_cfg.get('enable', False) and effective_mood is not None
        emo_max_dist = emo_ctx_cfg.get('max_distance', 1.414)
        emo_boost = emo_ctx_cfg.get('boost_factor', 0.0) # Additive boost
        emo_penalty = emo_ctx_cfg.get('penalty_factor', 0.0) # Subtractive penalty

        logger.info(f"Starting retrieval. Initial nodes: {initial_node_uuids} (SalInf: {activation_influence:.2f}, GuarSal>=: {guaranteed_saliency_threshold}, FocusBoost: {context_focus_boost}, RecentConcepts: {len(recent_concept_uuids_set)}, EmoCtx: {emo_ctx_enabled}, Mood: {effective_mood})")
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

        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph empty.")
            return [], effective_mood # Return empty list and the effective mood

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
                        except Exception as e:
                            logger.warning(f"Error checking concept links for focus boost on {uuid[:8]}: {e}")

                    if is_recent_concept or mentions_recent_concept:
                        boost_applied = 1.0 + context_focus_boost
                        logger.debug(f"Applying context focus boost ({boost_applied:.2f}) to node {uuid[:8]} (IsRecent: {is_recent_concept}, MentionsRecent: {mentions_recent_concept})")

                final_initial_activation = base_initial_activation * boost_applied

                # --- Apply Priming Boost ---
                priming_applied = 1.0
                if uuid in last_turn_uuids_for_priming: # Check against internally derived set
                    priming_applied = priming_boost_factor
                    logger.debug(f"Applying priming boost ({priming_applied:.2f}) to last turn node {uuid[:8]}")

                final_initial_activation *= priming_applied # Apply priming boost

                # --- Apply Intention Boost ---
                intention_applied = 1.0
                if node_data.get('node_type') == 'intention' and intention_boost_factor > 1.0:
                    intention_applied = intention_boost_factor
                    logger.debug(f"Applying intention boost ({intention_applied:.2f}) to intention node {uuid[:8]}")

                final_initial_activation *= intention_applied # Apply intention boost

                activation_levels[uuid] = final_initial_activation
                node_data['last_accessed_ts'] = current_time # Update access time
                valid_initial_nodes.add(uuid)
                logger.debug(f"Initialized node {uuid[:8]} - Strength: {initial_strength:.3f}, BaseAct: {base_initial_activation:.3f}, CtxBoost: {boost_applied:.2f}, Priming: {priming_applied:.2f}, Intention: {intention_applied:.2f}, FinalAct: {final_initial_activation:.3f}")
            else:
                logger.warning(f"Initial node {uuid} not in graph.")

        if not activation_levels:
            logger.warning("No valid initial nodes found in graph.")
            return [], effective_mood # Return empty list and the effective mood

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
                if not source_data:
                    continue
                source_act = activation_levels.get(source_uuid, 0)
                # Safely get saliency score, default to 0 if missing or not a number for calculation
                raw_saliency = source_data.get('saliency_score', 0.0)
                source_saliency = raw_saliency if isinstance(raw_saliency, (int, float)) else 0.0

                if source_act < 1e-6:
                    continue # Skip if effectively inactive

                neighbors = set(self.graph.successors(source_uuid)) | set(self.graph.predecessors(source_uuid))

                for neighbor_uuid in neighbors:
                    if neighbor_uuid == source_uuid:
                        continue
                    neighbor_data = self.graph.nodes.get(neighbor_uuid)
                    if not neighbor_data:
                        continue

                    # No status check needed here anymore

                    is_forward = self.graph.has_edge(source_uuid, neighbor_uuid)
                    edge_data = self.graph.get_edge_data(source_uuid, neighbor_uuid) if is_forward else self.graph.get_edge_data(neighbor_uuid, source_uuid)
                    if not edge_data:
                        continue

                    edge_type = edge_data.get('type', 'UNKNOWN')
                    # type_factor = prop_unknown # Default - Redundant, assigned below

                    # --- Assign base type_factor based on edge_type ---
                    base_type_factor = prop_unknown # Default
                    if edge_type == 'TEMPORAL': base_type_factor = prop_temporal_fwd if is_forward else prop_temporal_bwd
                    elif edge_type == 'SUMMARY_OF': base_type_factor = prop_summary_fwd if is_forward else prop_summary_bwd
                    elif edge_type == 'MENTIONS_CONCEPT': base_type_factor = prop_concept_fwd if is_forward else prop_concept_bwd
                    elif edge_type == 'ASSOCIATIVE': base_type_factor = prop_assoc
                    elif edge_type == 'HIERARCHICAL': base_type_factor = prop_hier_fwd if is_forward else prop_hier_bwd
                    elif edge_type == 'CAUSES': base_type_factor = prop_causes # Assume forward A->B means A causes B
                    elif edge_type == 'PART_OF': base_type_factor = prop_part_of
                    elif edge_type == 'HAS_PROPERTY': base_type_factor = prop_has_prop
                    elif edge_type == 'ENABLES': base_type_factor = prop_enables # Corrected assignment
                    elif edge_type == 'PREVENTS': base_type_factor = prop_prevents # Corrected assignment
                    elif edge_type == 'CONTRADICTS': base_type_factor = prop_contradicts # Corrected assignment
                    elif edge_type == 'SUPPORTS': base_type_factor = prop_supports # Corrected assignment
                    elif edge_type == 'EXAMPLE_OF': base_type_factor = prop_example_of
                    elif edge_type == 'MEASURES': base_type_factor = prop_measures
                    elif edge_type == 'LOCATION_OF': base_type_factor = prop_location_of
                    elif edge_type == 'ANALOGY': base_type_factor = prop_analogy
                    elif edge_type == 'INFERRED_RELATED_TO': base_type_factor = prop_inferred
                    elif edge_type.startswith('SPACY_'): base_type_factor = prop_spacy # Generic for all spaCy types
                    # else: base_type_factor remains prop_unknown

                    # --- Dynamic Edge Weighting based on Drive State ---
                    drive_weight_multiplier = 1.0 # Default: no change
                    if drives_enabled and drive_state_snapshot.get("short_term"):
                        # Example: Boost HIERARCHICAL/CAUSES if Understanding drive is low (negative deviation = unmet need)
                        understanding_level = drive_state_snapshot["short_term"].get("Understanding", 0.0)
                        understanding_baseline = base_drives.get("Understanding", 0.0) + (drive_state_snapshot["long_term"].get("Understanding", 0.0) * long_term_influence)
                        understanding_deviation = understanding_level - understanding_baseline
                        if understanding_deviation < -0.1: # If Understanding need is unmet
                            if edge_type in ['HIERARCHICAL', 'CAUSES', 'SUMMARY_OF', 'PART_OF', 'HAS_PROPERTY']: # Added related types
                                drive_weight_multiplier = 1.2 # Boost these edge types by 20%
                                logger.debug(f"    Drive Boost (Understanding Low): Edge {edge_type} mult={drive_weight_multiplier:.2f}")

                        # Example: Boost CONNECTION if Connection drive is low (negative deviation = unmet need)
                        connection_level = drive_state_snapshot["short_term"].get("Connection", 0.0)
                        connection_baseline = base_drives.get("Connection", 0.0) + (drive_state_snapshot["long_term"].get("Connection", 0.0) * long_term_influence)
                        connection_deviation = connection_level - connection_baseline
                        if connection_deviation < -0.1: # If Connection need is unmet
                            if edge_type in ['ASSOCIATIVE', 'ANALOGY', 'SUPPORTS', 'MENTIONS_CONCEPT']: # Added related types
                                drive_weight_multiplier = 1.15 # Boost by 15%
                                logger.debug(f"    Drive Boost (Connection Low): Edge {edge_type} mult={drive_weight_multiplier:.2f}")
                        # Add more drive-based weighting rules here...

                    # Apply drive multiplier to the base type factor
                    type_factor = base_type_factor * drive_weight_multiplier

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
                            # logger.debug(f"    EmoCtx: Mood=({mood_v:.2f},{mood_a:.2f}), Nbr=({neighbor_v:.2f},{neighbor_a:.2f}), Dist={emo_dist:.3f}, NormDist={norm_dist:.3f}, Adjust={emo_adjustment:.3f}") # Moved logging below
                            # --- Log emotional context bias calculation ---
                            log_tuning_event("RETRIEVAL_EMO_CTX_BIAS_CALC", {
                                "personality": self.personality,
                                "source_node_uuid": source_uuid,
                                "neighbor_node_uuid": neighbor_uuid,
                                "effective_mood": effective_mood,
                                "neighbor_mood": (neighbor_v, neighbor_a),
                                "emotional_distance": emo_dist,
                                "normalized_distance": norm_dist,
                                "calculated_adjustment": emo_adjustment,
                                "base_activation_pass": base_act_pass,
                            })

                        except Exception as e:
                            logger.warning(f"Error calculating emotional context bias for {neighbor_uuid[:8]}: {e}")
                            emo_adjustment = 0.0 # Default to no adjustment on error

                    # Apply adjustment (additive/subtractive)
                    act_pass = base_act_pass + emo_adjustment
                    # Ensure activation doesn't go below zero due to penalty
                    act_pass = max(0.0, act_pass)

                    if act_pass > 1e-6:
                        newly_activated[neighbor_uuid] += act_pass
                        logger.debug(f"  Spread: {source_uuid[:8]}(A:{source_act:.2f},S:{source_saliency:.2f}) -> {neighbor_uuid[:8]}(Str:{neighbor_strength:.2f}) ({edge_type},{'F' if is_forward else 'B'}), DStr:{dyn_str:.2f}, TypeF:{type_factor:.2f}, SalB:{saliency_boost:.2f}, EmoAdj:{emo_adjustment:.3f} => Pass:{act_pass:.3f}")

                        edge_key = (source_uuid, neighbor_uuid) if is_forward else (neighbor_uuid, source_uuid)
                        if edge_key in self.graph.edges:
                            self.graph.edges[edge_key]['last_traversed_ts'] = current_time

            # --- Apply Decay and Combine Activation ---
            nodes_to_decay = list(activation_levels.keys())
            active_nodes.clear()
            all_involved_nodes = set(nodes_to_decay) | set(newly_activated.keys())

            for uuid in all_involved_nodes:
                node_data = self.graph.nodes.get(uuid)
                if not node_data:
                    continue

                current_activation = activation_levels.get(uuid, 0.0)
                if current_activation > 0:
                    decay_mult = self._calculate_node_decay(node_data, current_time)
                    activation_levels[uuid] *= decay_mult
                    # logger.debug(f"  Decay: {uuid[:8]} ({current_activation:.3f} * {decay_mult:.3f} -> {activation_levels[uuid]:.3f})")

                activation_levels[uuid] += newly_activated.get(uuid, 0.0)
                # Update access time whenever activation is touched (decayed or increased)
                self.graph.nodes[uuid]['last_accessed_ts'] = current_time

                if activation_levels[uuid] > 1e-6: # Check activation *after* decay and addition
                    active_nodes.add(uuid)
                elif uuid in activation_levels: # If activation fell to zero or below during this step
                    del activation_levels[uuid]

            logger.debug(f" Step {depth+1} finished. Active Nodes: {len(active_nodes)}. Max Activation: {max(activation_levels.values()) if activation_levels else 0:.3f}")
            if not active_nodes:
                break

        # --- Interference Simulation Step ---
        interference_cfg = act_cfg.get('interference', {})
        interference_applied_count = 0 # Initialize here
        penalized_nodes = set() # Initialize here
        if interference_cfg.get('enable', False) and self.index and self.index.ntotal > 0:
            logger.info("--- Applying Interference Simulation ---")
            check_threshold = interference_cfg.get('check_threshold', 0.15)
            sim_threshold = interference_cfg.get('similarity_threshold', 0.25) # L2 distance
            penalty_factor = interference_cfg.get('penalty_factor', 0.90)
            k_neighbors = interference_cfg.get('max_neighbors_check', 5)
            # penalized_nodes = set() # Moved initialization up
            # interference_applied_count = 0 # Moved initialization up

            # Iterate through nodes activated above the check threshold
            nodes_to_check = sorted(activation_levels.items(), key=lambda item: item[1], reverse=True)

            for source_uuid, source_activation in nodes_to_check:
                if source_uuid in penalized_nodes:
                    continue # Already penalized, skip check
                if source_activation < check_threshold:
                    continue # Below threshold to cause interference

                source_embedding = self.embeddings.get(source_uuid)
                if source_embedding is None:
                    continue

                # Find nearest neighbors in embedding space
                try:
                    source_embed_np = np.array([source_embedding], dtype='float32')
                    distances, indices = self.index.search(source_embed_np, k_neighbors + 1) # Search k+1 to include self potentially

                    local_cluster = [] # (uuid, activation, distance)
                    if len(indices) > 0 and len(indices[0]) > 0: # Check if search returned results
                        for i, faiss_id in enumerate(indices[0]):
                            neighbor_uuid = self.faiss_id_to_uuid.get(int(faiss_id))
                            if neighbor_uuid is None or neighbor_uuid == source_uuid: # Check if uuid exists and skip self
                                continue
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

                except AttributeError:
                    logger.warning("Interference check failed: Faiss index or faiss_id_to_uuid mapping likely not initialized correctly.")
                    break # Stop interference if index is broken
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
            "interference_enabled": interference_cfg.get('enable', False),
            "interference_applied_count": interference_applied_count,
            "penalized_node_uuids": list(penalized_nodes),
        })


        # --- Final Selection & Update ---
        relevant_nodes_dict = {} # Use dict to avoid duplicates easily: uuid -> node_info
        processed_uuids_for_access_count = set()
        # guaranteed_added_count = 0 # Unused variable

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
                        # Don't log here, log summary later
                        # logger.debug(f"Incremented access count for {uuid[:8]} to {node_data['access_count']}")

                    node_info = node_data.copy()
                    node_info['final_activation'] = final_activation
                    node_info['guaranteed_inclusion'] = False # Mark as normally included
                    relevant_nodes_dict[uuid] = node_info

                    # --- Boost Saliency on Successful Recall (Threshold Pass) ---
                    if saliency_enabled:
                        boost_factor = saliency_cfg.get('recall_boost_factor', 0.05)
                        if boost_factor > 0:
                            current_saliency = node_data.get('saliency_score', 0.0)
                            if isinstance(current_saliency, (int, float)): # Ensure it's a number
                                new_saliency = min(1.0, current_saliency + boost_factor) # Additive boost, clamped
                                if new_saliency > current_saliency:
                                    self.graph.nodes[uuid]['saliency_score'] = new_saliency # Update graph directly
                                    # logger.debug(f"Boosted saliency (threshold recall) for {uuid[:8]} to {new_saliency:.3f}")

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
                                # Ensure node emotions are valid numbers
                                if isinstance(node_v, (int, float)) and isinstance(node_a, (int, float)):
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
                                        # logger.debug(f"  EmoRecon (Thresh): Node {uuid[:8]} V/A ({node_v:.2f},{node_a:.2f}) nudged towards mood ({mood_v:.2f},{mood_a:.2f}) -> ({new_v:.2f},{new_a:.2f}). Dist={emo_dist:.3f}")
                            except Exception as e:
                                logger.warning(f"Error during emotional reconsolidation for {uuid[:8]}: {e}")

        logger.info(f"Found {len(relevant_nodes_dict)} active nodes above activation threshold ({activation_threshold}).")

        # Pass 2: Check for high-saliency nodes missed by activation threshold OR core memory nodes
        core_added_count = 0
        saliency_guaranteed_added_count = 0
        for uuid, final_activation in activation_levels.items():
            if uuid not in relevant_nodes_dict: # Only check nodes not already included
                node_data = self.graph.nodes.get(uuid)
                if node_data:
                    is_core = node_data.get('is_core_memory', False)
                    current_saliency = node_data.get('saliency_score', 0.0)
                    # Ensure saliency is a number for comparison
                    current_saliency = current_saliency if isinstance(current_saliency, (int, float)) else 0.0
                    should_include = False
                    inclusion_reason = ""

                    # Check Core Memory Guarantee FIRST
                    if is_core and always_retrieve_core:
                        should_include = True
                        inclusion_reason = "Core Memory Guarantee"
                        core_added_count += 1
                    # Check Saliency Guarantee if not already included by core
                    elif current_saliency >= guaranteed_saliency_threshold:
                        should_include = True
                        inclusion_reason = f"High Saliency ({current_saliency:.3f} >= {guaranteed_saliency_threshold})"
                        saliency_guaranteed_added_count += 1

                    if should_include:
                        logger.info(f"Guaranteed inclusion for node {uuid[:8]} (Reason: {inclusion_reason}, Act: {final_activation:.3f})")
                        # Increment access count if not already done
                        if uuid not in processed_uuids_for_access_count:
                            self.graph.nodes[uuid]['access_count'] = self.graph.nodes[uuid].get('access_count', 0) + 1 # Update graph directly
                            processed_uuids_for_access_count.add(uuid)
                            # logger.debug(f"Incremented access count for guaranteed node {uuid[:8]} to {node_data['access_count']}")

                        node_info = node_data.copy()
                        # Store the actual activation, even if below threshold
                        node_info['final_activation'] = final_activation
                        # Mark guaranteed inclusion type
                        node_info['guaranteed_inclusion'] = 'core' if is_core and always_retrieve_core else 'saliency'
                        relevant_nodes_dict[uuid] = node_info

                        # --- Boost Saliency on Successful Recall (Guarantee Pass - Core or Saliency) ---
                        if saliency_enabled:
                            boost_factor = saliency_cfg.get('recall_boost_factor', 0.05)
                            if boost_factor > 0:
                                # current_saliency already fetched and validated above
                                new_saliency = min(1.0, current_saliency + boost_factor) # Additive boost, clamped
                                if new_saliency > current_saliency:
                                    self.graph.nodes[uuid]['saliency_score'] = new_saliency # Update graph directly
                                    # logger.debug(f"Boosted saliency (guaranteed recall) for {uuid[:8]} to {new_saliency:.3f}")

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
                                    # Ensure node emotions are valid numbers
                                    if isinstance(node_v, (int, float)) and isinstance(node_a, (int, float)):
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
                                            # logger.debug(f"  EmoRecon (Guar): Node {uuid[:8]} V/A ({node_v:.2f},{node_a:.2f}) nudged towards mood ({mood_v:.2f},{mood_a:.2f}) -> ({new_v:.2f},{new_a:.2f}). Dist={emo_dist:.3f}")
                                except Exception as e:
                                    logger.warning(f"Error during emotional reconsolidation for guaranteed node {uuid[:8]}: {e}")


        if core_added_count > 0:
            logger.info(f"Added {core_added_count} additional nodes due to Core Memory guarantee.")
        if saliency_guaranteed_added_count > 0:
            logger.info(f"Added {saliency_guaranteed_added_count} additional nodes due to high saliency guarantee.")
        if len(processed_uuids_for_access_count) > 0:
            logger.info(f"Incremented access count for {len(processed_uuids_for_access_count)} retrieved nodes.")


        # Convert dict back to list and sort
        relevant_nodes = list(relevant_nodes_dict.values())
        # Sort primarily by activation, then timestamp as secondary
        # Core memories might have low activation but should still be sorted reasonably
        relevant_nodes.sort(key=lambda x: (x.get('final_activation', 0.0), x.get('timestamp', '')), reverse=True)

        # Log final nodes with guarantee type marker
        if relevant_nodes:
            log_parts = []
            for n in relevant_nodes:
                marker = ""
                if n.get('guaranteed_inclusion') == 'core':
                    marker = "**" # Core marker
                elif n.get('guaranteed_inclusion') == 'saliency':
                    marker = "*" # Saliency marker
                log_parts.append(f"{n['uuid'][:8]}({n['final_activation']:.3f}{marker})")
            logger.info(f"Final nodes ({len(relevant_nodes)} total): [{', '.join(log_parts)}]")
        else:
            logger.info("No relevant nodes found above threshold or guaranteed.")

        # --- Corrected Debug Logging ---
        logger.debug("--- Retrieved Node Details (Top 5) ---")
        for i, node in enumerate(relevant_nodes[:5]):
            # Safely get and format saliency and strength scores
            saliency_val = node.get('saliency_score', '?')
            strength_val = node.get('memory_strength', '?')
            saliency_str = f"{saliency_val:.2f}" if isinstance(saliency_val, (int, float)) else str(saliency_val)
            strength_str = f"{strength_val:.2f}" if isinstance(strength_val, (int, float)) else str(strength_val)
            guar_str = f" Guar:{node.get('guaranteed_inclusion')}" if node.get('guaranteed_inclusion') else ""
            # Format the log message including strength
            logger.debug(f"  {i+1}. ({node['final_activation']:.3f}) UUID:{node['uuid'][:8]} Str:{strength_str} Count:{node.get('access_count','?')} Sal:{saliency_str}{guar_str} Text: '{strip_emojis(node.get('text', 'N/A')[:80])}...'") # Strip emojis
        logger.debug("------------------------------------")

        # --- Tuning Log: Retrieval Result ---
        final_retrieved_data = [{
            "uuid": n['uuid'],
            "type": n.get('node_type'),
            "final_activation": n.get('final_activation'),
            "saliency_score": n.get('saliency_score'),
            "memory_strength": n.get('memory_strength'),
            "access_count": n.get('access_count'),
            "guaranteed": n.get('guaranteed_inclusion'),
            "text_preview": n.get('text', '')[:50]
        } for n in relevant_nodes]

        log_tuning_event("RETRIEVAL_RESULT", {
            "personality": self.personality,
            "initial_node_uuids": initial_node_uuids, # Include for context
            "activation_threshold": activation_threshold,
            "guaranteed_saliency_threshold": guaranteed_saliency_threshold,
            "final_retrieved_count": len(relevant_nodes),
            "final_retrieved_nodes": final_retrieved_data, # Log detailed info
            "effective_mood": effective_mood, # Log the mood used for retrieval
        })

        # --- Dynamic ASM Check (Experimental) ---
        asm_check_cfg = self.config.get('autobiographical_model', {}).get('dynamic_check', {})
        if asm_check_cfg.get('enable', False) and self.autobiographical_model:
            contradiction_threshold = asm_check_cfg.get('contradiction_saliency_threshold', 0.8)
            contradiction_found = False
            contradicting_node_uuid = None
            node_text = "" # Initialize outside loop
            asm_summary = "" # Initialize outside loop
            node_saliency = 0.0 # Initialize outside loop

            for node_info in relevant_nodes: # Check retrieved nodes
                node_saliency = node_info.get('saliency_score', 0.0)
                # Ensure saliency is numeric for comparison
                node_saliency = node_saliency if isinstance(node_saliency, (int, float)) else 0.0

                if node_saliency >= contradiction_threshold:
                    # Simple check: Does node text contradict ASM summary statement?
                    # This requires an LLM call for robust checking. Placeholder logic:
                    node_text = node_info.get('text', '').lower()
                    asm_summary = self.autobiographical_model.get('summary_statement', '').lower()
                    # Very basic check (e.g., presence of negations or opposite keywords) - Needs improvement!
                    if (" not " in node_text and " not " not in asm_summary) or \
                            (" never " in node_text and " always " in asm_summary): # Example keywords
                        logger.warning(f"Potential ASM contradiction detected! Node {node_info['uuid'][:8]} (Sal: {node_saliency:.2f}) vs ASM Summary.")
                        contradiction_found = True
                        contradicting_node_uuid = node_info['uuid']
                        # --- Log potential contradiction ---
                        log_tuning_event("ASM_CONTRADICTION_DETECTED", {
                            "personality": self.personality,
                            "node_uuid": contradicting_node_uuid,
                            "node_text_preview": node_text[:100],
                            "node_saliency": node_saliency,
                            "asm_summary_preview": asm_summary[:100],
                        })
                        break # Handle first contradiction found for now

            if contradiction_found and contradicting_node_uuid: # Ensure node UUID is set
                # Trigger action: e.g., flag ASM for review, or trigger targeted regeneration
                logger.warning(f"*** Potential ASM Contradiction Detected! *** Node {contradicting_node_uuid[:8]} vs Current ASM.")
                # Simple flag for now, actual regeneration needs more logic
                self.autobiographical_model['needs_review'] = True
                self.autobiographical_model['last_contradiction_node'] = contradicting_node_uuid # Store conflicting node UUID
                self.autobiographical_model['last_contradiction_time'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"ASM flagged for review. Conflicting node: {contradicting_node_uuid[:8]}")
                # --- Log more details about the contradiction ---
                log_tuning_event("ASM_CONTRADICTION_FLAGGED", {
                    "personality": self.personality,
                    "conflicting_node_uuid": contradicting_node_uuid,
                    "conflicting_node_text_preview": node_text[:100],
                    "conflicting_node_saliency": node_saliency,
                    "asm_summary_preview": asm_summary[:100],
                    "asm_state_at_detection": self.autobiographical_model.copy() # Log the ASM state when contradiction found
                })
                # Optionally trigger _generate_autobiographical_model immediately with specific context?
                # self._generate_autobiographical_model(focus_node_uuid=contradicting_node_uuid) # Needs modification to accept focus

        # Return both the nodes and the effective mood used
        return relevant_nodes, effective_mood


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

                            logger.debug(f"Saliency feedback heuristic: Direction='{direction}', Target='{target_drive}', AdjustmentValue={adjustment:.4f}") # Log calculated adjustment

                            if abs(adjustment) > 1e-4 and target_drive in self.drive_state["short_term"]:
                                current_level = self.drive_state["short_term"][target_drive]
                                new_level = current_level + adjustment
                                self.drive_state["short_term"][target_drive] = new_level
                                logger.info(f"Applied heuristic drive adjustment to '{target_drive}' due to saliency feedback ({direction}): {current_level:.3f} -> {new_level:.3f} (Adj: {adjustment:.3f})")
                                # --- Log heuristic adjustment ---
                                log_tuning_event("DRIVE_HEURISTIC_ADJUSTMENT", {
                                    "personality": self.personality,
                                    "trigger": f"saliency_feedback_{direction}",
                                    "node_uuid": uuid, # Log the node involved
                                    "target_drive": target_drive,
                                    "adjustment_value": adjustment,
                                    "level_before": current_level,
                                    "level_after": new_level,
                                })
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
                    # --- Log Feedback Saliency Change ---
                    log_tuning_event("FEEDBACK_SALIENCY_ADJUSTMENT", {
                        "personality": self.personality,
                        "node_uuid": node_uuid,
                        "feedback_type": feedback_type,
                        "saliency_adjustment": saliency_adjustment,
                        "old_saliency": current_saliency,
                        "new_saliency": new_saliency,
                        "old_feedback_score": current_feedback_score,
                        "new_feedback_score": new_feedback_score,
                    })
                else:
                    logger.debug("  Saliency unchanged (at limit or no effective change).")

                # --- NEW: Check if saliency now exceeds core memory threshold ---
                # Check using the potentially updated new_saliency value
                if self.config.get('features', {}).get('enable_core_memory', False):
                    core_threshold = self.config.get('core_memory', {}).get('saliency_threshold', 1.1) # Default > 1 to disable if not set
                    # Fetch the potentially updated saliency score directly from the node data
                    current_saliency_after_update = node_data.get('saliency_score', 0.0)
                    if current_saliency_after_update >= core_threshold and not node_data.get('is_core_memory', False):
                        node_data['is_core_memory'] = True
                        logger.info(f"Node {node_uuid[:8]} flagged as CORE MEMORY due to high saliency ({current_saliency_after_update:.3f} >= {core_threshold:.3f}).")
                        log_tuning_event("CORE_MEMORY_FLAGGED", {
                            "personality": self.personality,
                            "node_uuid": node_uuid,
                            "reason": "high_saliency_feedback",
                            "saliency_score": current_saliency_after_update,
                            "threshold": core_threshold,
                        })
                        # Save happens below anyway

            # Save changes (including potential core memory flag)
            self._save_memory()

        except Exception as e:
            logger.error(f"Error applying feedback or checking core memory flag for node {node_uuid[:8]}: {e}", exc_info=True)


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

        # --- Apply Saliency Decay (Applied to ALL nodes, not just strength candidates) ---
        saliency_decay_rate_per_hour = self.config.get('saliency', {}).get('saliency_decay_rate', 0.0)
        saliency_decayed_count = 0
        if saliency_decay_rate_per_hour > 0:
            logger.info(f"Applying saliency decay (Rate: {saliency_decay_rate_per_hour * 100:.2f}% per hour)...")
            saliency_decay_details = {}
            # Iterate through all nodes again for saliency decay
            for uuid, data in nodes_to_check: # Use the same snapshot
                if uuid not in self.graph: continue # Check if node still exists
                current_saliency = data.get('saliency_score', 0.0)
                if current_saliency <= 0: continue # Skip nodes with no saliency

                last_accessed = data.get('last_accessed_ts', current_time)
                hours_since_access = (current_time - last_accessed) / 3600.0
                if hours_since_access <= 0: continue # Skip if accessed very recently

                # Calculate decay multiplier (simple exponential decay)
                # decay_multiplier = (1.0 - saliency_decay_rate_per_hour) ** hours_since_access # This might decay too fast
                # Let's try linear decay for simplicity: reduction = rate * hours
                saliency_reduction = saliency_decay_rate_per_hour * hours_since_access
                new_saliency = current_saliency - saliency_reduction
                new_saliency = max(0.0, new_saliency) # Clamp at 0

                if new_saliency < current_saliency:
                    self.graph.nodes[uuid]['saliency_score'] = new_saliency
                    nodes_changed = True # Mark that a change occurred
                    saliency_decayed_count += 1
                    saliency_decay_details[uuid] = {
                        "before": current_saliency,
                        "after": new_saliency,
                        "hours_since_access": hours_since_access,
                    }
                    logger.debug(f"  Decayed saliency for node {uuid[:8]} from {current_saliency:.3f} to {new_saliency:.3f} ({hours_since_access:.1f} hrs)")

            if saliency_decay_details:
                log_tuning_event("MAINTENANCE_SALIENCY_DECAY", {
                    "personality": self.personality,
                    "decay_rate_per_hour": saliency_decay_rate_per_hour,
                    "decay_details": saliency_decay_details,
                    "decayed_node_count": saliency_decayed_count,
                })
            logger.info(f"Saliency decay applied to {saliency_decayed_count} nodes.")

        # 5. Save memory if any changes (strength or saliency) were made
        if nodes_changed:
            logger.info(f"Changes detected (Strength: {strength_reduced_count}, Saliency: {saliency_decayed_count}). Saving memory...")
            self._save_memory() # Save changes after maintenance
        else:
            logger.info("No node strengths or saliency scores were changed in this maintenance cycle.")

        # --- Moved Logging Block ---
        logger.info(f"--- Memory Maintenance Finished (Strength Reduced: {strength_reduced_count}, Saliency Decayed: {saliency_decayed_count}) ---")
        # --- Tuning Log: Maintenance End ---
        log_tuning_event("MAINTENANCE_END", { # Renamed event type
            "personality": self.personality,
            "strength_reduced_count": strength_reduced_count,
            "nodes_changed": nodes_changed,
        })
        # --- End Moved Logging Block ---
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

        # --- Apply Saliency Decay (Applied to ALL nodes, not just strength candidates) ---
        saliency_decay_rate_per_hour = self.config.get('saliency', {}).get('saliency_decay_rate', 0.0)
        saliency_decayed_count = 0
        if saliency_decay_rate_per_hour > 0:
            logger.info(f"Applying saliency decay (Rate: {saliency_decay_rate_per_hour * 100:.2f}% per hour)...")
            saliency_decay_details = {}
            # Iterate through all nodes again for saliency decay
            for uuid, data in nodes_to_check: # Use the same snapshot
                if uuid not in self.graph: continue # Check if node still exists
                current_saliency = data.get('saliency_score', 0.0)
                if current_saliency <= 0: continue # Skip nodes with no saliency

                last_accessed = data.get('last_accessed_ts', current_time)
                hours_since_access = (current_time - last_accessed) / 3600.0
                if hours_since_access <= 0: continue # Skip if accessed very recently

                # Calculate decay multiplier (simple exponential decay)
                # decay_multiplier = (1.0 - saliency_decay_rate_per_hour) ** hours_since_access # This might decay too fast
                # Let's try linear decay for simplicity: reduction = rate * hours
                saliency_reduction = saliency_decay_rate_per_hour * hours_since_access
                new_saliency = current_saliency - saliency_reduction
                new_saliency = max(0.0, new_saliency) # Clamp at 0

                if new_saliency < current_saliency:
                    self.graph.nodes[uuid]['saliency_score'] = new_saliency
                    nodes_changed = True # Mark that a change occurred
                    saliency_decayed_count += 1
                    saliency_decay_details[uuid] = {
                        "before": current_saliency,
                        "after": new_saliency,
                        "hours_since_access": hours_since_access,
                    }
                    logger.debug(f"  Decayed saliency for node {uuid[:8]} from {current_saliency:.3f} to {new_saliency:.3f} ({hours_since_access:.1f} hrs)")

            if saliency_decay_details:
                log_tuning_event("MAINTENANCE_SALIENCY_DECAY", {
                    "personality": self.personality,
                    "decay_rate_per_hour": saliency_decay_rate_per_hour,
                    "decay_details": saliency_decay_details,
                    "decayed_node_count": saliency_decayed_count,
                })
            logger.info(f"Saliency decay applied to {saliency_decayed_count} nodes.")

        # 5. Save memory if any changes (strength or saliency) were made
        if nodes_changed:
            logger.info(f"Changes detected (Strength: {strength_reduced_count}, Saliency: {saliency_decayed_count}). Saving memory...")
            self._save_memory() # Save changes after maintenance
        else:
            logger.info("No node strengths or saliency scores were changed in this maintenance cycle.")

        # --- Moved Logging Block ---
        logger.info(f"--- Memory Maintenance Finished (Strength Reduced: {strength_reduced_count}, Saliency Decayed: {saliency_decayed_count}) ---")
        # --- Tuning Log: Maintenance End ---
        log_tuning_event("MAINTENANCE_END", { # Renamed event type
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
        norm_recency_raw = 1.0 - math.exp(-decay_constant * recency_sec) # Score approaches 1 as time increases
        # --- Cap Recency Contribution ---
        # Limit the maximum impact of recency, even after very long breaks.
        # Example: Cap normalized recency at 0.9 to prevent it from reaching 1.0.
        max_norm_recency = weights.get('max_norm_recency_cap', 0.95) # Add this to config if needed, default 0.95
        norm_recency = min(norm_recency_raw, max_norm_recency)
        if norm_recency_raw > max_norm_recency:
            logger.debug(f"    Recency capped for node {node_uuid[:8]}: Raw={norm_recency_raw:.3f} -> Capped={norm_recency:.3f}")

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

        # --- Log Raw and Normalized Factors ---
        log_tuning_event("FORGETTABILITY_FACTORS", {
            "personality": self.personality,
            "node_uuid": node_uuid,
            "node_type": node_type,
            "raw_factors": {
                "recency_sec": recency_sec,
                "activation": activation,
                "saliency": saliency,
                "emotion_magnitude": emotion_magnitude,
                "degree": degree,
                "access_count": access_count,
            },
            "normalized_factors": {
                "norm_recency": norm_recency,
                "norm_inv_activation": norm_inv_activation,
                "norm_type_forgettability": norm_type_forgettability,
                "norm_inv_saliency": norm_inv_saliency,
                "norm_inv_emotion": norm_inv_emotion,
                "norm_inv_connectivity": norm_inv_connectivity,
                "norm_inv_access_count": norm_inv_access_count,
            },
            "weights": weights, # Log the weights used for this calculation
        })

        # Clamp intermediate score 0-1 before applying resistance factors
        intermediate_score = max(0.0, min(1.0, score))
        logger.debug(f"    Forget Score Factors for {node_uuid[:8]}: Rec({norm_recency:.2f}), Act({norm_inv_activation:.2f}), Typ({norm_type_forgettability:.2f}), Sal({norm_inv_saliency:.2f}), Emo({norm_inv_emotion:.2f}), Con({norm_inv_connectivity:.2f}), Acc({norm_inv_access_count:.2f}) -> Intermediate Score: {intermediate_score:.3f}")

        # --- Apply Decay Resistance (Type-Based) ---
        type_resistance_factor = node_data.get('decay_resistance_factor', 1.0)
        score_after_type_resistance = intermediate_score * type_resistance_factor
        logger.debug(f"    Node {node_uuid[:8]} Type Resistance Factor: {type_resistance_factor:.3f}. Score after type resist: {score_after_type_resistance:.4f}")

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
            logger.debug(f"    Node {node_uuid[:8]} Emotion Mag: {emotion_magnitude:.3f} (Norm: {clamped_emo_mag:.3f}), Emo Resist Factor: {emotion_resistance_multiplier:.3f}. Score updated to: {final_adjusted_score:.4f}")

        # --- Apply Core Memory Resistance/Immunity ---
        is_core = node_data.get('is_core_memory', False)
        core_mem_enabled = self.config.get('features', {}).get('enable_core_memory', False)
        core_mem_cfg = self.config.get('core_memory', {})
        core_immunity_enabled = core_mem_cfg.get('forget_immunity', True) # Default immunity to True now

        if core_mem_enabled and is_core:
            if core_immunity_enabled:
                logger.debug(f"    Node {node_uuid[:8]} is Core Memory and immunity is enabled. Setting forgettability to 0.0.")
                final_adjusted_score = 0.0 # Immune to forgetting
            else:
                # Apply a strong resistance factor if immunity is off but it's still core
                core_resistance_factor = weights.get('core_memory_resistance_factor', 0.05) # Use updated default
                final_adjusted_score *= core_resistance_factor
                logger.debug(f"    Node {node_uuid[:8]} is Core Memory (Immunity OFF). Applying resistance factor {core_resistance_factor:.2f}. Score updated to: {final_adjusted_score:.4f}")

        # Final clamp and log before returning
        final_adjusted_score = max(0.0, min(1.0, final_adjusted_score))
        logger.debug(f"    Final calculated forgettability score for {node_uuid[:8]}: {final_adjusted_score:.4f}")

        # --- Log Final Score ---
        log_tuning_event("FORGETTABILITY_FINAL_SCORE", {
            "personality": self.personality,
            "node_uuid": node_uuid,
            "node_type": node_type,
            "intermediate_score": intermediate_score,
            "score_after_type_resistance": score_after_type_resistance,
            "score_after_emotion_resistance": final_adjusted_score if 'emotion_magnitude_resistance_factor' in weights else score_after_type_resistance, # Log score before core check
            "is_core_memory": is_core,
            "core_immunity_enabled": core_immunity_enabled,
            "final_forgettability_score": final_adjusted_score,
            "current_memory_strength": node_data.get('memory_strength', 1.0), # Log current strength for context
        })

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
        # --- Get NEW Purge Criteria from Config ---
        min_purge_age_days = strength_cfg.get('purge_min_age_days', 30) # Default 30 days
        max_purge_saliency = strength_cfg.get('purge_max_saliency', 0.2) # Default max saliency 0.2
        max_purge_access_count = strength_cfg.get('purge_max_access_count', 3) # Default max access count 3
        min_purge_age_seconds = min_purge_age_days * 24 * 3600

        logger.warning(f"--- Purging Weak Nodes (Strength<{purge_threshold}, Age>{min_purge_age_days}d, Sal<{max_purge_saliency}, Access<{max_purge_access_count}) ---")
        # --- Tuning Log: Purge Start ---
        log_tuning_event("PURGE_START", {
            "personality": self.personality,
            "strength_threshold": purge_threshold,
            "min_age_days": min_purge_age_days,
            "max_saliency": max_purge_saliency,
            "max_access_count": max_purge_access_count,
        })

        purge_count = 0
        current_time = time.time()
        nodes_to_purge = []
        nodes_snapshot = list(self.graph.nodes(data=True)) # Snapshot

        purge_check_details = {} # For logging detailed checks

        for uuid, data in nodes_snapshot:
            # --- Check ALL Purge Criteria ---
            current_strength = data.get('memory_strength', 1.0)
            current_saliency = data.get('saliency_score', 0.0)
            current_access_count = data.get('access_count', 0)
            last_accessed = data.get('last_accessed_ts', 0)
            age_seconds = current_time - last_accessed
            is_core = data.get('is_core_memory', False)

            # Store check results for logging
            check_results = {
                "strength_ok": current_strength < purge_threshold,
                "age_ok": age_seconds >= min_purge_age_seconds,
                "saliency_ok": current_saliency < max_purge_saliency,
                "access_count_ok": current_access_count < max_purge_access_count,
                "is_not_core": not is_core,
            }
            purge_check_details[uuid] = {
                "strength": current_strength, "saliency": current_saliency,
                "access_count": current_access_count, "age_days": age_seconds / 86400.0,
                "is_core": is_core, "checks": check_results
            }

            # Check if ALL conditions are met
            if all(check_results.values()):
                nodes_to_purge.append(uuid)
                logger.debug(f"Marked node {uuid[:8]} for purging (Strength: {current_strength:.3f}, Age: {age_seconds/86400.0:.1f}d, Sal: {current_saliency:.3f}, Access: {current_access_count}, Core: {is_core})")
            # else: logger.debug(f"Node {uuid[:8]} did not meet all purge criteria.") # Can be verbose

        # Log detailed checks before deletion
        log_tuning_event("PURGE_CHECKS", {
            "personality": self.personality,
            "check_details": purge_check_details,
            "nodes_marked_for_purge": nodes_to_purge,
        })

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
    def _construct_prompt(self, user_input: str, conversation_history: list, memory_chain: list, tokenizer,
                          max_context_tokens: int, current_mood: tuple[float, float] | None = None,
                          emotional_instructions: str = "") -> str:
        """
        Constructs the prompt for the LLM, incorporating time, memory, history, mood, drives, ASM, and emotional
    instructions.
        Applies memory strength budgeting.
        """
        logger.debug(f"_construct_prompt received user_input: '{user_input}', Mood: {current_mood}")

        if tokenizer is None:
            logger.error("Tokenizer unavailable for prompt construction.")
            # Provide a minimal prompt even without tokenizer
            return f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

        # --- Time formatting ---
        time_str = "[Current time unavailable]"
        try:
            # Use ZoneInfo for proper timezone handling if available (Python 3.9+)
            localtz = ZoneInfo("Europe/Berlin") # Use a known timezone string
        except ImportError:
            logger.warning("zoneinfo module not available, falling back to UTC.")
            localtz = timezone.utc # Fallback to UTC if zoneinfo is not installed
        except Exception as e:
            logger.warning(f"Could not initialize ZoneInfo, falling back to UTC: {e}")
            localtz = timezone.utc # Fallback to UTC on other errors

        try:
            now = datetime.now(localtz)
            time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except Exception as e:
            logger.warning(f"Could not get/format local time: {e}")

        # --- Gemma Instruct format tags ---
        start_turn, end_turn = "<start_of_turn>", "<end_of_turn>"
        user_tag, model_tag = f"{start_turn}user\n", f"{start_turn}model\n"

        # --- Format CURRENT user input ---
        user_input_fmt = f"{user_tag}{user_input}{end_turn}\n"
        logger.debug(f"Formatted current user input (user_input_fmt): '{user_input_fmt[:150]}...'")

        final_model_tag = f"{model_tag}"
        time_info_block = f"{model_tag}Current time is {time_str}.{end_turn}\n"
        asm_block = "" # Initialize ASM block

        # --- Format Structured ASM Block (Using updated fields) ---
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
                # Use new fields
                if self.autobiographical_model.get("goals_motivations"):
                    asm_parts.append(f"- Goals/Motivations: {', '.join(self.autobiographical_model['goals_motivations'])}")
                if self.autobiographical_model.get("relational_stance"):
                    asm_parts.append(f"- My Role: {self.autobiographical_model['relational_stance']}")
                if self.autobiographical_model.get("emotional_profile"):
                    asm_parts.append(f"- Emotional Profile: {self.autobiographical_model['emotional_profile']}")

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
        mem_budget_ratio = prompt_cfg.get('memory_budget_ratio', 0.40)
        hist_budget_ratio = prompt_cfg.get('history_budget_ratio', 0.45)

        # Add the system note to fixed parts
        # Note: System instructions are added later as separate turns for clarity
        system_note_block = "" # Placeholder, actual system notes added later

        # --- Format Drive State Block ---
        drive_block = ""
        if self.config.get('subconscious_drives', {}).get('enabled', False) and self.drive_state:
            try:
                drive_parts = ["[Current Drive State (Internal Motivations - Use to inform response style):]"]
                st_drives = self.drive_state.get("short_term", {})
                lt_drives = self.drive_state.get("long_term", {})
                base_drives = self.config.get('subconscious_drives', {}).get('base_drives', {})
                lt_influence = self.config.get('subconscious_drives', {}).get('long_term_influence_on_baseline', 1.0)

                for drive, st_level in st_drives.items():
                    config_baseline = base_drives.get(drive, 0.0)
                    lt_level = lt_drives.get(drive, 0.0)
                    dynamic_baseline = config_baseline + (lt_level * lt_influence)
                    deviation = st_level - dynamic_baseline
                    state_desc = "Neutral"
                    if deviation > 0.2:
                        state_desc = "High"
                    elif deviation < -0.2:
                        state_desc = "Low"
                    drive_parts.append(f"- {drive}: {state_desc} (Deviation: {deviation:+.2f})")

                if len(drive_parts) > 1:
                    drive_text = "\n".join(drive_parts)
                    drive_block = f"{model_tag}{drive_text}{end_turn}\n"
                    logger.debug("Formatted drive state block created for prompt.")
            except Exception as e:
                logger.error(f"Error formatting drive state for prompt: {e}", exc_info=True)
                drive_block = ""

        # --- Use provided emotional instructions ---
        emotional_instructions_block = ""
        if emotional_instructions:
            emotional_instructions_block = f"{model_tag}{emotional_instructions}{end_turn}\n"
            logger.debug("Adding emotional instructions block (provided) to prompt.")

        # --- Calculate fixed tokens (excluding history, memory, workspace, system instructions) ---
        try:
            fixed_tokens = (len(tokenizer.encode(time_info_block)) +
                            len(tokenizer.encode(asm_block)) +
                            len(tokenizer.encode(drive_block)) +
                            len(tokenizer.encode(emotional_instructions_block)) +
                            len(tokenizer.encode(user_input_fmt)) +
                            len(tokenizer.encode(final_model_tag)))
        except Exception as e:
            logger.error(f"Tokenization error for initial fixed prompt parts: {e}")
            # Estimate if tokenization fails
            fixed_tokens = (len(time_info_block) + len(asm_block) + len(drive_block) +
                            len(emotional_instructions_block) + len(user_input_fmt) + len(final_model_tag))
            logger.warning("Using character count proxy for initial fixed tokens.")

        # --- Estimate tokens for system instructions ---
        # System instructions are added later, but we need to budget for them now.
        system_instructions_text = """
        [System Note: Be aware of the current time provided.]
        [System Note: Pay close attention to sequence and timing.]
        [System Note: **Synthesize** information.]
        [System Note: Handle conversation breaks.]
        [System Note: Use mood/drive state.]
        [System Note: Use 'Self-Perception' summary.]
        [System Note: **CRITICAL: Only use provided context.**]
        [System Note: Action Capability Instructions...]
        [System Note: Handle retrieved intentions.]
        [System Note: CRITICAL - Use <thought> tags.]
        """ # Simplified text for estimation
        try:
            # Estimate tokens for system instructions block (including model tags)
            system_instructions_tokens_estimate = len(tokenizer.encode(f"{model_tag}{system_instructions_text}{end_turn}\n")) * 10 # Rough estimate, multiply by number of instructions
        except Exception:
            system_instructions_tokens_estimate = 300 # Fallback estimate

        # Calculate total available budget AFTER accounting for fixed parts AND system instructions estimate
        total_available_budget = max_context_tokens - fixed_tokens - system_instructions_tokens_estimate - context_headroom
        logger.debug(f"Token counts: Max={max_context_tokens}, Fixed={fixed_tokens}, SysInstrEst={system_instructions_tokens_estimate}, Headroom={context_headroom}, Total Available={total_available_budget}")

        if total_available_budget <= 0:
            logger.warning(f"Low token budget ({total_available_budget}) after fixed parts and system instructions estimate. Only including essentials.")
            # Assemble minimal prompt: Time, User Input, Model Tag (System instructions won't fit)
            final_prompt = time_info_block + user_input_fmt + final_model_tag
            logger.debug(f"Final Prompt (Low Budget): '{final_prompt[:150]}...'")
            return final_prompt # Return early if no budget

        # --- Calculate Budgets (Memory & History first, Workspace gets remainder) ---
        mem_budget = int(total_available_budget * mem_budget_ratio)
        hist_budget = int(total_available_budget * hist_budget_ratio)
        workspace_budget = total_available_budget - mem_budget - hist_budget
        workspace_budget = max(0, workspace_budget) # Ensure non-negative
        mem_budget = max(0, mem_budget) # Ensure non-negative
        hist_budget = max(0, hist_budget) # Ensure non-negative

        logger.debug(f"Budget Allocation: Memory={mem_budget}, History={hist_budget}, Workspace={workspace_budget}")
        log_tuning_event("PROMPT_BUDGETING", {
            "personality": self.personality,
            "user_input_preview": user_input[:100],
            "max_context_tokens": max_context_tokens,
            "fixed_tokens": fixed_tokens,
            "system_instructions_tokens_estimate": system_instructions_tokens_estimate,
            "headroom": context_headroom,
            "total_available_budget": total_available_budget,
            "memory_budget": mem_budget,
            "history_budget": hist_budget,
            "workspace_budget": workspace_budget,
        })

        # --- Workspace Context Construction (Summaries) ---
        workspace_context_str = ""
        cur_workspace_tokens = 0
        logger.debug(f"Constructing Workspace Context (Budget: {workspace_budget})")

        workspace_files, list_msg = file_manager.list_files(self.config, self.personality)
        if workspace_files is None:
            logger.error(f"Failed to list workspace files for context: {list_msg}")
            workspace_context_str = f"{model_tag}[Workspace State: Error retrieving file list]{end_turn}\n"
        elif not workspace_files:
            workspace_context_str = f"{model_tag}[Workspace State: Empty]{end_turn}\n"
        else:
            max_files_to_summarize = self.config.get('prompting', {}).get('max_files_to_summarize_in_context', 5)
            files_to_process = sorted(workspace_files)[:max_files_to_summarize]
            logger.info(f"Processing up to {len(files_to_process)} files for workspace context summary (limit: {max_files_to_summarize}).")

            ws_parts = ["[Workspace State (Filename: Summary):]"]
            ws_header_footer_tags = f"{model_tag}{end_turn}\n"
            try:
                ws_format_tokens = len(tokenizer.encode(ws_header_footer_tags + "\n".join(ws_parts)))
            except Exception:
                ws_format_tokens = 50 # Estimate

            effective_ws_budget = workspace_budget - ws_format_tokens

            for filename in files_to_process:
                summary_line = f"Filename: {filename}\nSummary: [Content unavailable or too long]" # Default
                file_content, read_msg = file_manager.read_file(self.config, self.personality, filename)

                if file_content is not None:
                    max_content_chars = 2000
                    content_for_summary = file_content[:max_content_chars]
                    if len(file_content) > max_content_chars:
                        content_for_summary += "\n... [Content Truncated]"

                    summary = self._summarize_file_content(content_for_summary)
                    if summary:
                        summary_line = f"Filename: {filename}\nSummary: {summary}"
                    else:
                        summary_line = f"Filename: {filename}\nSummary: [Summary generation failed]"
                else:
                    logger.warning(f"Could not read file '{filename}' for context summary: {read_msg}")
                    summary_line = f"Filename: {filename}\nSummary: [Could not read file]"

                try:
                    line_tokens = len(tokenizer.encode(summary_line + "\n---\n"))
                except Exception:
                    line_tokens = len(summary_line) // 3 # Estimate

                if cur_workspace_tokens + line_tokens <= effective_ws_budget:
                    ws_parts.append(summary_line)
                    ws_parts.append("---") # Separator
                    cur_workspace_tokens += line_tokens
                else:
                    logger.warning(f"Workspace context budget ({effective_ws_budget}) reached. Skipping remaining files.")
                    ws_parts.append("[Remaining files omitted due to context length limit]")
                    break

            workspace_content = "\n".join(ws_parts)
            workspace_context_str = f"{model_tag}{workspace_content}{end_turn}\n"
            try:
                cur_workspace_tokens = len(tokenizer.encode(workspace_context_str))
            except Exception:
                cur_workspace_tokens = len(workspace_context_str) // 3 # Estimate
            logger.debug(f"Actual Workspace Tokens Used: {cur_workspace_tokens}")

        # --- Memory Context Construction (with Strength Budgeting) ---
        mem_ctx_str = ""
        core_mem_ctx_str = ""
        cur_mem_tokens = 0
        cur_core_mem_tokens = 0
        mem_header = "---\n[Relevant Past Information - Use this to recall facts (like names) and context]:\n"
        core_header = "---\n[CORE MEMORY - CRITICAL CONTEXT - Adhere Closely]:\n"
        mem_footer = "\n---"
        mem_placeholder_no_mem = "[No relevant memories found or fit budget]"
        mem_placeholder_error = "[Memory Error Processing Context]"
        mem_content = mem_placeholder_no_mem # Default if no memories fit
        core_mem_content = "[No Core Memories Retrieved]" # Default for core memories
        included_mem_uuids = []
        included_core_mem_uuids = []

        if memory_chain and mem_budget > 0:
            core_memories = []
            regular_memories = []
            for node in memory_chain:
                if node.get('is_core_memory', False):
                    core_memories.append(node)
                else:
                    regular_memories.append(node)

            regular_memories.sort(key=lambda x: (x.get('memory_strength', 0.0), x.get('timestamp', '')), reverse=True)
            core_memories.sort(key=lambda x: x.get('timestamp', ''))

            # --- Format Core Memories ---
            core_mem_parts = []
            core_mem_tokens_calc = 0
            try:
                core_format_tokens = len(tokenizer.encode(f"{model_tag}{core_header}{mem_footer}{end_turn}\n")) if core_memories else 0
            except Exception:
                core_format_tokens = 50 if core_memories else 0

            effective_core_budget = mem_budget - core_format_tokens # Budget just for core content

            for node in core_memories:
                spk = node.get('speaker', '?'); txt = node.get('text', ''); ts = node.get('timestamp', '')
                relative_time_desc = self._get_relative_time_desc(ts)
                fmt_mem = f"[CORE] {spk} ({relative_time_desc}): {txt}\n"
                try:
                    mem_tok_len = len(tokenizer.encode(fmt_mem))
                except Exception:
                    continue # Skip if tokenization fails

                if core_mem_tokens_calc + mem_tok_len <= effective_core_budget:
                    core_mem_parts.append(fmt_mem)
                    core_mem_tokens_calc += mem_tok_len
                    included_core_mem_uuids.append(node['uuid'][:8])
                else:
                    logger.warning("Core memory budget reached. Some core memories omitted.")
                    break

            if core_mem_parts:
                core_mem_content = core_header + "".join(core_mem_parts) + mem_footer
                core_mem_ctx_str = f"{model_tag}{core_mem_content}{end_turn}\n"
                try:
                    cur_core_mem_tokens = len(tokenizer.encode(core_mem_ctx_str))
                except Exception:
                    cur_core_mem_tokens = len(core_mem_ctx_str) // 3 # Estimate
            else:
                core_mem_ctx_str = ""
                cur_core_mem_tokens = 0


            # Update remaining budget for regular memories
            remaining_mem_budget = max(0, mem_budget - cur_core_mem_tokens)

            # --- Format Regular Memories ---
            mem_parts = []
            regular_mem_tokens_calc = 0
            try:
                format_tokens = len(tokenizer.encode(f"{model_tag}{mem_header}{mem_footer}{end_turn}\n")) if regular_memories else 0
            except Exception:
                format_tokens = 50 if regular_memories else 0

            effective_mem_budget = remaining_mem_budget - format_tokens

            mem_parts_with_ts = [] # Store (timestamp, formatted_string) for sorting later

            for node in regular_memories:
                spk = node.get('speaker', '?'); txt = node.get('text', ''); ts = node.get('timestamp', '')
                strength = node.get('memory_strength', 0.0)
                saliency = node.get('saliency_score', 0.0)
                relative_time_desc = self._get_relative_time_desc(ts)
                importance_marker = "[IMPORTANT] " if saliency >= 0.9 else ""

                max_chars_for_node = 1000
                if strength < 0.3: max_chars_for_node = 80
                elif strength < 0.6: max_chars_for_node = 200

                truncated_txt = txt[:max_chars_for_node]
                if len(txt) > max_chars_for_node: truncated_txt += "..."

                fmt_mem = f"{importance_marker}{spk} ({relative_time_desc}) [Str: {strength:.2f}]: {truncated_txt}\n"
                try:
                    mem_tok_len = len(tokenizer.encode(fmt_mem))
                except Exception as e:
                    logger.warning(f"Tokenization error for memory item: {e}. Skipping.")
                    continue

                if regular_mem_tokens_calc + mem_tok_len <= effective_mem_budget:
                    # Store with timestamp for later sorting
                    mem_parts_with_ts.append((ts, fmt_mem))
                    regular_mem_tokens_calc += mem_tok_len
                    included_mem_uuids.append(node['uuid'][:8]) # Track included regular memories
                else:
                    logger.debug("Regular memory budget reached during context construction.")
                    break # Stop processing regular memories

            if mem_parts_with_ts:
                # Sort the included regular memories chronologically
                mem_parts_with_ts.sort(key=lambda item: item[0])
                sorted_mem_parts = [item[1] for item in mem_parts_with_ts]
                mem_content = mem_header + "".join(sorted_mem_parts) + mem_footer
                mem_ctx_str = f"{model_tag}{mem_content}{end_turn}\n"
                try:
                    cur_mem_tokens = len(tokenizer.encode(mem_ctx_str))
                except Exception:
                    cur_mem_tokens = len(mem_ctx_str) // 3 # Estimate
            elif not core_mem_ctx_str: # Only add placeholder if NO memories (core or regular) were added
                mem_ctx_str = f"{model_tag}{mem_placeholder_no_mem}{end_turn}\n"
                try:
                    cur_mem_tokens = len(tokenizer.encode(mem_ctx_str))
                except Exception:
                    cur_mem_tokens = 50 # Estimate
            else:
                mem_ctx_str = "" # No regular memories, but core memories exist
                cur_mem_tokens = 0

        else: # No memory chain provided or zero memory budget
            if mem_budget <= 0: logger.debug("Memory budget is zero, skipping memory context.")
            else: logger.debug("No memory chain provided.")
            mem_ctx_str = f"{model_tag}{mem_placeholder_no_mem}{end_turn}\n"
            try:
                cur_mem_tokens = len(tokenizer.encode(mem_ctx_str))
            except Exception:
                cur_mem_tokens = 50 # Estimate
            core_mem_ctx_str = ""
            cur_core_mem_tokens = 0

        # Log included UUIDs
        if included_core_mem_uuids: logger.debug(f"Included Core Memory UUIDs: {included_core_mem_uuids}")
        if included_mem_uuids: logger.debug(f"Included Regular Memory UUIDs (chrono): {included_mem_uuids}") # Log the tracked UUIDs

        total_mem_tokens_used = cur_core_mem_tokens + cur_mem_tokens
        original_total_mem_budget = int(total_available_budget * mem_budget_ratio)
        if total_mem_tokens_used > original_total_mem_budget + 10:
            logger.warning(f"Final memory tokens ({total_mem_tokens_used}) exceed original budget ({original_total_mem_budget}). Check logic.")
        logger.debug(f"Total Memory Tokens Used: {total_mem_tokens_used} (Core: {cur_core_mem_tokens}, Regular: {cur_mem_tokens})")

        # --- History Context Construction ---
        # Recalculate remaining budget for history AFTER memory and workspace are finalized
        remaining_budget_for_hist = total_available_budget - total_mem_tokens_used - cur_workspace_tokens
        hist_budget = max(0, remaining_budget_for_hist) # History gets whatever is left

        logger.debug(f"Re-calculated History Budget: {hist_budget}")

        hist_parts = []
        cur_hist_tokens = 0
        included_hist_count = 0

        history_to_process = conversation_history

        if history_to_process and hist_budget > 0:
            for turn in reversed(history_to_process):
                spk = turn.get('speaker', '?')
                txt = turn.get('text', '')
                logger.debug(f"Processing history turn: Speaker={spk}, Text='{txt[:80]}...'")

                if spk == 'User':
                    fmt_turn = f"{user_tag}{txt}{end_turn}\n"
                elif spk in ['AI', 'System', 'Error']:
                    fmt_turn = f"{model_tag}{txt}{end_turn}\n"
                else:
                    logger.warning(f"Unknown speaker '{spk}' in history, skipping.")
                    continue

                try:
                    turn_tok_len = len(tokenizer.encode(fmt_turn))
                except Exception as e:
                    logger.warning(f"Tokenization error for history turn: {e}. Skipping.")
                    continue

                if cur_hist_tokens + turn_tok_len <= hist_budget:
                    hist_parts.append(fmt_turn)
                    cur_hist_tokens += turn_tok_len
                    included_hist_count += 1
                else:
                    logger.debug("History budget reached.")
                    break

            hist_parts.reverse()
            history_context_for_log = "".join(hist_parts)
            logger.debug(f"Included history ({cur_hist_tokens} tokens / {included_hist_count} turns):\n--- START HISTORY CONTEXT ---\n{history_context_for_log}\n--- END HISTORY CONTEXT ---")

        # --- Assemble Final Prompt ---
        # Order: Time, Workspace, System Instructions, Emo Instructions, ASM, Drives, Core Mem, Regular Mem, History, User Input, Model Tag
        final_parts = []
        final_parts.append(time_info_block)
        if workspace_context_str:
            final_parts.append(workspace_context_str)

        # Add system instructions (as separate model turns)
        system_instructions = [
            # General Context Instructions
            "[System Note: Be aware of the current time provided. Use it to inform your responses when relevant (e.g., acknowledging time of day, interpreting time-sensitive requests).]",
            "[System Note: Pay close attention to the sequence and relative timing ('X minutes ago', 'yesterday', etc.) of the provided memories and conversation history to maintain context.]",
            "[System Note: **Synthesize** the information from the 'Relevant Past Information' (memories), 'Conversation History', and your 'Self-Perception' summary to generate a **specific and personalized** response relevant to the current user query. Avoid generic templates or merely listing possibilities if the context provides specific reasons.]",
            "[System Note: When resuming a conversation after a break (indicated by timestamps or a re-greeting message from you in the history), ensure your response considers the context from *before* the break as well as the user's latest message. Avoid asking questions already answered in the provided history.]",
            # Mood/Drive Tone Instruction
            f"[System Note: Your current internal state is reflected in the 'Current Drive State' block. Your calculated mood is Valence={current_mood[0]:.2f} (Pleasantness) and Arousal={current_mood[1]:.2f} (Energy). **Actively use this state** to shape your response's tone, word choice, and even content focus. For example:\n"
            f"  - High Connection need (Low deviation): Be more engaging, seek common ground.\n"
            f"  - High Safety need (Low deviation): Be more cautious, seek reassurance, avoid ambiguity.\n"
            f"  - High Control need (Low deviation): Be more proactive, structured, offer clear steps.\n"
            f"  - High Valence (Happy/Content): Use warmer, more positive language.\n"
            f"  - High Arousal (Excited/Agitated): Use more energetic or intense language (appropriately).\n"
            f"  - Low Arousal (Calm/Sad): Use calmer or more subdued language.]" if current_mood else "[System Note: Current mood unavailable.]",
            # ASM Integration Instruction (Revised for Adaptation)
            "[System Note: Use your 'Self-Perception' summary (Traits, Goals, Role, etc.) as a baseline understanding of yourself. **Explicitly reference** how your traits or goals inform your current thinking or response when relevant to the user's query. However, **adapt your immediate response** based on your current Mood, Drive State deviations, and the immediate context of recent Memories and History. Note any significant shifts or contradictions observed. Prioritize recent information when it conflicts with the baseline summary.]",
            # ANTI-HALLUCINATION INSTRUCTION
            "[System Note: **CRITICAL: Only use information explicitly provided in the 'Relevant Past Information' (memories) or 'Conversation History' context.** If you cannot recall specific details based *only* on the provided context, state that you do not remember or cannot find that information (e.g., 'I don't recall the specifics of that event.'). **DO NOT INVENT details, events, or memories.** Ground your response firmly and exclusively in the given context.]",
            # Action Capability Instructions
            "[System Note: You have the ability to manage files and calendar events.",
            "  To request an action, end your *entire* response with a special tag: `[ACTION: {\"action\": \"action_name\", \"args\": {\"arg1\": \"value1\", ...}}]`.",
            "  **Available Actions:** `create_file`, `append_file`, `list_files`, `read_file`, `delete_file`, `consolidate_files`, `add_calendar_event`, `read_calendar`.",
            "  **CRITICAL: `edit_file` is NOT a valid action.** Use `read_file` then `create_file` (overwrites).",
            "  **To Edit a File:** Use `read_file` then `create_file` (overwrites).",
            "  **Using Actions:**",
            "    - For `list_files`, `read_calendar`: Use `[ACTION: {\"action\": \"action_name\", \"args\": {}}]` (or add optional 'date' arg for read_calendar).",
            "    - For `read_file`, `delete_file`: Use `[ACTION: {\"action\": \"action_name\", \"args\": {\"filename\": \"target_file.txt\"}}]`.",
            "    - For `append_file`: Use `[ACTION: {\"action\": \"append_file\", \"args\": {\"filename\": \"target_file.txt\", \"content\": \"Text to append...\"}}]` (Generate the actual content to append).",
            "    - For `add_calendar_event`: Use `[ACTION: {\"action\": \"add_calendar_event\", \"args\": {\"date\": \"YYYY-MM-DD\", \"time\": \"HH:MM\", \"description\": \"Event details...\"}}]`.",
            "    - **For `create_file`:** Signal your *intent* by providing a brief description. The system will handle filename/content generation separately. Use `[ACTION: {\"action\": \"create_file\", \"args\": {\"description\": \"Brief description of what to save, e.g., 'List of project ideas'\"}}]`.",
            "  Only use the ACTION tag if you decide an action is necessary based on the context.]",
            # NEW: Instruction for handling retrieved intentions
            "[System Note: If you see a retrieved memory starting with 'Remember:' (indicating a stored intention), check if the trigger condition seems relevant to the current conversation. If so, incorporate the reminder into your response or perform the implied task if appropriate (potentially using the ACTION tag).]",
            # Thought Process Instruction
            "[System Note: CRITICAL - Before generating your response to the user, first output your internal thought process, rationale, or step-by-step thinking relevant to formulating the response. Enclose these thoughts STRICTLY within <thought> and </thought> tags. After the closing </thought> tag, provide the final conversational response meant for the user. The final response itself should NOT contain the <thought> tags or directly refer to 'inner thoughts' unless appropriate for the persona/context.]",
        ]
        for instruction in system_instructions:
            final_parts.append(f"{model_tag}{instruction}{end_turn}\n")

        # --- Add Emotional Instructions Block (if generated) ---
        if emotional_instructions_block:
            final_parts.append(emotional_instructions_block)

        # --- Add Other Context Blocks ---
        if asm_block: final_parts.append(asm_block)
        if drive_block: final_parts.append(drive_block)
        if core_mem_ctx_str: final_parts.append(core_mem_ctx_str) # Add Core Memories
        if mem_ctx_str: final_parts.append(mem_ctx_str) # Add Regular Memories (or placeholder)
        final_parts.extend(hist_parts)
        final_parts.append(user_input_fmt)
        final_parts.append(final_model_tag)
        final_prompt = "".join(final_parts)

        # --- Final Logging and Checks ---
        logger.debug("--- Final Prompt Structure ---")
        if len(final_prompt) > 500:
            logger.debug(f"{final_prompt[:250]}...\n...\n...{final_prompt[-250:]}")
        else:
            logger.debug(final_prompt)
        logger.debug("--- End Final Prompt Structure ---")

        final_tok_count = -1 # Initialize
        try:
            final_tok_count = len(tokenizer.encode(final_prompt))
            # Use the originally calculated total_available_budget for logging comparison
            logger.info(f"Constructed prompt final token count: {final_tok_count} (Initial Budget Available: {total_available_budget + system_instructions_tokens_estimate})") # Log against initial budget before sys instr estimate
            if final_tok_count > max_context_tokens:
                logger.error(f"CRITICAL: Final prompt ({final_tok_count} tokens) EXCEEDS max context ({max_context_tokens}).")
            elif final_tok_count > max_context_tokens - context_headroom:
                logger.warning(f"Final prompt ({final_tok_count} tokens) close to max context ({max_context_tokens}). Less headroom ({max_context_tokens - final_tok_count}).")
        except Exception as e:
            logger.error(f"Tokenization error for final prompt: {e}")

        # --- Write final prompt to debug log file ---
        try:
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            last_prompt_file = os.path.join(log_dir, 'last_prompt.txt')
            with open(last_prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"--- Prompt for Personality: {self.personality} at {datetime.now(timezone.utc).isoformat()} ---\n\n")
                f.write(final_prompt)
            logger.debug(f"Wrote final prompt to {last_prompt_file}")
        except Exception as e:
            logger.error(f"Failed to write last_prompt.txt: {e}", exc_info=True)

        log_tuning_event("PROMPT_CONSTRUCTION_RESULT", {
            "personality": self.personality,
            "user_input_preview": strip_emojis(user_input[:100]),
            "included_core_memory_uuids": included_core_mem_uuids,
            "included_regular_memory_uuids": included_mem_uuids, # Log regular UUIDs
            "included_history_turns": included_hist_count,
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
            logger.error(f"Action analysis failed due to LLM API error: {error_reason}")
            return {'action': 'error', 'reason': error_reason}

        parsed_result = None
        json_str = "" # Initialize json_str for potential use in error logging
        try:
            logger.debug(f"Raw action analysis response:  ```{llm_response_str}```")
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

    # --- DEPRECATED ---
    # def analyze_action_request(self, request_text: str) -> dict:
    #     """ DEPRECATED: Use workspace planning prompt instead. """
    #     logger.warning("analyze_action_request is deprecated. Use workspace planning.")
    #     return {"action": "none"}

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


    def execute_plan(self, plan: list) -> list[tuple[bool, str, str, bool]]:
        """
        Executes a list of planned actions sequentially.

        Args:
            plan: A list of action dictionaries, e.g.,
                  [{"action": "action_name", "args": {...}}, ...]

        Returns:
            A list of result tuples for each executed action:
            [(success: bool, message: str, action_suffix: str, silent_and_successful: bool), ...]
            Execution stops after the first failure.
        """
        results = []
        if not isinstance(plan, list):
            logger.error(f"Invalid plan format received: Expected list, got {type(plan)}")
            results.append((False, "Internal Error: Invalid plan format received.", "plan_error", False))
            return results

        logger.info(f"Executing workspace plan with {len(plan)} step(s)...")

        for i, step in enumerate(plan):
            final_result_tuple_4 = None
            action_name = "unknown"
            args = {}
            is_silent_request = False

            try:
                if not isinstance(step, dict) or "action" not in step or "args" not in step:
                    logger.error(f"Invalid action step format in plan (Step {i+1}): {step}")
                    final_result_tuple_4 = (False, f"Internal Error: Invalid action format in step {i+1}.", "step_error", False)
                    # <<< DEBUG POINT >>>
                    logger.debug(f"[DEBUG] Step {i+1} Error (Format): Assigning final_result_tuple_4: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}")
                else:
                    action_name = step.get("action", "unknown")
                    args = step.get("args", {})
                    is_silent_request = args.get("silent", False) is True
                    logger.info(f"Executing Step {i+1}/{len(plan)}: Action='{action_name}', Args={args}, Silent={is_silent_request}")

                    action_result_tuple_3 = None # Stores result from successful helper call

                    # --- Dispatch to helper methods (Return 3-tuple on success) ---
                    if action_name == "create_file":
                        action_result_tuple_3 = self._execute_create_file(args)
                    elif action_name == "append_file":
                        action_result_tuple_3 = self._execute_append_file(args)
                    elif action_name == "list_files":
                        action_result_tuple_3 = self._execute_list_files(args)
                    elif action_name == "read_file":
                        action_result_tuple_3 = self._execute_read_file(args)
                    elif action_name == "delete_file":
                        action_result_tuple_3 = self._execute_delete_file(args)
                    elif action_name == "add_calendar_event":
                        action_result_tuple_3 = self._execute_add_calendar_event(args)
                    elif action_name == "read_calendar":
                        action_result_tuple_3 = self._execute_read_calendar(args)
                    elif action_name == "consolidate_files":
                        action_result_tuple_3 = self._execute_consolidate_files(args)
                    else:
                        logger.error(f"Unsupported action '{action_name}' in plan (Step {i+1}).")
                        final_result_tuple_4 = (False, f"Error: Action '{action_name}' is not supported.", f"{action_name}_unsupported", False)
                        # <<< DEBUG POINT >>>
                        logger.debug(f"[DEBUG] Step {i+1} Error (Unsupported): Assigning final_result_tuple_4: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}")

                    # --- Process successful helper result ---
                    if action_result_tuple_3 is not None:
                        # <<< DEBUG POINT >>>
                        logger.debug(f"[DEBUG] Step {i+1} Success: Helper returned action_result_tuple_3: Type={type(action_result_tuple_3)}, Value={action_result_tuple_3}")
                        if isinstance(action_result_tuple_3, tuple) and len(action_result_tuple_3) == 3:
                            success, message, action_suffix = action_result_tuple_3
                            silent_and_successful = success and is_silent_request
                            final_result_tuple_4 = (success, message, action_suffix, silent_and_successful)
                            # <<< DEBUG POINT >>>
                            logger.debug(f"[DEBUG] Step {i+1} Success: Constructed final_result_tuple_4: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4)}")
                        else:
                            # This case should ideally not happen if helpers are correct
                            logger.error(f"!!! Helper for '{action_name}' returned unexpected value: {action_result_tuple_3}. Expected 3-tuple. !!!")
                            final_result_tuple_4 = (False, f"Internal Error: Helper for '{action_name}' returned invalid data.", f"{action_name}_helper_error", False)
                            # <<< DEBUG POINT >>>
                            logger.debug(f"[DEBUG] Step {i+1} Error (Helper Return): Assigning final_result_tuple_4: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}")

            except Exception as e:
                logger.error(f"Unexpected exception executing plan step {i+1} (Action: {action_name}): {e}", exc_info=True)
                # Construct the 4-tuple error result directly
                final_result_tuple_4 = (False, f"Internal error during execution of '{action_name}': {e}", f"{action_name}_exception", False)
                # <<< DEBUG POINT >>>
                logger.debug(f"[DEBUG] Step {i+1} Error (Exception): Assigning final_result_tuple_4: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}")

            # --- Append the final 4-tuple result ---
            if final_result_tuple_4 is not None:
                # <<< DEBUG POINT >>>
                logger.debug(f"[DEBUG] Step {i+1} Appending: Checking final_result_tuple_4 before append: Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}, Value={str(final_result_tuple_4)[:200]}...") # Log preview
                # >>> FINAL CHECK <<<
                if not isinstance(final_result_tuple_4, tuple) or len(final_result_tuple_4) != 4:
                    logger.critical(f"CRITICAL ERROR: Attempting to append a non 4-tuple to results! Type={type(final_result_tuple_4)}, Len={len(final_result_tuple_4) if isinstance(final_result_tuple_4, tuple) else 'N/A'}, Value={final_result_tuple_4}")
                    # Fallback to a generic error tuple to avoid crashing the caller later
                    results.append((False, "Internal Agent Error: Invalid tuple generated.", "internal_tuple_error", False))
                    break # Stop processing plan
                else:
                    # Append the validated 4-tuple
                    results.append(final_result_tuple_4)
                    # Check success flag (index 0) to stop on failure
                    if not final_result_tuple_4[0]:
                        logger.warning(f"Plan execution stopped at step {i+1} due to failure.")
                        break
            else:
                logger.error(f"Result tuple was None for step {i+1} (Action: {action_name}). Stopping plan.")
                results.append((False, f"Internal Error: No result generated for action '{action_name}'.", f"{action_name}_internal_error", False))
                break

        logger.info(f"Workspace plan execution finished. {len(results)} step(s) attempted.")
        silent_success_count = sum(1 for r in results if r[3])
        if silent_success_count > 0:
            logger.info(f"  {silent_success_count} action(s) were executed silently and successfully.")
        return results

    
    
    # --- Need to add the modified _apply_heuristic_drive_adjustment method ---
    def _apply_heuristic_drive_adjustment(self, target_drive: str, adjustment_value: float, trigger_reason: str, context_uuid: str | None = None):
        """Applies a direct adjustment to a short-term drive and logs it."""
        if not target_drive or abs(adjustment_value) < 1e-4:
            return

        if target_drive in self.drive_state["short_term"]:
            current_level = self.drive_state["short_term"][target_drive]
            new_level = current_level + adjustment_value
            # Optional clamping?
            self.drive_state["short_term"][target_drive] = new_level
            logger.info(f"Applied heuristic drive adjustment to '{target_drive}' due to '{trigger_reason}': {current_level:.3f} -> {new_level:.3f} (Adj: {adjustment_value:.3f})")
            # --- Log heuristic adjustment ---
            log_tuning_event("DRIVE_HEURISTIC_ADJUSTMENT", {
                "personality": self.personality,
                "trigger": trigger_reason,
                "context_node_uuid": context_uuid, # Add context node
                "target_drive": target_drive,
                "adjustment_value": adjustment_value,
                "level_before": current_level,
                "level_after": new_level,
            })
            # Save state? Let's assume saving happens at end of interaction or consolidation.
        else:
            logger.warning(f"Cannot apply heuristic adjustment: Target drive '{target_drive}' not found in state.")


    def _update_next_interaction_context(self, user_node_uuid: str | None, ai_node_uuid: str | None):
        """Helper to calculate and store concept/mood context for the *next* interaction's bias."""
        current_turn_concept_uuids = set()
        nodes_to_check_for_concepts = [uuid for uuid in [user_node_uuid, ai_node_uuid] if uuid] # Filter out None UUIDs

        # --- Find Concepts Mentioned in Current Turn ---
        for turn_uuid in nodes_to_check_for_concepts:
            if turn_uuid in self.graph:
                try:
                    # Check outgoing edges for MENTIONS_CONCEPT
                    for successor_uuid in self.graph.successors(turn_uuid):
                        edge_data = self.graph.get_edge_data(turn_uuid, successor_uuid)
                        if edge_data and edge_data.get('type') == 'MENTIONS_CONCEPT':
                            if successor_uuid in self.graph and self.graph.nodes[successor_uuid].get('node_type') == 'concept':
                                current_turn_concept_uuids.add(successor_uuid)
                except Exception as concept_find_e:
                     logger.warning(f"Error finding concepts linked from turn {turn_uuid[:8]} for next bias: {concept_find_e}")

        logger.info(f"Storing {len(current_turn_concept_uuids)} concepts for next interaction's bias.")
        self.last_interaction_concept_uuids = current_turn_concept_uuids # Update state

        # --- Calculate Average Mood of Current Turn ---
        current_turn_mood = (0.0, 0.1) # Default: Neutral valence, low arousal
        mood_nodes_found = 0
        total_valence = 0.0
        total_arousal = 0.0
        node_moods_for_avg = {} # Log individual moods
        for node_uuid in nodes_to_check_for_concepts: # Iterate over the same nodes used for concepts
            if node_uuid in self.graph:
                node_data = self.graph.nodes[node_uuid]
                default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                node_v = node_data.get('emotion_valence', default_v)
                node_a = node_data.get('emotion_arousal', default_a)
                node_moods_for_avg[node_uuid[:8]] = {"V": node_v, "A": node_a} # Log individual node mood
                total_valence += node_v
                total_arousal += node_a
                mood_nodes_found += 1
        if mood_nodes_found > 0:
            current_turn_mood = (total_valence / mood_nodes_found, total_arousal / mood_nodes_found)

        logger.info(f"Storing mood (Avg V/A): {current_turn_mood[0]:.2f} / {current_turn_mood[1]:.2f} for next interaction's bias. Storing in self.last_interaction_mood.") # Added detail
        # --- Log mood averaging details ---
        log_tuning_event("MOOD_AVERAGING_UPDATE", {
            "personality": self.personality,
            "nodes_averaged": list(node_moods_for_avg.keys()),
            "individual_node_moods": node_moods_for_avg,
            "total_valence": total_valence,
            "total_arousal": total_arousal,
            "mood_nodes_found": mood_nodes_found,
            "final_averaged_mood": current_turn_mood,
        })
        self.last_interaction_mood = current_turn_mood # Update state


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
            files_to_delete = [
                self.graph_file, self.index_file, self.embeddings_file, self.mapping_file,
                self.asm_file, self.drives_file, self.last_conversation_file # Add last conversation file
            ]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    try: os.remove(file_path); logger.info(f"Deleted: {file_path}")
                    except OSError as e: logger.error(f"Error deleting {file_path}: {e}")
                else: logger.debug(f"Not found, skip delete: {file_path}")
            # Re-initialize drive state and last conversation turns after deleting files
            self._initialize_drive_state()
            self.last_conversation_turns = [] # Clear the list
            logger.info("--- MEMORY RESET COMPLETE ---"); return True
        except Exception as e:
            logger.error(f"Error during memory reset: {e}", exc_info=True)
            # Ensure drive state and last conversation are also reset in case of error
            self.graph = nx.DiGraph(); self.embeddings = {}; self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}; self.last_added_node_uuid = None; self.index = faiss.IndexFlatL2(self.embedding_dim); self._initialize_drive_state(); self.last_conversation_turns = []; logger.warning("Reset failed, re-initialized empty state.")
            return False

    def _load_initial_history(self, count=3):
        """DEPRECATED: Initial history is now loaded from self.last_conversation_turns."""
        # This method is now effectively a no-op, as the data is loaded
        # by _load_last_conversation() during __init__.
        # The time gap calculation and re-greeting check are also moved to __init__.
        logger.debug("_load_initial_history called, but logic moved to __init__ / _load_last_conversation.")
        pass # Keep the method signature but do nothing here.

    def _calculate_time_since_last_interaction(self):
        """Calculates time gap based on self.last_conversation_turns."""
        self.time_since_last_interaction_hours = 0.0 # Reset before calculation
        if self.last_conversation_turns:
            try:
                # Use the last turn from the separately stored list
                last_turn_timestamp_str = self.last_conversation_turns[-1].get("timestamp")
                if last_turn_timestamp_str:
                    last_turn_dt = datetime.fromisoformat(last_turn_timestamp_str.replace('Z', '+00:00'))
                    now_dt = datetime.now(timezone.utc)
                    time_delta = now_dt - last_turn_dt
                    self.time_since_last_interaction_hours = time_delta.total_seconds() / 3600.0
                    logger.info(f"Time since last interaction: {self.time_since_last_interaction_hours:.2f} hours.")
                else:
                    logger.warning("Last turn in initial history has no timestamp.")
                    self.time_since_last_interaction_hours = 0.0
            except Exception as e:
                logger.error(f"Error calculating time since last interaction: {e}", exc_info=True)
                self.time_since_last_interaction_hours = 0.0
        else:
            self.time_since_last_interaction_hours = 0.0 # No history, so no gap

    def get_initial_history(self, count=3) -> list:
        """Returns the last 'count' turns from the separately stored conversation history."""
        # Return a copy of the relevant slice to prevent external modification
        return self.last_conversation_turns[-count:].copy() if self.last_conversation_turns else []

    def _check_and_generate_re_greeting(self):
        """Checks time gap and generates re-greeting if needed during initialization."""
        self.pending_re_greeting = None # Ensure it's clear initially
        re_greeting_threshold = self.config.get('prompting', {}).get('re_greeting_threshold_hours', 3.0)
        logger.debug(f"Checking re-greeting on init: TimeGap={self.time_since_last_interaction_hours:.2f}h, Threshold={re_greeting_threshold}h")

        if self.time_since_last_interaction_hours > re_greeting_threshold:
            logger.info(f"Time gap ({self.time_since_last_interaction_hours:.2f}h) exceeds threshold ({re_greeting_threshold}h). Generating re-greeting during init.")
            # --- Tuning Log: Re-Greeting Triggered (Init) ---
            log_tuning_event("RE_GREETING_TRIGGERED_INIT", {
                "personality": self.personality,
                "time_gap_hours": self.time_since_last_interaction_hours,
                "threshold_hours": re_greeting_threshold,
            })

            # Prepare context for re-greeting prompt using last_conversation_turns
            last_messages_context = "\n".join([f"- {turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}" for turn in self.last_conversation_turns[-3:]]) # Use last_conversation_turns
            asm_summary = self.autobiographical_model.get("summary_statement", "[No self-summary available]")

            re_greeting_prompt_template = self._load_prompt("re_greeting_prompt.txt")
            if not re_greeting_prompt_template:
                logger.error("Re-greeting prompt template missing. Cannot generate greeting.")
                return # Exit if prompt missing

            re_greeting_prompt = re_greeting_prompt_template.format(
                time_gap_hours=self.time_since_last_interaction_hours,
                last_messages_context=last_messages_context,
                asm_summary=asm_summary
            )
            logger.debug(f"Sending re-greeting prompt (init):\n{re_greeting_prompt}")
            # Call LLM using dedicated config
            ai_response = self._call_configured_llm('re_greeting_generation', prompt=re_greeting_prompt)
            parsed_response = ai_response.strip() if ai_response and not ai_response.startswith("Error:") else "Hello again! It's been a while." # Fallback greeting

            # Store the generated greeting
            self.pending_re_greeting = parsed_response
            logger.info(f"Generated and stored pending re-greeting: '{self.pending_re_greeting[:50]}...'")
        else:
            logger.debug("Time gap does not exceed threshold. No re-greeting needed on init.")

    def get_pending_re_greeting(self) -> str | None:
        """Returns the pending re-greeting message (if any) and clears it."""
        greeting = self.pending_re_greeting
        self.pending_re_greeting = None # Clear after retrieval
        return greeting

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

    # --- Signature changed to accept context_text ---
    def _update_drive_state(self, context_text: str = ""):
        """
        Updates drive activation levels based on recent experience (context_text).
        Applies decay and potentially LLM analysis based on the provided text.
        """

        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False):
            return # Do nothing if disabled

        logger.debug("Running short-term drive state update (Decay + LLM)...")
        decay_rate = drive_cfg.get('short_term_decay_rate', 0.05)
        base_drives = drive_cfg.get('base_drives', {})
        long_term_influence = drive_cfg.get('long_term_influence_on_baseline', 1.0)
        changed = False
        drive_state_before_update = self.drive_state.copy() # Log initial state

        # --- Decay Step (towards dynamic baseline) ---
        if decay_rate > 0:
            decay_details = {}
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
                    decay_details[drive_name] = {
                        "before": current_activation,
                        "after": new_activation,
                        "dynamic_baseline": dynamic_baseline,
                    }
                    logger.debug(f"  Drive '{drive_name}' decayed towards dynamic baseline {dynamic_baseline:.3f}: {current_activation:.3f} -> {new_activation:.3f}")
            if decay_details:
                log_tuning_event("DRIVE_STATE_DECAY", {
                    "personality": self.personality,
                    "decay_rate": decay_rate,
                    "decay_details": decay_details,
                })

        # --- LLM Analysis for Short-Term Drive Satisfaction/Frustration ---
        # LLM analysis is now triggered either by interval OR by high emotional impact event
        update_interval = drive_cfg.get('short_term_update_interval_interactions', 0)
        trigger_on_high_impact = drive_cfg.get('trigger_drive_update_on_high_impact', False)
        # Determine if LLM analysis should run for this update cycle
        run_llm_analysis = False
        if context_text and update_interval > 0: # Check interval trigger
            # Need interaction counter - assume it's passed or accessible (e.g., self.interaction_count)
            # This logic needs refinement based on where interaction count is tracked.
            # For now, let's assume it runs if context is provided and interval > 0.
            # A better approach would be to pass an 'is_interval_trigger' flag.
            # Let's simplify: Run if context_text is provided (meaning it's likely an interval or event trigger)
            run_llm_analysis = True
            logger.info("Attempting LLM analysis for drive state update (context provided)...")
        elif trigger_on_high_impact and self.high_impact_nodes_this_interaction:
            # Triggered by high impact event, even without explicit context_text here?
            # This implies _update_drive_state needs to be called immediately after such an interaction.
            # Let's assume the calling function provides context_text if triggering based on event.
            # So, the check `if context_text:` below handles both cases if called correctly.
            pass # Logic handled by context_text check below

        if context_text: # Run if context is provided (for interval or event trigger)
            logger.info("Attempting LLM analysis for drive state update...")
            try:
                # 1. Context is already provided as context_text
                if not context_text.strip():
                     logger.warning("Received empty context_text for drive analysis.")
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
                        # --- Log prompt sent to LLM ---
                        log_tuning_event("DRIVE_ANALYSIS_LLM_PROMPT", {
                            "personality": self.personality,
                            "prompt": full_prompt,
                        })
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
                                    logger.info(f"LLM Drive Analysis Result: {drive_adjustments}") # Log the full result at INFO level
                                    # --- Log parsed LLM adjustments ---
                                    log_tuning_event("DRIVE_ANALYSIS_LLM_PARSED", {
                                        "personality": self.personality,
                                        "raw_response": llm_response_str,
                                        "parsed_adjustments": drive_adjustments,
                                    })

                                    # 5. Adjust short_term drive_state based on LLM score
                                    base_adjustment_factor = drive_cfg.get('llm_score_adjustment_factor', 0.15)

                                    # --- Amplify adjustment factor if high-impact nodes were involved ---
                                    amplification_factor = 1.0 # Default: no amplification
                                    if self.high_impact_nodes_this_interaction: # Check if the dictionary is non-empty
                                        max_magnitude = max(self.high_impact_nodes_this_interaction.values()) if self.high_impact_nodes_this_interaction else 0.0
                                        amp_config_factor = drive_cfg.get('emotional_impact_amplification_factor', 1.5) # How much to amplify by
                                        # Simple amplification: scale based on max magnitude relative to threshold
                                        impact_threshold = drive_cfg.get('emotional_impact_threshold', 1.0)
                                        # Ensure threshold is not zero to avoid division error
                                        if impact_threshold > 0:
                                             # Scale amplification based on how much magnitude exceeds threshold, up to configured max factor
                                             magnitude_ratio = max(0.0, (max_magnitude - impact_threshold) / impact_threshold) # How much over threshold, relative
                                             amplification_factor = 1.0 + (magnitude_ratio * (amp_config_factor - 1.0))
                                             amplification_factor = min(amp_config_factor, amplification_factor) # Cap at max configured factor
                                             logger.info(f"Amplifying drive adjustments due to high emotional impact. MaxMag={max_magnitude:.3f}, AmpFactor={amplification_factor:.3f}")
                                             # Log amplification details
                                             log_tuning_event("DRIVE_ADJUSTMENT_AMPLIFICATION", {
                                                 "personality": self.personality,
                                                 "amplification_factor": amplification_factor,
                                                 "max_magnitude": max_magnitude,
                                                 "impact_threshold": impact_threshold,
                                                 "high_impact_nodes": list(self.high_impact_nodes_this_interaction.keys())
                                             })
                                        else:
                                             # Log that no amplification occurred if dict was empty
                                             logger.debug("No high-impact nodes detected in this interaction, no amplification applied.")


                                    # Use amplified adjustment factor for this update cycle
                                    adjustment_factor = base_adjustment_factor * amplification_factor
                                    logger.debug(f"Using LLM Score Adjustment Factor: {adjustment_factor:.4f} (Base: {base_adjustment_factor:.4f}, Amp: {amplification_factor:.3f})")
                                    # --- End amplification ---

                                    for drive_name, score in drive_adjustments.items():
                                        if drive_name in self.drive_state["short_term"]:
                                            current_activation = self.drive_state["short_term"][drive_name]
                                            # Validate score is a number between -1 and 1
                                            if not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
                                                logger.warning(f"LLM returned invalid score '{score}' for drive '{drive_name}'. Skipping adjustment.")
                                                continue

                                            # Calculate adjustment based on score and factor
                                            # Positive score (satisfied) should decrease drive if above baseline, increase if below? No, simplify:
                                            # Positive score -> move towards baseline (negative adjustment if above, positive if below)
                                            # Negative score -> move away from baseline (positive adjustment if above, negative if below) - This seems complex.
                                            # Satisfied (score=1) -> adj = -factor (reduces drive)
                                            # Frustrated (score=-1) -> adj = +factor (increases drive)
                                            # Neutral (score=0) -> adj = 0
                                            adjustment = score * adjustment_factor
                                            logger.debug(f"  Drive '{drive_name}' LLM Score: {score:.2f}. Calculated Adjustment: {adjustment:.4f}")

                                            if abs(adjustment) > 1e-4:
                                                # Apply adjustment to short-term state
                                                new_level = current_activation + adjustment
                                                # Optional: Clamp short-term activation? (e.g., between -1 and 2?)
                                                # new_level = max(-1.0, min(2.0, new_level))
                                                self.drive_state["short_term"][drive_name] = new_level
                                                changed = True # Mark that state was changed by LLM analysis
                                                logger.info(f"Applied LLM drive adjustment to '{drive_name}' (Score: {score:.2f}): {current_activation:.3f} -> {new_level:.3f} (Adj: {adjustment:.3f})")
                                                # --- Log individual adjustment ---
                                                log_tuning_event("DRIVE_ANALYSIS_LLM_ADJUSTMENT", {
                                                    "personality": self.personality,
                                                    "drive_name": drive_name,
                                                    "llm_score": score, # Changed from llm_status
                                                    "adjustment_value": adjustment,
                                                    "level_before": current_activation,
                                                    "level_after": new_level,
                                                })
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

        # --- Inter-Drive Dynamics Step (NEW) ---
        inter_drive_cfg = drive_cfg.get('inter_drive_interactions', {})
        if inter_drive_cfg:
            logger.debug("Applying inter-drive dynamics...")
            adjustments_applied = {} # Track adjustments per drive for logging
            # Create a snapshot of the state *before* inter-drive adjustments
            state_before_inter_drive = {k: v for k, v in self.drive_state["short_term"].items()}

            for influencing_drive, targets in inter_drive_cfg.items():
                if influencing_drive in state_before_inter_drive:
                    influencing_level = state_before_inter_drive[influencing_drive]
                    for target_drive, interaction_params in targets.items():
                        if target_drive in self.drive_state["short_term"]:
                            threshold = interaction_params.get('threshold', 0.0)
                            factor = interaction_params.get('factor', 0.0)

                            if influencing_level > threshold and abs(factor) > 1e-4:
                                # Calculate adjustment based on how much level exceeds threshold
                                adjustment = factor * (influencing_level - threshold)
                                # Apply adjustment to the *current* state (allowing cascade effects within step?)
                                # Or apply all adjustments based on the state *before* this step? Let's use state_before_inter_drive.
                                current_target_level = self.drive_state["short_term"][target_drive] # Get potentially already adjusted level
                                new_target_level = current_target_level + adjustment
                                # Optional clamping?
                                self.drive_state["short_term"][target_drive] = new_target_level
                                changed = True # Mark change
                                logger.debug(f"  Inter-Drive: '{influencing_drive}' (Lvl:{influencing_level:.2f} > Thr:{threshold:.2f}) -> '{target_drive}' (Adj:{adjustment:.3f}) NewLvl:{new_target_level:.3f}")
                                # Store adjustment details for logging
                                if target_drive not in adjustments_applied: adjustments_applied[target_drive] = []
                                adjustments_applied[target_drive].append({
                                    "influencer": influencing_drive,
                                    "influencer_level": influencing_level,
                                    "threshold": threshold,
                                    "factor": factor,
                                    "adjustment": adjustment,
                                    "level_before_inter": state_before_inter_drive[target_drive],
                                    "level_after_inter": new_target_level
                                })

            if adjustments_applied:
                 log_tuning_event("DRIVE_INTER_INTERACTIONS", {
                     "personality": self.personality,
                     "state_before": state_before_inter_drive,
                     "adjustments_applied": adjustments_applied,
                     "state_after": self.drive_state["short_term"].copy()
                 })
                 logger.info(f"Short-term drive state updated after inter-drive dynamics: {self.drive_state['short_term']}")


        # Saving happens in the calling function (e.g., run_consolidation or _save_memory)

    def _update_long_term_drives(self, high_impact_memory_uuid: str | None = None):
        """
        Updates long-term drive levels based on LLM analysis of the ASM or other
        long-term memory indicators. Can also be nudged by a specific high-impact memory.
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

        # --- Baseline Dynamics: Nudge by High-Impact Memory (NEW) ---
        if high_impact_memory_uuid and high_impact_memory_uuid in self.graph:
            logger.info(f"Applying long-term drive nudge from high-impact memory: {high_impact_memory_uuid[:8]}")
            try:
                node_data = self.graph.nodes[high_impact_memory_uuid]
                valence = node_data.get('emotion_valence', 0.0)
                arousal = node_data.get('emotion_arousal', 0.1)
                shift_factor = drive_cfg.get('high_impact_memory_baseline_shift_factor', 0.1)
                lt_changed = False
                nudge_details = {}

                # Example Nudge Logic:
                # - High positive valence -> Increase Connection, Control?
                # - High negative valence -> Decrease Safety, Control? Increase Understanding?
                # - High arousal -> Increase Novelty? Decrease Safety?
                # This needs refinement based on desired personality effects.

                # Simple Example: Strong positive experience boosts Connection LT state
                if valence > 0.7 and shift_factor > 0: # Threshold for strong positive
                    target_drive = "Connection"
                    if target_drive in self.drive_state["long_term"]:
                        current_lt = self.drive_state["long_term"][target_drive]
                        # Nudge towards max (+1.0)
                        adjustment = (1.0 - current_lt) * shift_factor
                        if abs(adjustment) > 1e-4:
                            new_lt = max(-1.0, min(1.0, current_lt + adjustment))
                            self.drive_state["long_term"][target_drive] = new_lt
                            lt_changed = True
                            nudge_details[target_drive] = {"before": current_lt, "after": new_lt, "adjustment": adjustment, "reason": "high_valence"}
                            logger.info(f"  Nudged LT '{target_drive}' due to high valence: {current_lt:.3f} -> {new_lt:.3f}")

                # Simple Example: Strong negative experience hurts Safety LT state
                if valence < -0.7 and shift_factor > 0: # Threshold for strong negative
                    target_drive = "Safety"
                    if target_drive in self.drive_state["long_term"]:
                        current_lt = self.drive_state["long_term"][target_drive]
                        # Nudge towards min (-1.0)
                        adjustment = (-1.0 - current_lt) * shift_factor
                        if abs(adjustment) > 1e-4:
                            new_lt = max(-1.0, min(1.0, current_lt + adjustment))
                            self.drive_state["long_term"][target_drive] = new_lt
                            lt_changed = True
                            nudge_details[target_drive] = {"before": current_lt, "after": new_lt, "adjustment": adjustment, "reason": "low_valence"}
                            logger.info(f"  Nudged LT '{target_drive}' due to low valence: {current_lt:.3f} -> {new_lt:.3f}")

                if lt_changed:
                    log_tuning_event("DRIVE_LT_NUDGE_HIGH_IMPACT", {
                        "personality": self.personality,
                        "memory_uuid": high_impact_memory_uuid,
                        "memory_valence": valence,
                        "memory_arousal": arousal,
                        "shift_factor": shift_factor,
                        "nudge_details": nudge_details,
                        "lt_state_after": self.drive_state["long_term"].copy()
                    })
                    self._save_drive_state() # Save if nudged

            except Exception as e:
                logger.error(f"Error applying high-impact memory nudge to long-term drives: {e}", exc_info=True)


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

    def _summarize_file_content(self, file_content: str) -> str | None:
        """Uses LLM to generate a concise summary of file content."""
        if not file_content:
            return None

        prompt_template = self._load_prompt("workspace_file_summary_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load workspace file summary prompt template.")
            return None

        # Ensure content isn't excessively long for the summarizer prompt itself
        max_summary_input_len = 3000 # Limit input to summarizer prompt
        if len(file_content) > max_summary_input_len:
            file_content = file_content[:max_summary_input_len] + "\n... [Content Truncated for Summarizer]"

        summary_prompt = prompt_template.format(file_content=file_content)
        logger.debug(f"Sending file summary prompt (Content length: {len(file_content)})...")

        summary = self._call_configured_llm('workspace_file_summary', prompt=summary_prompt)

        if summary and not summary.startswith("Error:"):
            # Basic cleaning: remove potential leading/trailing quotes or list markers
            summary = summary.strip().strip('"-* ')
            logger.debug(f"Generated file summary: '{summary}'")
            return summary
        else:
            logger.error(f"File summary generation failed: {summary}")
            return None

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


    def _handle_input(self, interaction_id: str, user_input: str, conversation_history: list, attachment_data: dict | None) -> tuple[str | None, str, list]:
        """
        Handles input processing, choosing between text or multimodal, and calls the LLM.

        Returns:
                Tuple: (inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion)
                        user_emotion and ai_emotion are (valence, arousal) tuples or None.
        """
        inner_thoughts = None
        raw_llm_response = "Error: LLM call failed."
        user_emotion = None # Initialize emotion tuples
        ai_emotion = None
        memories_retrieved = []

        if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('data_url'):
            logger.info(f"Interaction {interaction_id[:8]}: Handling multimodal input.")
            # Call multimodal handler (which calls the appropriate LLM)
            # Assuming _handle_multimodal_input internally calls _call_configured_llm
            # and returns thoughts, response_text
            inner_thoughts, raw_llm_response = self._handle_multimodal_input(user_input, attachment_data)
            # No memory retrieval for multimodal yet
            memories_retrieved = []
        else:
            logger.info(f"Interaction {interaction_id[:8]}: Handling text input.")
            # Call text handler (which includes retrieval and LLM call)
            # Assuming _handle_text_input internally calls retrieve, construct prompt, call LLM
            # and returns thoughts, response_text, memories_used, user_emotion, ai_emotion
            inner_thoughts, raw_llm_response, memories_retrieved, user_emotion, ai_emotion = self._handle_text_input(user_input, conversation_history)

        return inner_thoughts, raw_llm_response, memories_retrieved, user_emotion, ai_emotion


    def process_interaction(self, user_input: str, conversation_history: list, attachment_data: dict | None = None) -> InteractionResult:
        """
        Processes user input, calls LLM, updates memory, checks for actions.

        Args:
            user_input: The text input from the user.
            conversation_history: The recent conversation history.
            attachment_data: Optional dictionary containing attachment info (type, filename, data_url/path).

        Returns:
            InteractionResult: An object containing the final response, thoughts, memories, node UUIDs, and planning flag.
        """
        interaction_id = str(uuid.uuid4())
        logger.info(f"--- Processing Interaction START (ID: {interaction_id[:8]}) ---")
        logger.info(f"Input='{strip_emojis(user_input[:60])}...', Attachment: {attachment_data.get('type') if attachment_data else 'No'}")
        self.high_impact_nodes_this_interaction.clear() # Reset interaction-specific state

        # Default error result
        error_result = InteractionResult(final_response_text="Error: Processing failed unexpectedly.")

        log_tuning_event("INTERACTION_START", {
            "interaction_id": interaction_id,
            "personality": self.personality,
            "user_input_preview": strip_emojis(user_input[:100]),
            "has_attachment": bool(attachment_data),
            "attachment_type": attachment_data.get('type') if attachment_data else None,
            "history_length": len(conversation_history)
        })

        # --- Initial Check ---
        if not hasattr(self, 'embedder') or self.embedder is None:
            logger.critical("PROCESS_INTERACTION CRITICAL ERROR: Embedder not initialized!")
            log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "embedder_check", "error": "Embedder not initialized" })
            return error_result

        try:
            # --- Step 1: Handle Input & Call LLM ---
            # Determine text to save in the graph for the user turn
            graph_user_input = user_input
            if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('filename'):
                placeholder = f" [Image Attached: {attachment_data['filename']}]"
                separator = " " if graph_user_input else ""
                graph_user_input += separator + placeholder

            # --- Step 1: Handle Input & Call LLM (returns emotions for text input) ---
            # Note: _handle_multimodal_input currently doesn't return emotions
            (inner_thoughts, raw_llm_response, memories_retrieved,
             user_emotion, ai_emotion) = self._handle_input(
                interaction_id, user_input, conversation_history, attachment_data)

            # --- Step 2: Parse LLM Response for Thoughts ---
            # (Note: _handle_input might already do this, adjust if needed)
            # If _handle_input returns raw response, parse here:
            # inner_thoughts, final_response_text = self._parse_llm_response(raw_llm_response)
            # If _handle_input returns parsed response, use it directly:
            final_response_text = raw_llm_response # Assuming _handle_input returns parsed response

            # Check for critical LLM call failure
            if final_response_text is None or "Error:" in final_response_text[:20]: # Check beginning for errors
                 logger.error(f"Interaction {interaction_id[:8]}: LLM call failed or returned error: '{final_response_text}'")
                 log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "llm_call", "error": final_response_text or "Empty LLM response" })
                 # Use the error message from LLM if available, otherwise default
                 error_result.final_response_text = final_response_text or "Error: LLM processing failed."
                 return error_result

            # --- Step 3: Check for Action Request ---
            response_before_action_check = final_response_text
            final_response_text_cleaned, needs_planning = self._check_for_action_request(
                response_text=response_before_action_check,
                user_input=user_input # Pass original user input for keyword check
            )
            logger.debug(f"Interaction {interaction_id[:8]}: Needs Planning Flag = {needs_planning}. Cleaned Response: '{final_response_text_cleaned[:60]}...'")

            # --- Step 4: Update Graph & Context (Pass emotions) ---
            # Pass the text intended for the graph nodes
            user_node_uuid, ai_node_uuid = self._update_graph_and_context(
                graph_user_input=graph_user_input, # Text for user node
                user_node_uuid=None, # Let the function handle adding
                parsed_response=final_response_text_cleaned, # Cleaned text for AI node
                ai_node_uuid=None, # Let the function handle adding
                conversation_history=conversation_history,
                user_input_for_analysis=user_input, # Original input for analysis triggers
                user_emotion=user_emotion, # Pass calculated user emotion
                ai_emotion=ai_emotion # Pass calculated AI emotion
            )
            logger.debug(f"Interaction {interaction_id[:8]}: Graph updated. User Node: {user_node_uuid}, AI Node: {ai_node_uuid}")

            # --- Step 5: Assemble and Return Result ---
            final_result = InteractionResult(
                final_response_text=final_response_text_cleaned,
                inner_thoughts=inner_thoughts,
                memories_used=memories_retrieved,
                user_node_uuid=user_node_uuid,
                ai_node_uuid=ai_node_uuid,
                needs_planning=needs_planning
            )

            log_tuning_event("INTERACTION_END", {
                "interaction_id": interaction_id,
                "personality": self.personality,
                "final_response_preview": strip_emojis(final_result.final_response_text[:100]),
                "retrieved_memory_count": len(final_result.memories_used),
                "user_node_added": final_result.user_node_uuid[:8] if final_result.user_node_uuid else None,
                "ai_node_added": final_result.ai_node_uuid[:8] if final_result.ai_node_uuid else None,
                "needs_planning": final_result.needs_planning
            })
            logger.info(f"--- Processing Interaction END (ID: {interaction_id[:8]}) ---")
            return final_result

        except Exception as e:
            logger.error(f"--- CRITICAL Outer Error during process_interaction (ID: {interaction_id[:8]}) ---", exc_info=True)
            error_message_for_user = f"Error: Processing failed unexpectedly in main loop. Details: {type(e).__name__}"
            log_tuning_event("INTERACTION_ERROR", {
                "interaction_id": interaction_id,
                "personality": self.personality,
                "stage": "outer_exception_handler",
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return default error result object
            error_result.final_response_text = error_message_for_user
            return error_result

    def _handle_text_input(self, user_input: str, conversation_history: list) -> tuple[str | None, str, list, tuple | None, tuple | None]:
        """
        Handles text-based input, including emotional analysis, memory retrieval, and LLM call.
        Returns: (inner_thoughts, final_response, memories_retrieved, user_emotion, ai_emotion)
        """
        logger.info("Handling text input...")
        user_emotion_result = None # (valence, arousal)
        ai_emotion_result = None # (valence, arousal)
        effective_mood = self.last_interaction_mood # Start with mood from last interaction
        emotional_instructions = "" # Initialize

        # --- 1. Emotional Analysis (if enabled) ---
        if self.emotional_core and self.emotional_core.is_enabled:
            try:
                # Prepare context for analysis
                history_context_str = "\n".join([f"{turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}" for turn in conversation_history[-5:]]) # Last 5 turns
                kg_context_str = self._get_kg_context_for_emotion(user_input)
                logger.debug(f"Emotional Analysis Context: History='{history_context_str[:100]}...', KG='{kg_context_str[:100]}...'")

                # Run analysis on user input
                self.emotional_core.analyze_input(
                    user_input=user_input,
                    history_context=history_context_str,
                    kg_context=kg_context_str
                )

                # Aggregate results (tendency/mood hints stored in self.emotional_core)
                # This updates self.emotional_core.derived_mood_hints
                self.emotional_core.aggregate_and_combine()
                emotional_instructions = self.emotional_core.craft_prompt_instructions()

                # --- Extract User Emotion (VADER from EmotionalCore's analysis of user_input) ---
                user_sentiment = self.emotional_core.current_analysis_results.get("sentiment", {})
                user_valence = user_sentiment.get("compound", 0.0)
                # Simple arousal from pos/neg VADER scores (can be refined)
                user_arousal = (user_sentiment.get("pos", 0.0) + user_sentiment.get("neg", 0.0)) * 0.5
                user_emotion_result = (user_valence, user_arousal)
                logger.info(f"Derived user input emotion (VADER via EmoCore): V={user_valence:.2f}, A={user_arousal:.2f}")

            except Exception as emo_e:
                logger.error(f"Error during emotional analysis: {emo_e}", exc_info=True)
                emotional_instructions = "" # Ensure it's empty on error
                user_emotion_result = None # Ensure None on error


        # --- 2. Memory Retrieval ---
        query_type = self._classify_query_type(user_input)
        max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7)
        initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes, query_type=query_type)
        initial_uuids = [uid for uid, score in initial_nodes]
        memories_retrieved = []
        # effective_mood is already initialized with self.last_interaction_mood
        # retrieve_memory_chain will update it based on drives and emotional_core hints

        # --- Retrieve memory chain (this applies drive influence and EmotionalCore hints internally) ---
        # Call retrieve_memory_chain regardless of initial_uuids to ensure effective_mood is updated
        # by drive state and EmotionalCore hints.
        memories_retrieved, effective_mood = self.retrieve_memory_chain(
            initial_node_uuids=initial_uuids, # Can be empty
            recent_concept_uuids=list(self.last_interaction_concept_uuids),
            current_mood=effective_mood # Pass the mood (potentially influenced by drives/emocore)
        )
        if not initial_uuids:
            logger.info("No initial nodes found for retrieval, but mood updated by retrieve_memory_chain.")


        # --- 3. Construct Prompt ---
        # effective_mood is now updated by retrieve_memory_chain to include drive/emocore influence
        # emotional_instructions were retrieved from EmotionalCore earlier
        prompt = self._construct_prompt(
            user_input=user_input,
            conversation_history=conversation_history,
            memory_chain=memories_retrieved,
            tokenizer=self.tokenizer,
            max_context_tokens=self.config.get('prompting', {}).get('max_context_tokens', 4096),
            current_mood=effective_mood, # Pass mood after drive/emocore influence
            emotional_instructions=emotional_instructions
        )

        # --- 4. Call LLM ---
        raw_llm_response = self._call_configured_llm('main_chat_text', prompt=prompt)

        # --- 5. Parse LLM Response ---
        inner_thoughts, final_response = self._parse_llm_response(raw_llm_response)

        # --- 6. Analyze AI Response Emotion (if VADER available in EmotionalCore) ---
        if self.emotional_core and self.emotional_core.sentiment_analyzer and final_response:
            try:
                # Use EmotionalCore's VADER method directly
                ai_scores = self.emotional_core._analyze_sentiment_vader(final_response)
                ai_valence = ai_scores.get("compound", 0.0)
                # Simple arousal from pos/neg VADER scores (can be refined)
                ai_arousal = (ai_scores.get("pos", 0.0) + ai_scores.get("neg", 0.0)) * 0.5
                ai_emotion_result = (ai_valence, ai_arousal)
                logger.info(f"Derived AI response emotion (VADER via EmoCore): V={ai_valence:.2f}, A={ai_arousal:.2f}")
            except Exception as ai_emo_e:
                logger.error(f"Error analyzing AI response emotion: {ai_emo_e}", exc_info=True)
                ai_emotion_result = None # Ensure None on error

        # --- 7. Return Results ---
        # Return thoughts, final response, memories, and emotions
        return inner_thoughts, final_response, memories_retrieved, user_emotion_result, ai_emotion_result

    def _handle_multimodal_input(self, user_input: str, attachment_data: dict) -> tuple[str | None, str]:
        """Handles multimodal input processing and LLM call via Chat Completions API."""
        # >>> Placeholder - Ensure the actual implementation exists and returns:
        # >>> (inner_thoughts: str|None, raw_llm_response: str)
        # NOTE: This currently doesn't calculate/return user/ai emotions.
        #       Multimodal emotional analysis would require a different approach.
        logger.warning("_handle_multimodal_input called - Ensure full implementation exists. Emotion analysis skipped.")
        # Example placeholder logic (replace with actual logic):
        messages = []
        user_content = []
        if user_input: user_content.append({"type": "text", "text": user_input})
        user_content.append({"type": "image_url", "image_url": {"url": attachment_data['data_url']}})
        messages.append({"role": "user", "content": user_content})

        raw_llm_response = self._call_configured_llm('main_chat_multimodal', messages=messages)
        # --- Parse thoughts here ---
        inner_thoughts, final_response = self._parse_llm_response(raw_llm_response)
        return inner_thoughts, final_response # Return thoughts and final response

    def _update_graph_and_context(self, graph_user_input: str, user_node_uuid: str | None, parsed_response: str, ai_node_uuid: str | None, conversation_history: list, user_input_for_analysis: str, user_emotion: tuple | None, ai_emotion: tuple | None) -> tuple[str | None, str | None]:
        """
        Adds user/AI nodes with emotions, updates context, runs heuristics, checks for high impact.
        Returns: (final_user_node_uuid: str|None, final_ai_node_uuid: str|None)
        """
        logger.warning("_update_graph_and_context called - Ensure full implementation exists.") # Keep warning for now
        final_user_node_uuid = user_node_uuid
        final_ai_node_uuid = ai_node_uuid

        # --- 1. Add User Node (with emotion) ---
        if final_user_node_uuid is None: # Only add if not already provided
            user_v, user_a = user_emotion if user_emotion else (None, None)
            # Use graph_user_input which includes potential attachment placeholder
            final_user_node_uuid = self.add_memory_node(
                graph_user_input, "User",
                emotion_valence=user_v, emotion_arousal=user_a
            )
            if final_user_node_uuid:
                logger.debug(f"Added user node {final_user_node_uuid[:8]} with emotion V={user_v}, A={user_a}")
                # Check for high impact
                self._check_and_log_high_impact(final_user_node_uuid)

        # --- 2. Store Intention (if any) ---
        intention_result = self._analyze_intention_request(user_input_for_analysis) # Use original input
        if intention_result.get("action") == "store_intention":
            intention_content = f"Remember: {intention_result['content']} (Trigger: {intention_result['trigger']})"
            # Add as a separate 'intention' node linked to the user turn?
            intention_ts = self.graph.nodes[user_node_uuid]['timestamp'] if user_node_uuid and user_node_uuid in self.graph else datetime.now(timezone.utc).isoformat()
            intention_node_uuid = self.add_memory_node(intention_content, "System", 'intention', timestamp=intention_ts)
            if intention_node_uuid and user_node_uuid:
                try:
                    # Link user turn -> intention node
                    self.graph.add_edge(user_node_uuid, intention_node_uuid, type='GENERATED_INTENTION', base_strength=0.9, last_traversed_ts=time.time())
                    logger.info(f"Linked user turn {user_node_uuid[:8]} to intention {intention_node_uuid[:8]}")
                except Exception as link_e:
                    logger.error(f"Failed to link user turn to intention node: {link_e}")

        # --- 3. Add AI Node ---
        if ai_node_uuid is None and parsed_response: # Only add if not already provided and response exists
            ai_v, ai_a = ai_emotion if ai_emotion else (None, None) # Get AI emotion
            final_ai_node_uuid = self.add_memory_node( # Assign to final_ai_node_uuid
                parsed_response, "AI",
                emotion_valence=ai_v, emotion_arousal=ai_a
            )
            if final_ai_node_uuid: # Use the correct variable
                logger.debug(f"Added AI node {final_ai_node_uuid[:8]} with emotion V={ai_v}, A={ai_a}")
                # Check for high impact
                self._check_and_log_high_impact(final_ai_node_uuid)


        # --- 4. Update Context for Next Interaction ---
        self._update_next_interaction_context(final_user_node_uuid, final_ai_node_uuid) # Use final UUIDs

        # --- 5. Apply Heuristics (Example: Repeated Corrections) ---
        # This is a simplified example - requires tracking correction patterns
        correction_keywords = ["actually,", "no,", "you're wrong", "correction:"]
        if any(keyword in user_input_for_analysis.lower() for keyword in correction_keywords):
            self._apply_heuristic_drive_adjustment("Understanding", -0.05, "user_correction", final_user_node_uuid) # Small decrease

        # --- 6. Trigger Drive Update on High Impact ---
        # Check the interaction-specific dictionary populated by _check_and_log_high_impact
        if self.config.get('subconscious_drives', {}).get('trigger_drive_update_on_high_impact', False) and self.high_impact_nodes_this_interaction:
            logger.info("High emotional impact detected in current interaction, triggering immediate drive state update analysis.")
            # Combine context from user/ai turns for analysis
            combined_context = f"User: {graph_user_input}\nAI: {parsed_response}"
            # Pass context to _update_drive_state, which handles LLM call and amplification
            self._update_drive_state(context_text=combined_context)
            # Clearing of high_impact_nodes_this_interaction happens at the start of process_interaction

        # --- 7. Update EmotionalCore Memory (if enabled) ---
        if self.emotional_core and self.emotional_core.is_enabled:
            self.emotional_core.update_memory_with_emotional_insight()


        # --- 8. Save Memory ---
        self._save_memory()

        return final_user_node_uuid, final_ai_node_uuid # Return the final UUIDs

    def _check_and_log_high_impact(self, node_uuid: str):
        """Checks if a node's emotion magnitude exceeds the threshold and logs it for drive updates."""
        if not node_uuid or node_uuid not in self.graph: return

        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False) or not drive_cfg.get('trigger_drive_update_on_high_impact', False):
            return # Skip if drives or trigger disabled

        impact_threshold = drive_cfg.get('emotional_impact_threshold', 1.0)
        if impact_threshold <= 0: return # Skip if threshold invalid

        try:
            node_data = self.graph.nodes[node_uuid]
            valence = node_data.get('emotion_valence', 0.0)
            arousal = node_data.get('emotion_arousal', 0.1)
            # Calculate magnitude (ensure non-negative components if needed, though V can be negative)
            magnitude = math.sqrt(valence**2 + arousal**2)

            if magnitude >= impact_threshold:
                logger.info(f"High emotional impact node detected: {node_uuid[:8]} (Magnitude: {magnitude:.3f} >= {impact_threshold:.3f})")
                self.high_impact_nodes_this_interaction[node_uuid] = magnitude # Store magnitude for amplification
                log_tuning_event("HIGH_IMPACT_NODE_DETECTED", {
                    "personality": self.personality,
                    "node_uuid": node_uuid,
                    "node_type": node_data.get('node_type'),
                    "valence": valence,
                    "arousal": arousal,
                    "magnitude": magnitude,
                    "threshold": impact_threshold,
                })
        except Exception as e:
            logger.error(f"Error checking high impact for node {node_uuid[:8]}: {e}", exc_info=True)

    def _parse_llm_response(self, raw_response_text: str) -> tuple[str | None, str]:
        """
        Parses the raw LLM response to extract inner thoughts and the final response text.

        Args:
            raw_response_text: The raw string output from the LLM.

        Returns:
            A tuple: (inner_thoughts: str | None, final_response_text: str)
        """
        inner_thoughts = None
        final_response_text = raw_response_text # Default to raw response

        if not raw_response_text:
            logger.warning("LLM response was empty.")
            return None, "" # Return empty string for final response if raw was empty/None

        # Ensure input is a string
        if not isinstance(raw_response_text, str):
            logger.error(f"LLM response ('raw_response_text') is not a string! Type: {type(raw_response_text)}. Converting.")
            raw_response_text = str(raw_response_text)

        logger.debug(f"Parsing LLM response: '{raw_response_text[:200]}...'")
        try:
            # Correct regex to find thoughts and capture content, ensuring it's non-greedy
            # Looks for <thought>...</thought> and captures content inside and text after.
            thought_match = re.search(r"<thought>(.*?)</thought>(.*)", raw_response_text, re.DOTALL | re.IGNORECASE) # Added IGNORECASE
            if thought_match:
                inner_thoughts = thought_match.group(1).strip()
                final_response_text = thought_match.group(2).strip()
                logger.debug(f"Extracted thoughts: '{inner_thoughts[:100]}...'")
                logger.debug(f"Remaining response: '{final_response_text[:100]}...'")
            else:
                # No thoughts found, use the full response as the final response
                logger.debug("No <thought> tags found in LLM response.")
                final_response_text = raw_response_text.strip() # Ensure it's stripped

        except TypeError as te:
            # This error indicates re.search failed likely due to input type, though we try to prevent this.
            logger.error(f"TypeError during thought parsing (re.search likely failed): {te}. Raw text type: {type(raw_response_text)}", exc_info=True)
            logger.error(f"Problematic raw_response_text for parsing: ```{raw_response_text}```")
            inner_thoughts = None
            final_response_text = raw_response_text # Fallback to original text
        except Exception as parse_err:
            logger.error(f"Unexpected error parsing thoughts from LLM response: {parse_err}", exc_info=True)
            inner_thoughts = None
            final_response_text = raw_response_text # Fallback to original text

        # --- Ensure final_response_text is never None ---
        if final_response_text is None:
            final_response_text = ""

        return inner_thoughts, final_response_text

    def _check_for_action_request(self, response_text: str, user_input: str) -> tuple[str, bool]:
        """
        Checks the AI's response text for an [ACTION:] tag or the user input for keywords
        to determine if workspace planning is needed.

        Args:
            response_text: The AI's final response text (after thought stripping).
            user_input: The original user input text.

        Returns:
            A tuple: (cleaned_response_text: str, needs_planning: bool)
        """
        needs_planning = False
        cleaned_response_text = response_text # Start with the input response text

        # Ensure text is a string
        if not isinstance(cleaned_response_text, str):
            logger.warning(f"_check_for_action_request received non-string: {type(cleaned_response_text)}. Coercing.")
            cleaned_response_text = str(cleaned_response_text) if cleaned_response_text is not None else ""

        # 1. Check for [ACTION:] tag at the end of the AI response
        try:
            # Regex looks for [ACTION: {maybe whitespace} {JSON content} {maybe whitespace}] at the end ($)
            action_match = re.search(r'\[ACTION:\s*(\{.*?\})\s*\]$', cleaned_response_text, re.DOTALL | re.IGNORECASE) # Added IGNORECASE
            if action_match:
                action_json_str = action_match.group(1)
                logger.info(f"AI requested action detected via tag: {action_json_str}")
                # Remove action tag from the text
                cleaned_response_text = cleaned_response_text[:action_match.start()].strip()
                # Validate JSON structure slightly to confirm intent
                try:
                    action_data = json.loads(action_json_str)
                    if isinstance(action_data, dict) and "action" in action_data:
                        needs_planning = True
                        logger.info("Setting needs_planning=True due to valid [ACTION:] tag.")
                    else:
                        logger.warning(f"ACTION tag found but content is not a valid JSON object with 'action' key: {action_json_str}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON in ACTION tag: {action_json_str}")
            else:
                 logger.debug("No AI action request tag found in response.")
        except Exception as search_err:
            logger.error(f"Unexpected error during ACTION tag search: {search_err}", exc_info=True)
            # Proceed without planning if tag search fails

        # 2. Check user input keywords (only if tag didn't already set planning)
        if not needs_planning:
            user_input_lower = user_input.lower() if isinstance(user_input, str) else ""
            # Use the WORKSPACE_KEYWORDS list defined globally
            if any(keyword in user_input_lower for keyword in WORKSPACE_KEYWORDS):
                needs_planning = True
                logger.info(f"Potential workspace action detected via keywords in user input. Setting needs_planning=True.")

        return cleaned_response_text, needs_planning

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
        norm_recency_raw = 1.0 - math.exp(-decay_constant * recency_sec) # Score approaches 1 as time increases
        # --- Cap Recency Contribution ---
        # Limit the maximum impact of recency, even after very long breaks.
        # Example: Cap normalized recency at 0.9 to prevent it from reaching 1.0.
        max_norm_recency = weights.get('max_norm_recency_cap', 0.95) # Add this to config if needed, default 0.95
        norm_recency = min(norm_recency_raw, max_norm_recency)
        if norm_recency_raw > max_norm_recency:
            logger.debug(f"    Recency capped for node {node_uuid[:8]}: Raw={norm_recency_raw:.3f} -> Capped={norm_recency:.3f}")

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

        # --- Log Raw and Normalized Factors ---
        log_tuning_event("FORGETTABILITY_FACTORS", {
            "personality": self.personality,
            "node_uuid": node_uuid,
            "node_type": node_type,
            "raw_factors": {
                "recency_sec": recency_sec,
                "activation": activation,
                "saliency": saliency,
                "emotion_magnitude": emotion_magnitude,
                "degree": degree,
                "access_count": access_count,
            },
            "normalized_factors": {
                "norm_recency": norm_recency,
                "norm_inv_activation": norm_inv_activation,
                "norm_type_forgettability": norm_type_forgettability,
                "norm_inv_saliency": norm_inv_saliency,
                "norm_inv_emotion": norm_inv_emotion,
                "norm_inv_connectivity": norm_inv_connectivity,
                "norm_inv_access_count": norm_inv_access_count,
            },
            "weights": weights, # Log the weights used for this calculation
        })

        # Clamp intermediate score 0-1 before applying resistance factors
        intermediate_score = max(0.0, min(1.0, score))
        logger.debug(f"    Forget Score Factors for {node_uuid[:8]}: Rec({norm_recency:.2f}), Act({norm_inv_activation:.2f}), Typ({norm_type_forgettability:.2f}), Sal({norm_inv_saliency:.2f}), Emo({norm_inv_emotion:.2f}), Con({norm_inv_connectivity:.2f}), Acc({norm_inv_access_count:.2f}) -> Intermediate Score: {intermediate_score:.3f}")

        # --- Apply Decay Resistance (Type-Based) ---
        type_resistance_factor = node_data.get('decay_resistance_factor', 1.0)
        score_after_type_resistance = intermediate_score * type_resistance_factor
        logger.debug(f"    Node {node_uuid[:8]} Type Resistance Factor: {type_resistance_factor:.3f}. Score after type resist: {score_after_type_resistance:.4f}")

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
            logger.debug(f"    Node {node_uuid[:8]} Emotion Mag: {emotion_magnitude:.3f} (Norm: {clamped_emo_mag:.3f}), Emo Resist Factor: {emotion_resistance_multiplier:.3f}. Score updated to: {final_adjusted_score:.4f}")

        # --- Apply Core Memory Resistance/Immunity ---
        is_core = node_data.get('is_core_memory', False)
        core_mem_enabled = self.config.get('features', {}).get('enable_core_memory', False)
        core_mem_cfg = self.config.get('core_memory', {})
        core_immunity_enabled = core_mem_cfg.get('forget_immunity', True) # Default immunity to True now

        if core_mem_enabled and is_core:
            if core_immunity_enabled:
                logger.debug(f"    Node {node_uuid[:8]} is Core Memory and immunity is enabled. Setting forgettability to 0.0.")
                final_adjusted_score = 0.0 # Immune to forgetting
            else:
                # Apply a strong resistance factor if immunity is off but it's still core
                core_resistance_factor = weights.get('core_memory_resistance_factor', 0.05) # Use updated default
                final_adjusted_score *= core_resistance_factor
                logger.debug(f"    Node {node_uuid[:8]} is Core Memory (Immunity OFF). Applying resistance factor {core_resistance_factor:.2f}. Score updated to: {final_adjusted_score:.4f}")

        # Final clamp and log before returning
        final_adjusted_score = max(0.0, min(1.0, final_adjusted_score))
        logger.debug(f"    Final calculated forgettability score for {node_uuid[:8]}: {final_adjusted_score:.4f}")

        # --- Log Final Score ---
        log_tuning_event("FORGETTABILITY_FINAL_SCORE", {
            "personality": self.personality,
            "node_uuid": node_uuid,
            "node_type": node_type,
            "intermediate_score": intermediate_score,
            "score_after_type_resistance": score_after_type_resistance,
            "score_after_emotion_resistance": final_adjusted_score if 'emotion_magnitude_resistance_factor' in weights else score_after_type_resistance, # Log score before core check
            "is_core_memory": is_core,
            "core_immunity_enabled": core_immunity_enabled,
            "final_forgettability_score": final_adjusted_score,
            "current_memory_strength": node_data.get('memory_strength', 1.0), # Log current strength for context
        })

        return final_adjusted_score

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

    # --- DEPRECATED ---
    # def execute_action(self, action_data: dict) -> tuple[bool, str, str]:
    #     """ DEPRECATED: Logic moved to WorkspaceAgent. """
    #     logger.warning("execute_action is deprecated. Use WorkspaceAgent.")
    #     action = action_data.get("action", "unknown")
    #     return False, f"Action '{action}' execution is deprecated.", f"{action}_deprecated"


    def _consolidate_summarize(self, context_text: str, nodes_data: list, processed_node_uuids: list) -> tuple[str | None, bool]: # Renamed param
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
                for orig_uuid in processed_node_uuids: # Use the correct parameter name
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

                    # --- Attempt to add unknown concepts ---
                    if not subj_uuid:
                        logger.info(f"Rich Relation: Subject concept '{subj_text}' not found. Attempting add.")
                        subj_uuid = self._add_or_find_concept_node(subj_text, concept_node_map)
                        if subj_uuid: logger.info(f"Added/Found subject concept: {subj_uuid[:8]}")
                        else: logger.warning(f"Failed to add/find subject concept '{subj_text}'. Skipping relation."); continue
                    if not obj_uuid:
                        logger.info(f"Rich Relation: Object concept '{obj_text}' not found. Attempting add.")
                        obj_uuid = self._add_or_find_concept_node(obj_text, concept_node_map)
                        if obj_uuid: logger.info(f"Added/Found object concept: {obj_uuid[:8]}")
                        else: logger.warning(f"Failed to add/find object concept '{obj_text}'. Skipping relation."); continue
                    # --- End attempt to add unknown concepts ---

                    # Check again if nodes exist in graph after potential add
                    if subj_uuid and obj_uuid and subj_uuid in self.graph and obj_uuid in self.graph:
                        try:
                            # Check if edge exists OR if the reverse edge exists with the same type
                            edge_exists = self.graph.has_edge(subj_uuid, obj_uuid) and self.graph.edges[subj_uuid, obj_uuid].get("type") == rel_type
                            reverse_edge_exists = self.graph.has_edge(obj_uuid, subj_uuid) and self.graph.edges[obj_uuid, subj_uuid].get("type") == rel_type

                            if not edge_exists and not reverse_edge_exists: # Add only if neither direction exists with this type
                                base_strength = 0.65 # Slightly lower strength for LLM relations?
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

    def _add_or_find_concept_node(self, concept_text: str, concept_node_map: dict) -> str | None:
        """
        Checks if a concept exists (or is very similar). If not, adds it.
        Updates concept_node_map and returns the UUID.
        """
        if not concept_text: return None

        # Check if already in the map for this consolidation run
        existing_uuid = concept_node_map.get(concept_text)
        if existing_uuid: return existing_uuid

        # Search for existing similar nodes in the graph
        concept_sim_threshold = self.config.get('consolidation', {}).get('concept_similarity_threshold', 0.3)
        similar_concepts = self._search_similar_nodes(concept_text, k=1, node_type_filter='concept')

        if similar_concepts and similar_concepts[0][1] <= concept_sim_threshold:
            found_uuid = similar_concepts[0][0]
            logger.info(f"Found existing similar node {found_uuid[:8]} for '{concept_text}'. Using existing.")
            concept_node_map[concept_text] = found_uuid # Add to map for this run
            # Update access time?
            if found_uuid in self.graph: self.graph.nodes[found_uuid]['last_accessed_ts'] = time.time()
            return found_uuid
        else:
            # Add as a new concept node
            # Use slightly lower base strength for concepts added implicitly during relation extraction?
            new_concept_uuid = self.add_memory_node(concept_text, "System", 'concept', base_strength=0.75)
            if new_concept_uuid:
                logger.info(f"Added new concept node {new_concept_uuid[:8]} for '{concept_text}' during relation extraction.")
                concept_node_map[concept_text] = new_concept_uuid # Add to map for this run
                return new_concept_uuid
            else:
                logger.error(f"Failed to add new concept node '{concept_text}' during relation extraction.")
                return None

    def _load_prompt(self, filename: str) -> str:
        """Loads a prompt template from the prompts directory."""
        # Assumes a 'prompts' subdirectory relative to this script's location
        # Or adjust path logic as needed (e.g., relative to config?)
        # For simplicity, let's assume prompts are relative to the main script dir
        # If base_memory_path is reliable, maybe put prompts there?
        # Let's assume relative to the script for now.
        script_dir = os.path.dirname(os.path.abspath(__file__)) # Use absolute path for script dir
        prompt_path = os.path.join(script_dir, "prompts", filename)
        logger.debug(f"Attempting to load prompt from: {prompt_path}") # Add debug log
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
        if llm_response_str and not llm_response_str.startswith("Error:"): # Check for API errors first
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
                        # --- Process chains, adding new concepts if found ---
                        processed_chains = []
                        for chain in parsed_list:
                            if not isinstance(chain, list) or len(chain) < 2 or not all(isinstance(item, str) for item in chain):
                                logger.warning(f"Skipping invalid chain format: {chain}")
                                continue

                            chain_valid = True
                            concepts_in_chain = [] # Store (text, uuid) tuples for this chain
                            for concept_text in chain:
                                concept_uuid = concept_node_map.get(concept_text)
                                if not concept_uuid:
                                    # Concept not found - try adding it
                                    logger.info(f"Concept '{concept_text}' from causal chain not found. Attempting to add.")
                                    # Search if a very similar concept already exists to avoid near duplicates
                                    concept_sim_threshold = self.config.get('consolidation', {}).get('concept_similarity_threshold', 0.3)
                                    similar_concepts = self._search_similar_nodes(concept_text, k=1, node_type_filter='concept')
                                    if similar_concepts and similar_concepts[0][1] <= concept_sim_threshold:
                                         existing_uuid = similar_concepts[0][0]
                                         logger.info(f"Found existing similar node {existing_uuid[:8]} for '{concept_text}'. Using existing.")
                                         concept_uuid = existing_uuid
                                         concept_node_map[concept_text] = concept_uuid # Add to map for this consolidation run
                                    else:
                                         # Add as a new concept node
                                         new_concept_uuid = self.add_memory_node(concept_text, "System", 'concept', base_strength=0.8) # Slightly lower strength?
                                         if new_concept_uuid:
                                             logger.info(f"Added new concept node {new_concept_uuid[:8]} for '{concept_text}' from causal chain.")
                                             concept_uuid = new_concept_uuid
                                             concept_node_map[concept_text] = concept_uuid # Add to map for this consolidation run
                                         else:
                                             logger.error(f"Failed to add new concept node '{concept_text}' from causal chain. Skipping chain.")
                                             chain_valid = False
                                             break # Stop processing this chain

                                if concept_uuid:
                                     concepts_in_chain.append((concept_text, concept_uuid))
                                else: # Should not happen if add_memory_node worked or existing was found
                                     logger.error(f"Failed to get UUID for concept '{concept_text}' in chain. Skipping chain.")
                                     chain_valid = False
                                     break

                            if chain_valid:
                                processed_chains.append(concepts_in_chain) # Add list of (text, uuid) tuples

                        extracted_chains = processed_chains # Use the processed chains with UUIDs
                        logger.info(f"Successfully processed {len(extracted_chains)} causal chains (added new concepts if needed).")
                    else:
                        logger.warning(f"LLM response was valid JSON but not a list. Raw: {llm_response_str}")
                else:
                    logger.warning(f"Could not extract valid JSON list '[]' from causal chain response. Raw: '{llm_response_str}'")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for causal chains: {e}. Raw: '{llm_response_str}'")
            except Exception as e:
                logger.error(f"Unexpected error processing causal chains response: {e}", exc_info=True)

        added_edge_count = 0
        if extracted_chains: # This now contains lists of (text, uuid) tuples
            current_time = time.time()
            for chain_with_uuids in extracted_chains:
                # Add CAUSES edges between consecutive elements in the chain
                for i in range(len(chain_with_uuids) - 1):
                    cause_text, cause_uuid = chain_with_uuids[i]
                    effect_text, effect_uuid = chain_with_uuids[i+1]

                    # UUIDs should be valid because they were just added/retrieved
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
        if llm_response_str and not llm_response_str.startswith("Error:"): # Check for API errors first
            try:
                logger.debug(f"Raw Analogy response: ```{llm_response_str}```")
                # --- Improved JSON Extraction ---
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

                start_bracket = cleaned_response.find('[')
                end_bracket = cleaned_response.rfind(']')

                if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                    json_str = cleaned_response[start_bracket:end_bracket + 1]
                    logger.debug(f"Extracted potential JSON list string: {json_str}")
                    parsed_list = json.loads(json_str) # Attempt to parse the extracted string

                    if isinstance(parsed_list, list):
                        # Validate inner lists (Keep existing validation)
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
                # Log the specific string that failed to parse
                problematic_json_string = json_str if 'json_str' in locals() and json_str else cleaned_response
                logger.error(f"Failed to parse JSON response for analogies: {e}. String Attempted: ```{problematic_json_string}``` Raw: ```{llm_response_str}```")
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
                        # --- Updated required keys to match the new prompt structure ---
                        required_keys = ["core_traits", "recurring_themes", "goals_motivations", "relational_stance", "emotional_profile", "observed_changes", "summary_statement"] # Added observed_changes
                        if all(key in parsed_data for key in required_keys):
                            # Basic type validation (can be expanded)
                            if (isinstance(parsed_data["core_traits"], list) and
                                isinstance(parsed_data["recurring_themes"], list) and
                                isinstance(parsed_data["goals_motivations"], list) and
                                isinstance(parsed_data["relational_stance"], str) and
                                isinstance(parsed_data["emotional_profile"], str) and
                                isinstance(parsed_data["observed_changes"], str) and # Check new key type
                                isinstance(parsed_data["summary_statement"], str) and
                                isinstance(parsed_data.get("significant_event_uuids", []), list)): # Check new key type

                                # Update the entire model
                                self.autobiographical_model = parsed_data
                                self.autobiographical_model["last_updated"] = datetime.now(timezone.utc).isoformat()
                                logger.info(f"Structured Autobiographical Self-Model updated. Summary: '{self.autobiographical_model.get('summary_statement', '')[:100]}...'")

                                # --- NEW: Flag significant event nodes as Core Memory ---
                                if self.config.get('features', {}).get('enable_core_memory', False):
                                    significant_uuids = self.autobiographical_model.get("significant_event_uuids", [])
                                    flagged_count = 0
                                    for node_uuid in significant_uuids:
                                        if node_uuid in self.graph and not self.graph.nodes[node_uuid].get('is_core_memory', False):
                                            self.graph.nodes[node_uuid]['is_core_memory'] = True
                                            flagged_count += 1
                                            logger.info(f"Node {node_uuid[:8]} flagged as CORE MEMORY based on ASM significant event.")
                                            log_tuning_event("CORE_MEMORY_FLAGGED", {
                                                "personality": self.personality,
                                                "node_uuid": node_uuid,
                                                "reason": "asm_significant_event",
                                            })
                                    if flagged_count > 0:
                                        logger.info(f"Flagged {flagged_count} nodes as core memory from ASM.")
                                # --- End Core Memory Flagging ---

                                self._save_memory() # Save the updated model and potentially flagged nodes
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
        min_inferred_strength_threshold = inference_cfg.get('min_strength_threshold', 0.1) # NEW: Minimum strength to add edge
        max_depth = 2 # Fixed for V1
        inferred_edge_count = 0
        current_time = time.time()

        # Define stronger edge types to consider for inference paths
        strong_relation_types = {
            "MENTIONS_CONCEPT", "SUMMARY_OF", "CAUSES", "IS_A", "PART_OF",
            "HIERARCHICAL", "SUPPORTS", "ENABLES"
            # Exclude: ASSOCIATIVE, RELATED_TO, SPACY_*, ANALOGY, INFERRED_RELATED_TO, etc.
        }

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

                # Check A -> B edges, ensuring edge type is strong
                for _, node_b_uuid, edge_ab_data in self.graph.out_edges(node_a_uuid, data=True):
                    if node_b_uuid in candidate_nodes and edge_ab_data.get('type') in strong_relation_types: # Check intermediate node AND edge type
                        # Check B -> C edges, ensuring edge type is strong
                        if self.graph.has_edge(node_b_uuid, node_c_uuid):
                            edge_bc_data = self.graph.get_edge_data(node_b_uuid, node_c_uuid)
                            if edge_bc_data and edge_bc_data.get('type') in strong_relation_types: # Check second edge type
                                found_path = True
                                # Calculate path strength (e.g., product of edge strengths * factor)
                                strength_ab = edge_ab_data.get('base_strength', 0.5) # Use base_strength for calculation
                                strength_bc = edge_bc_data.get('base_strength', 0.5)
                                path_strength = strength_ab * strength_bc * strength_factor
                                path_strength_sum += path_strength
                                path_count += 1
                                logger.debug(f"  Found strong path: {node_a_uuid[:4]}({edge_ab_data.get('type')})->{node_b_uuid[:4]}({edge_bc_data.get('type')})->{node_c_uuid[:4]} (PathStrength: {path_strength:.3f})")
                            # else: logger.debug(f"  Path {node_a_uuid[:4]}->{node_b_uuid[:4]}->{node_c_uuid[:4]} skipped (Edge B->C type '{edge_bc_data.get('type')}' not strong)")
                        # else: logger.debug(f"  Path {node_a_uuid[:4]}->{node_b_uuid[:4]}->{node_c_uuid[:4]} skipped (No edge B->C)")
                    # else: logger.debug(f"  Path starting {node_a_uuid[:4]}->{node_b_uuid[:4]} skipped (Edge A->B type '{edge_ab_data.get('type')}' not strong or B not candidate)")

                if found_path:
                    # Calculate average strength if multiple strong paths exist
                    avg_strength = path_strength_sum / path_count if path_count > 0 else 0.0
                    clamped_strength = max(0.0, min(1.0, avg_strength)) # Clamp 0-1

                    # Add the inferred edge ONLY if strength meets threshold
                    if clamped_strength >= min_inferred_strength_threshold:
                        try:
                            self.graph.add_edge(
                                node_a_uuid, node_c_uuid,
                                type='INFERRED_RELATED_TO',
                                base_strength=clamped_strength, # Use calculated strength
                                last_traversed_ts=current_time # Set timestamp to now
                            )
                            inferred_edge_count += 1
                            logger.info(f"Added inferred edge: {node_a_uuid[:8]} --[INFERRED_RELATED_TO ({clamped_strength:.3f})]--> {node_c_uuid[:8]} (Threshold: {min_inferred_strength_threshold})")
                        except Exception as e:
                            logger.error(f"Error adding inferred edge {node_a_uuid[:8]} -> {node_c_uuid[:8]}: {e}")
                    else:
                         logger.debug(f"  Skipping inferred edge {node_a_uuid[:8]} -> {node_c_uuid[:8]} (Strength {clamped_strength:.3f} < Threshold {min_inferred_strength_threshold})")

        if inferred_edge_count > 0:
            logger.info(f"Added {inferred_edge_count} new inferred relationship edges.")
            # Saving happens after consolidation which calls this.
        else:
            logger.info("No new second-order relationships were inferred.")


    def _infer_second_order_relations(self):
        """Infers typed relationships based on paths of length 2 using LLM."""
        inference_cfg = self.config.get('consolidation', {}).get('inference', {})
        if not inference_cfg.get('enable', False):
            logger.info("Second-order inference disabled by config.")
            return

        logger.info("--- Inferring Typed Second-Order Relationships (LLM) ---")
        strength_factor = inference_cfg.get('strength_factor', 0.3) # Keep strength factor concept
        min_inferred_strength_threshold = inference_cfg.get('min_strength_threshold', 0.2) # Maybe slightly higher threshold for LLM inference?
        inferred_edge_count = 0
        current_time = time.time()

        # Define stronger edge types to consider for inference paths (same as before)
        strong_relation_types = {
            "MENTIONS_CONCEPT", "SUMMARY_OF", "CAUSES", "IS_A", "PART_OF",
            "HIERARCHICAL", "SUPPORTS", "ENABLES"
        }
        # Load target relation types for the LLM prompt
        consolidation_cfg = self.config.get('consolidation', {})
        target_relations = consolidation_cfg.get('target_relation_types', [])
        if not target_relations:
            logger.warning("No target_relation_types defined in config for inference. Using defaults.")
            target_relations = ["CAUSES", "PART_OF", "HAS_PROPERTY", "RELATED_TO", "IS_A", "ENABLES", "PREVENTS", "CONTRADICTS", "SUPPORTS", "EXAMPLE_OF", "MEASURES", "LOCATION_OF"]
        target_relations_str = ", ".join([f"'{r}'" for r in target_relations])

        # Load the new prompt template
        prompt_template = self._load_prompt("infer_typed_relation_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load infer_typed_relation_prompt.txt. Skipping typed inference.")
            return

        # Consider only concept and summary nodes as start/end/intermediate points
        candidate_nodes = [
            uuid for uuid, data in self.graph.nodes(data=True)
            if data.get('node_type') in ['concept', 'summary']
        ]

        if len(candidate_nodes) < 3:
            logger.info("Skipping typed inference: Not enough concept/summary nodes.")
            return

        logger.debug(f"Checking {len(candidate_nodes)} candidate nodes for typed inference...")

        # Iterate through all pairs of candidate nodes (A, C)
        for node_a_uuid in candidate_nodes:
            node_a_data = self.graph.nodes[node_a_uuid]
            for node_c_uuid in candidate_nodes:
                if node_a_uuid == node_c_uuid: continue
                node_c_data = self.graph.nodes[node_c_uuid]

                # Check if a direct edge already exists (A->C or C->A)
                if self.graph.has_edge(node_a_uuid, node_c_uuid) or self.graph.has_edge(node_c_uuid, node_a_uuid):
                    continue

                # Find paths of length 2 (A -> B -> C) via strong edge types
                strong_paths_found = [] # Store tuples: (node_b_uuid, strength_ab, strength_bc)
                for _, node_b_uuid, edge_ab_data in self.graph.out_edges(node_a_uuid, data=True):
                    if node_b_uuid in candidate_nodes and edge_ab_data.get('type') in strong_relation_types:
                        if self.graph.has_edge(node_b_uuid, node_c_uuid):
                            edge_bc_data = self.graph.get_edge_data(node_b_uuid, node_c_uuid)
                            if edge_bc_data and edge_bc_data.get('type') in strong_relation_types:
                                strength_ab = edge_ab_data.get('base_strength', 0.5)
                                strength_bc = edge_bc_data.get('base_strength', 0.5)
                                strong_paths_found.append((node_b_uuid, strength_ab, strength_bc))

                if strong_paths_found:
                    # Select the path with the highest combined strength (product) to feed to LLM
                    best_path = max(strong_paths_found, key=lambda p: p[1] * p[2])
                    node_b_uuid, strength_ab, strength_bc = best_path
                    node_b_data = self.graph.nodes[node_b_uuid]

                    # Calculate overall path strength for the edge weight
                    path_strength = strength_ab * strength_bc * strength_factor
                    clamped_strength = max(0.0, min(1.0, path_strength))

                    # Only proceed if potential strength meets threshold
                    if clamped_strength < min_inferred_strength_threshold:
                        logger.debug(f"  Skipping inference {node_a_uuid[:4]}->{node_c_uuid[:4]} via {node_b_uuid[:4]} (Strength {clamped_strength:.3f} < Threshold {min_inferred_strength_threshold})")
                        continue

                    # Prepare context for LLM (e.g., text of A, B, C)
                    context_text = f"Context A: {node_a_data.get('text', '')}\nContext B: {node_b_data.get('text', '')}\nContext C: {node_c_data.get('text', '')}"
                    # Format prompt
                    full_prompt = prompt_template.format(
                        context_text=context_text[:1500], # Limit context length
                        concept_a_text=node_a_data.get('text', ''),
                        concept_a_uuid=node_a_uuid,
                        concept_b_text=node_b_data.get('text', ''),
                        concept_b_uuid=node_b_uuid,
                        concept_c_text=node_c_data.get('text', ''),
                        concept_c_uuid=node_c_uuid,
                        target_relations_str=target_relations_str
                    )

                    # Call LLM to infer relation type
                    logger.debug(f"Calling LLM to infer relation type for {node_a_uuid[:4]} -> {node_c_uuid[:4]} via {node_b_uuid[:4]}")
                    # Use a suitable LLM config - maybe 'consolidation_relation' or a new one? Let's use 'consolidation_relation'.
                    llm_response_str = self._call_configured_llm('consolidation_relation', prompt=full_prompt)

                    inferred_relation_type = "RELATED_TO" # Default fallback
                    if llm_response_str and not llm_response_str.startswith("Error:"):
                        try:
                            match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
                            if match:
                                json_str = match.group(0)
                                parsed_data = json.loads(json_str)
                                relation = parsed_data.get("inferred_relation")
                                if relation and relation in target_relations: # Validate against target list
                                    inferred_relation_type = relation
                                    logger.debug(f"  LLM inferred relation: {inferred_relation_type}")
                                else:
                                    logger.warning(f"  LLM returned invalid relation '{relation}'. Defaulting to RELATED_TO. Raw: {llm_response_str}")
                            else:
                                logger.warning(f"  Could not extract JSON from inference response. Defaulting to RELATED_TO. Raw: {llm_response_str}")
                        except Exception as e:
                            logger.error(f"  Error parsing inference response: {e}. Defaulting to RELATED_TO. Raw: {llm_response_str}")
                    else:
                         logger.error(f"  LLM call failed for inference: {llm_response_str}. Defaulting to RELATED_TO.")

                    # Add the edge with the inferred type and calculated strength
                    try:
                        # Check if edge already exists with *any* type before adding
                        if not self.graph.has_edge(node_a_uuid, node_c_uuid):
                            self.graph.add_edge(
                                node_a_uuid, node_c_uuid,
                                type=inferred_relation_type, # Use inferred type
                                base_strength=clamped_strength,
                                last_traversed_ts=current_time,
                                inferred=True # Mark as inferred
                            )
                            inferred_edge_count += 1
                            logger.info(f"Added inferred edge: {node_a_uuid[:8]} --[{inferred_relation_type} ({clamped_strength:.3f})]--> {node_c_uuid[:8]}")
                        else:
                             logger.debug(f"  Skipping add: Direct edge {node_a_uuid[:8]} -> {node_c_uuid[:8]} already exists.")
                    except Exception as e:
                        logger.error(f"Error adding inferred edge {node_a_uuid[:8]} -> {node_c_uuid[:8]}: {e}")

        if inferred_edge_count > 0:
            logger.info(f"Added {inferred_edge_count} new typed inferred relationship edges.")
            # Saving happens after consolidation which calls this.
        else:
            logger.info("No new typed second-order relationships were inferred.")


    def run_consolidation(self, active_nodes_to_process=None):
        """
        Orchestrates the memory consolidation process: summarization, concept extraction,
        relation extraction, core memory flagging, and pruning.
        """
        logger.info("--- Running Consolidation ---")
        # --- Tuning Log: Consolidation Start ---
        log_tuning_event("CONSOLIDATION_START", {"personality": self.personality})

        consolidation_cfg = self.config.get('consolidation', {})
        min_nodes_for_consolidation = consolidation_cfg.get('min_nodes', 5)
        turn_count_for_consolidation = consolidation_cfg.get('turn_count', 10)
        prune_summarized = consolidation_cfg.get('prune_summarized_turns', True) # Default changed to True
        concept_sim_threshold = consolidation_cfg.get('concept_similarity_threshold', 0.3)
        features_cfg = self.config.get('features', {})
        rich_assoc_enabled = features_cfg.get('enable_rich_associations', False)
        emotion_analysis_enabled = features_cfg.get('enable_emotion_analysis', False)
        core_mem_enabled = features_cfg.get('enable_core_memory', False) # Check if core memory feature is on

        # --- 1. Select Nodes ---
        nodes_to_consolidate = self._select_nodes_for_consolidation(count=turn_count_for_consolidation)
        nodes_to_process = []
        nodes_data = []
        for uuid in nodes_to_consolidate:
            if uuid in self.graph:
                node_data = self.graph.nodes[uuid]
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
            log_tuning_event("CONSOLIDATION_END", {
                "personality": self.personality,
                "status": "skipped_min_nodes",
                "nodes_processed_count": len(nodes_to_process),
                "min_nodes_required": min_nodes_for_consolidation,
            })
            return

        logger.info(f"Consolidating {len(nodes_to_process)} nodes: {nodes_to_process}")
        log_tuning_event("CONSOLIDATION_NODES_SELECTED", {
            "personality": self.personality,
            "selected_node_uuids": nodes_to_process,
        })

        # --- 2. Prepare Context ---
        nodes_data.sort(key=lambda x: x.get('timestamp', ''))
        context_text = "\n".join([f"{d.get('speaker', '?')}: {d.get('text', '')}" for d in nodes_data])
        logger.debug(f"Consolidation context text (first 200 chars):\n{context_text[:200]}...")

        # --- 3. Summarization ---
        summary_node_uuid, summary_created = self._consolidate_summarize(context_text=context_text,
                                                                        nodes_data=nodes_data,
                                                                        processed_node_uuids=nodes_to_process)
        log_tuning_event("CONSOLIDATION_SUMMARY", {
            "personality": self.personality,
            "summary_created": summary_created,
            "summary_node_uuid": summary_node_uuid,
            "source_node_uuids": nodes_to_process,
        })

        # --- 4. Concept Extraction (LLM) ---
        llm_concepts = self._consolidate_extract_concepts(context_text)
        log_tuning_event("CONSOLIDATION_CONCEPTS_EXTRACTED", {
            "personality": self.personality,
            "llm_extracted_concepts": llm_concepts,
        })

        # --- 5. Concept Deduplication & Node Management ---
        concept_node_map = {}
        newly_added_concepts = []
        processed_llm_concepts = set()

        if llm_concepts:
            # (Existing concept deduplication logic remains the same)
            logger.info(f"LLM Concepts to process: {llm_concepts}")
            for concept_text in llm_concepts:
                if concept_text in processed_llm_concepts: continue
                processed_llm_concepts.add(concept_text)
                logger.debug(f"Processing LLM concept: '{concept_text}'")
                similar_concepts = self._search_similar_nodes(concept_text, k=1, node_type_filter='concept')

                existing_uuid = None
                if similar_concepts:
                    best_match_uuid, best_match_score = similar_concepts[0]
                    if best_match_score <= concept_sim_threshold:
                        existing_uuid = best_match_uuid
                        logger.info(
                            f"Concept '{concept_text}' matches existing node {existing_uuid[:8]} (Score: {best_match_score:.3f})")

                if existing_uuid:
                    concept_node_map[concept_text] = existing_uuid
                    if existing_uuid in self.graph: self.graph.nodes[existing_uuid]['last_accessed_ts'] = time.time()
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
            log_tuning_event("CONSOLIDATION_CONCEPTS_PROCESSED", {
                "personality": self.personality,
                "final_concept_map": concept_node_map, # text -> uuid
                "newly_added_concept_uuids": newly_added_concepts,
            })
        else:
            logger.info("No concepts extracted by LLM.")

        # --- 6. Link Concepts to Source Nodes ---
        # (Existing concept linking logic remains the same)
        if concept_node_map:
            current_time = time.time()
            for concept_text, concept_uuid in concept_node_map.items():
                if concept_uuid not in self.graph: continue
                # Link to summary
                if summary_node_uuid and summary_node_uuid in self.graph:
                    try:
                        if not self.graph.has_edge(summary_node_uuid, concept_uuid):
                            self.graph.add_edge(summary_node_uuid, concept_uuid, type='MENTIONS_CONCEPT',
                                                base_strength=0.7, last_traversed_ts=current_time)
                    except Exception as e:
                        logger.error(
                            f"Error adding MENTIONS_CONCEPT edge from summary {summary_node_uuid[:8]} to {concept_uuid[:8]}: {e}")
                # Link to original turns
                for node_uuid in nodes_to_process:
                    if node_uuid in self.graph and concept_text.lower() in self.graph.nodes[node_uuid].get('text', '').lower():
                        try:
                            if not self.graph.has_edge(node_uuid, concept_uuid):
                                self.graph.add_edge(node_uuid, concept_uuid, type='MENTIONS_CONCEPT', base_strength=0.7,
                                                    last_traversed_ts=current_time)
                                logger.debug(f"Added MENTIONS_CONCEPT edge: {node_uuid[:8]} -> {concept_uuid[:8]}")
                            else:
                                self.graph.edges[node_uuid, concept_uuid]['last_traversed_ts'] = current_time
                                logger.debug(f"Updated MENTIONS_CONCEPT edge timestamp: {node_uuid[:8]} -> {concept_uuid[:8]}")
                        except Exception as e:
                            logger.error(
                                f"Error adding/updating MENTIONS_CONCEPT edge from turn {node_uuid[:8]} to {concept_uuid[:8]}: {e}")

        # --- 7. Relation Extraction ---
        # (Existing relation extraction logic remains the same)
        if concept_node_map:
            spacy_doc = None
            if rich_assoc_enabled and self.nlp:
                try:
                    logger.info("Using spaCy for potential pre-processing of context...")
                    spacy_doc = self.nlp(context_text)
                    self._consolidate_extract_rich_relations(context_text, concept_node_map, spacy_doc)
                    self._consolidate_extract_causal_chains(context_text, concept_node_map)
                    self._consolidate_extract_analogies(context_text, concept_node_map)
                except Exception as spacy_err:
                    logger.error(f"Error processing context with spaCy: {spacy_err}. Falling back.", exc_info=True)
                    # Fallback to V1 methods
                    self._consolidate_extract_v1_associative(concept_node_map)
                    self._consolidate_extract_hierarchy(concept_node_map)
                    self._consolidate_extract_causal_chains(context_text, concept_node_map)
                    self._consolidate_extract_analogies(context_text, concept_node_map)
            else: # Rich associations disabled or spaCy failed/unavailable
                if not self.nlp and rich_assoc_enabled:
                    logger.warning(
                        "Rich associations enabled but spaCy model not loaded. Falling back to V1 relation extraction.")
                logger.info("Running V1 Associative/Hierarchy/Causal/Analogy extraction (LLM only).")
                self._consolidate_extract_v1_associative(concept_node_map)
                self._consolidate_extract_hierarchy(concept_node_map)
                self._consolidate_extract_causal_chains(context_text, concept_node_map)
                self._consolidate_extract_analogies(context_text, concept_node_map)
        else:
            logger.info("Skipping relation extraction as no concepts were identified.")
            log_tuning_event("CONSOLIDATION_RELATIONS_SKIPPED", {
                "personality": self.personality,
                "reason": "no_concepts_identified",
            })

        # --- 7b. Emotion Analysis (DEPRECATED - Handled by EmotionalCore if enabled) ---
        # (Old text2emotion logic removed)

        # --- 8. NEW: Core Memory Flagging ---
        flagged_count = 0
        if core_mem_enabled:
            logger.info("Checking nodes for Core Memory flagging...")
            # Define criteria from config
            core_mem_cfg = self.config.get('core_memory', {})
            saliency_threshold = core_mem_cfg.get('saliency_threshold', 0.95)
            # --- Get new access count threshold from config ---
            access_threshold = core_mem_cfg.get('access_count_threshold', 20) # Default to 20 if not set
            # --- Define keywords for identifying appearance summaries (heuristic) ---
            appearance_keywords = ["appearance", "look like", "wavy brown hair", "tattoo", "petite", "curvy", "freckles", "goth", "lipstick"] # Add more as needed

            # Identify nodes to check: Summary node, Concept nodes, maybe original turns if not pruned
            nodes_to_check_for_core = set()
            if summary_node_uuid: nodes_to_check_for_core.add(summary_node_uuid)
            nodes_to_check_for_core.update(concept_node_map.values())
            if not prune_summarized: nodes_to_check_for_core.update(nodes_to_process) # Check original turns if not pruned

            core_flag_details = {} # For logging

            for node_uuid in nodes_to_check_for_core:
                if node_uuid not in self.graph: continue # Node might have been deleted (e.g., concept merged)
                node_data = self.graph.nodes[node_uuid]

                # Skip if already core
                if node_data.get('is_core_memory', False): continue

                reason = None
                # Criterion 1: Appearance Summary Keywords
                if node_data.get('node_type') == 'summary':
                    node_text_lower = node_data.get('text', '').lower()
                    if any(keyword in node_text_lower for keyword in appearance_keywords):
                        reason = "appearance_keywords_in_summary"

                # Criterion 2: High Saliency
                if not reason and node_data.get('saliency_score', 0.0) >= saliency_threshold:
                    reason = f"high_saliency ({node_data['saliency_score']:.3f} >= {saliency_threshold})"

                # Criterion 3: High Access Count
                if not reason and node_data.get('access_count', 0) >= access_threshold:
                    reason = f"high_access_count ({node_data['access_count']} >= {access_threshold})"

                # If any criterion met, flag as core
                if reason:
                    node_data['is_core_memory'] = True
                    flagged_count += 1
                    logger.info(f"Node {node_uuid[:8]} flagged as CORE MEMORY (Reason: {reason})")
                    core_flag_details[node_uuid] = reason

            if flagged_count > 0:
                logger.info(f"Flagged {flagged_count} nodes as core memory during consolidation.")
                log_tuning_event("CORE_MEMORY_FLAGGED_CONSOLIDATION", {
                    "personality": self.personality,
                    "flagged_count": flagged_count,
                    "details": core_flag_details,
                })
        else:
            logger.info("Core memory feature disabled, skipping flagging during consolidation.")


        # --- 9. Pruning Summarized Nodes (Optional) ---
        pruned_count = 0 # Initialize here
        saved_in_consolidation = False # Initialize save flag
        if prune_summarized and summary_created:
            logger.info("Pruning original turn nodes that were summarized...")
            nodes_to_prune_list = list(nodes_to_process) # Create a copy to iterate over
            for uuid_to_prune in nodes_to_prune_list:
                # Check if node still exists before trying to delete
                if uuid_to_prune in self.graph:
                    if self.delete_memory_entry(uuid_to_prune):
                        pruned_count += 1
                    else:
                        logger.warning(f"Failed to prune summarized node {uuid_to_prune[:8]} (maybe deleted by other process?).")
                else:
                    logger.debug(f"Node {uuid_to_prune[:8]} already gone before pruning loop.")

            logger.info(f"Pruned {pruned_count} summarized turn nodes.")
            log_tuning_event("CONSOLIDATION_PRUNING", {
                "personality": self.personality,
                "pruning_enabled": prune_summarized,
                "summary_created": summary_created,
                "pruned_node_count": pruned_count,
                "pruned_node_uuids": nodes_to_process[:pruned_count],
            })
            # Note: delete_memory_entry rebuilds index and saves memory
            saved_in_consolidation = True # Memory was saved during pruning
        else:
            # Save memory if pruning is disabled or no summary was created,
            # but other changes (concepts, relations, core flags) might have occurred.
            logger.info("Saving memory state after consolidation (no pruning or pruning disabled/skipped).")
            self._save_memory()
            saved_in_consolidation = True

        # --- 10. Update Long-Term Drives & ASM (moved after pruning/saving logic) ---
        # Update Short-Term Drive State based on the consolidated context
        self._update_drive_state(context_text=context_text)

        # Update Long-Term Drive State (Less Frequently)
        drive_cfg = self.config.get('subconscious_drives', {})
        lt_update_interval = drive_cfg.get('long_term_update_interval_consolidations', 0)
        if not hasattr(self, '_consolidation_counter'): self._consolidation_counter = 0
        self._consolidation_counter += 1
        logger.debug(f"Consolidation counter: {self._consolidation_counter}")
        if lt_update_interval > 0 and self._consolidation_counter >= lt_update_interval:
            logger.info(f"Long-term drive update interval ({lt_update_interval}) reached. Triggering update.")
            self._update_long_term_drives() # This saves drives if changed
            self._consolidation_counter = 0
        else:
            logger.debug(f"Skipping long-term drive update (Interval: {lt_update_interval}, Count: {self._consolidation_counter}).")

        # Update Autobiographical Model after consolidation
        self._generate_autobiographical_model() # This saves ASM if changed

        # Infer Second-Order Relationships (Runs after all node additions/updates)
        self._infer_second_order_relations() # This does not save memory itself

        # --- Final Save (if not already saved by pruning/LT Drive/ASM updates) ---
        # Re-check if saving already happened
        # Note: _update_long_term_drives and _generate_autobiographical_model now save internally if changes occur.
        # We only need a final save here if none of those triggered a save AND pruning didn't happen.
        # This logic might be slightly redundant but ensures a save occurs.
        # Let's simplify: always call save at the end, _save_memory handles efficiency.
        logger.debug("Running final save check after consolidation steps.")
        self._save_memory()
        saved_in_consolidation = True # Mark as saved

        logger.info("--- Consolidation Finished ---")
        # --- Tuning Log: Consolidation End ---
        log_tuning_event("CONSOLIDATION_END", {
            "personality": self.personality,
            "memory_saved": saved_in_consolidation,
            "summary_created": summary_created,
            "summary_node_uuid": summary_node_uuid,
            "new_concepts_added": len(newly_added_concepts),
            "pruned_node_count": pruned_count,
            "core_nodes_flagged": flagged_count,
            # Add counts for different relation types if available
        })
     
    
    def plan_and_execute(self, user_input: str, conversation_history: list) -> list[tuple[bool, str, str, bool]]:
        """
        Plans and executes workspace actions based on user input and conversation context.
        Called separately by the worker thread if flagged by process_interaction.
        Returns list of tuples: (success, message, action_suffix, silent_and_successful)
        """
        task_id = str(uuid.uuid4())[:8] # Generate a short ID for this planning task
        logger.info(f"--- Starting Workspace Planning & Execution [ID: {task_id}] for input: '{strip_emojis(user_input[:50])}...' ---") # Strip emojis
        workspace_action_results = [] # Initialize results list

        try:
            # 1. Retrieve Relevant Memories
            logger.debug(f"[{task_id}] Retrieving memories for planning context...")
            query_type = self._classify_query_type(user_input)
            max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7)
            initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes, query_type=query_type)
            initial_uuids = [uid for uid, score in initial_nodes]
            memory_chain_data = []
            effective_mood = self.last_interaction_mood # Use last known mood
            if initial_uuids:
                concepts_for_retrieval = self.last_interaction_concept_uuids
                retrieved_nodes, effective_mood = self.retrieve_memory_chain(
                    initial_node_uuids=initial_uuids,
                    recent_concept_uuids=list(concepts_for_retrieval),
                    current_mood=effective_mood # Pass the potentially updated mood
                )
                memory_chain_data = retrieved_nodes
            logger.info(f"[{task_id}] Retrieved {len(memory_chain_data)} memories for planning context.")

            # 2. Prepare Planning Prompt Context
            logger.debug(f"[{task_id}] Preparing context for planning prompt...")
            planning_history_text = "\n".join([f"{turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}" for turn in conversation_history[-5:]]) # Strip emojis
            planning_memory_text = "\n".join([f"- {mem.get('speaker', '?')} ({self._get_relative_time_desc(mem.get('timestamp',''))}): {strip_emojis(mem.get('text', ''))}" for mem in memory_chain_data]) # Strip emojis
            if not planning_memory_text: planning_memory_text = "[No relevant memories retrieved]"

            workspace_files, list_msg = file_manager.list_files(self.config, self.personality)
            if workspace_files is None:
                logger.error(f"[{task_id}] Failed list workspace files for planning context: {list_msg}")
                workspace_files_list_str = "[Error retrieving file list]"
            elif not workspace_files:
                workspace_files_list_str = "[Workspace is empty]"
            else:
                workspace_files_list_str = "\n".join([f"- {fname}" for fname in sorted(workspace_files)])
            logger.debug(f"[{task_id}] Workspace files for planning prompt:\n{workspace_files_list_str}")

            asm_context_str = "[AI Self-Model: Not Available]"
            if self.autobiographical_model:
                 # ... (ASM formatting remains the same) ...
                 try:
                     asm_parts = []
                     if self.autobiographical_model.get("summary_statement"): asm_parts.append(f"- Summary: {self.autobiographical_model['summary_statement']}")
                     if self.autobiographical_model.get("core_traits"): asm_parts.append(f"- Traits: {', '.join(self.autobiographical_model['core_traits'])}")
                     if self.autobiographical_model.get("goals_motivations"): asm_parts.append(f"- Goals/Motivations: {', '.join(self.autobiographical_model['goals_motivations'])}")
                     if self.autobiographical_model.get("relational_stance"): asm_parts.append(f"- My Role: {self.autobiographical_model['relational_stance']}")
                     if self.autobiographical_model.get("emotional_profile"): asm_parts.append(f"- Emotional Profile: {self.autobiographical_model['emotional_profile']}")
                     if asm_parts: asm_context_str = "\n".join(asm_parts)
                 except Exception as asm_fmt_e: logger.error(f"Error formatting ASM for planning prompt: {asm_fmt_e}"); asm_context_str = "[AI Self-Model: Error Formatting]"
            logger.debug(f"[{task_id}] ASM context for planning prompt:\n{asm_context_str}")

            planning_prompt_template = self._load_prompt("workspace_planning_prompt.txt")
            if not planning_prompt_template:
                logger.error(f"[{task_id}] Workspace planning prompt template missing. Cannot generate plan.")
                workspace_action_results.append((False, "Internal Error: Planning prompt missing.", "planning_error", False)) # Added silent flag
                return workspace_action_results

            try:
                # --- Manual Prompt Construction using .replace() ---
                planning_prompt = planning_prompt_template # Start with the raw template
                replacements = { # Define placeholders and their values
                    "{user_request}": user_input, "{history_text}": planning_history_text,
                    "{memory_text}": planning_memory_text, "{workspace_files_list}": workspace_files_list_str,
                    "{asm_context}": asm_context_str
                }
                for placeholder, value in replacements.items(): # Iteratively replace
                    str_value = str(value) if value is not None else "" # Ensure value is string
                    planning_prompt = planning_prompt.replace(placeholder, str_value)
                remaining_placeholders = re.findall(r'\{[a-zA-Z0-9_]+\}', planning_prompt)
                if remaining_placeholders: raise ValueError(f"Unreplaced placeholders found: {remaining_placeholders}")
                logger.info(f"[{task_id}] Sending workspace planning prompt to LLM...") # Changed level to INFO
                logger.debug(f"[{task_id}] Planning Prompt Preview:\n{planning_prompt[:500]}...") # Added preview log

            except Exception as replace_e:
                 logger.error(f"[{task_id}] Error during planning prompt construction: {replace_e}", exc_info=True)
                 workspace_action_results.append((False, f"Internal Error: Planning prompt construction: {replace_e}", "planning_format_error", False))
                 return workspace_action_results

            # 3. Call Planning LLM
            plan_response_str = self._call_configured_llm('workspace_planning', prompt=planning_prompt)

            # 4. Parse Plan
            parsed_plan = None
            logger.debug(f"[{task_id}] Raw planning LLM response: ```{plan_response_str}```") # Log raw response
            if plan_response_str and not plan_response_str.startswith("Error:"):
                try:
                    # Improved JSON list extraction
                    match = re.search(r'(\[.*?\])', plan_response_str, re.DOTALL | re.MULTILINE)
                    if match:
                        plan_json_str = match.group(1)
                        logger.debug(f"[{task_id}] Extracted plan JSON string: {plan_json_str}") # Log extracted string
                        parsed_plan = json.loads(plan_json_str)
                        if not isinstance(parsed_plan, list):
                            logger.error(f"[{task_id}] Parsed plan is not a list: {type(parsed_plan)}")
                            parsed_plan = None # Treat as invalid
                        else:
                             logger.info(f"[{task_id}] Successfully parsed plan: {parsed_plan}") # Log parsed plan
                    else:
                        logger.warning(f"[{task_id}] Could not extract JSON list '[]' from planning response.")
                        parsed_plan = None # No valid plan found
                except json.JSONDecodeError as e:
                    logger.error(f"[{task_id}] Failed to parse JSON plan response: {e}. Raw: '{plan_response_str}'")
                    parsed_plan = None
            elif plan_response_str.startswith("Error:"):
                 logger.error(f"[{task_id}] Workspace planning LLM call failed: {plan_response_str}")
                 workspace_action_results.append((False, f"Planning Error: {plan_response_str}", "planning_llm_error", False))
                 return workspace_action_results # Return LLM error result
            else: # Empty response from LLM
                logger.warning(f"[{task_id}] Workspace planning LLM returned empty response.")
                parsed_plan = [] # Treat as empty plan

            # 5. Execute Plan (if valid)
            if parsed_plan is not None: # Check if parsing was successful (even if list is empty)
                if isinstance(parsed_plan, list) and len(parsed_plan) > 0:
                    logger.info(f"[{task_id}] Plan contains {len(parsed_plan)} step(s). Instantiating WorkspaceAgent...")
                    try:
                        # Pass the current client instance (self) to the agent
                        agent = WorkspaceAgent(self)
                        logger.info(f"[{task_id}] Calling WorkspaceAgent.execute_plan...")
                        workspace_action_results = agent.execute_plan(parsed_plan)
                        logger.info(f"[{task_id}] WorkspaceAgent execution finished. Results count: {len(workspace_action_results)}")
                    except Exception as agent_exec_e:
                         logger.error(f"[{task_id}] Error during WorkspaceAgent execution: {agent_exec_e}", exc_info=True)
                         workspace_action_results.append((False, f"Internal Error: Agent execution failed: {agent_exec_e}", "agent_execution_error", False))
                elif isinstance(parsed_plan, list) and len(parsed_plan) == 0:
                    logger.info(f"[{task_id}] LLM generated an empty plan. No workspace actions executed.")
                    # Return empty results list (or a specific success message?)
                    # workspace_action_results.append((True, "No actions required by plan.", "no_actions", True)) # Example message
                else: # Plan parsed but invalid (e.g., not list) - handled above by setting parsed_plan to None
                     logger.error(f"[{task_id}] Parsed plan was invalid. No actions executed.")
                     workspace_action_results.append((False, "Internal Error: Invalid plan structure from LLM.", "planning_invalid_plan", False))
            else: # Plan parsing failed or no plan found in response
                 logger.warning(f"[{task_id}] No valid workspace plan parsed from LLM response. No actions executed.")
                 workspace_action_results.append((False, "Planning Error: Could not parse plan from LLM.", "planning_parse_fail", False))

        except Exception as plan_exec_e:
             logger.error(f"[{task_id}] Unexpected error during plan_and_execute: {plan_exec_e}", exc_info=True)
             error_type = type(plan_exec_e).__name__
             workspace_action_results.append((False, f"Internal Error ({error_type}): {plan_exec_e}", "planning_exception", False))

    def _get_kg_context_for_emotion(self, user_input: str, k: int = 3) -> str:
        """
        Retrieves relevant context from the knowledge graph for emotional analysis.
        (Basic implementation: ASM summary + text from k most similar nodes).
        """
        context_parts = []
        # 1. Add ASM Summary
        if self.autobiographical_model:
            asm_summary = self.autobiographical_model.get("summary_statement", "")
            if asm_summary:
                context_parts.append(f"AI Self-Summary: {asm_summary}")

        # 2. Add text from k most similar nodes to user input
        if k > 0 and self.index and self.index.ntotal > 0:
            try:
                similar_nodes = self._search_similar_nodes(user_input, k=k)
                if similar_nodes:
                    context_parts.append("\nRelevant Past Snippets:")
                    for node_uuid, score in similar_nodes:
                        if node_uuid in self.graph:
                            node_data = self.graph.nodes[node_uuid]
                            speaker = node_data.get('speaker', '?')
                            text = node_data.get('text', '')
                            context_parts.append(f"- {speaker}: {text[:150]}{'...' if len(text) > 150 else ''}")
            except Exception as e:
                logger.error(f"Error retrieving similar nodes for emotional context: {e}", exc_info=True)

        if not context_parts:
            return "[No relevant KG context found]"

        return "\n".join(context_parts)


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
