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
from emotional_core import EmotionalCore


# --- Profiling Logger Setup ---
profiling_logger = logging.getLogger('ProfilingLogger')
profiling_logger.setLevel(logging.INFO) # INFO level is fine for structured timing data
profiling_logger.propagate = False # Don't let it go to the root logger

# Remove existing handlers to prevent duplicates if re-running in same session
for handler in profiling_logger.handlers[:]:
    profiling_logger.removeHandler(handler)
    handler.close()

try:
    log_dir = os.path.join(os.path.dirname(__file__), 'logs') # Assuming logs dir is in the same dir as this script
    os.makedirs(log_dir, exist_ok=True)
    profiling_log_file = os.path.join(log_dir, 'performance_profile.csv') # Use CSV for easy analysis

    # Use a FileHandler
    # For CSV, we'll format manually, so the logger's formatter is less critical but still good practice
    profiling_handler = logging.FileHandler(profiling_log_file, mode='a', encoding='utf-8')
    
    # If the file is new, write a header row for the CSV
    if os.path.getsize(profiling_log_file) == 0:
        profiling_handler.stream.write("Timestamp,InteractionID,Personality,StepName,DurationSeconds,ContextInfo\n")

    # Basic formatter, as we'll mostly log pre-formatted CSV strings
    profiling_formatter = logging.Formatter('%(message)s') # Just pass the message through
    profiling_handler.setFormatter(profiling_formatter)
    profiling_logger.addHandler(profiling_handler)
    
    # Test log (optional)
    # profiling_logger.info(f"{datetime.now().isoformat()},INIT,System,LoggerInit,0.0,LoggerInitialized")

except Exception as e:
    # Fallback to console if file logging fails for profiling
    print(f"!!! FAILED TO CONFIGURE PROFILING LOGGER: {e}. Profiling logs might not be saved to file. !!!", file=sys.stderr)
    profiling_logger.disabled = True # Or handle more gracefully
    # Add a console handler as a fallback for critical profiling messages
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - PROFILE - %(message)s')
    console_handler.setFormatter(console_formatter)
    profiling_logger.addHandler(console_handler)
    profiling_logger.warning("Profiling logger falling back to console due to file setup error.")

def log_profile_event(interaction_id: str, personality: str, step_name: str, duration_seconds: float, context_info: str = ""):
    """Helper to log a structured profiling event to the CSV."""
    if hasattr(profiling_logger, 'disabled') and profiling_logger.disabled:
        return
    try:
        timestamp = datetime.now().isoformat()
        # Sanitize context_info to ensure it doesn't break CSV (remove commas, newlines)
        safe_context_info = str(context_info).replace(",", ";").replace("\n", " ")
        log_message = f"{timestamp},{interaction_id},{personality},{step_name},{duration_seconds:.4f},{safe_context_info}"
        profiling_logger.info(log_message)
    except Exception as e:
        print(f"Error logging profile event '{step_name}': {e}", file=sys.stderr)


# Keywords that might indicate a need for workspace planning (STILL USEFUL for initial check)
WORKSPACE_KEYWORDS = [
    "file", "save", "create", "write", "append", "read", "open", "list", "delete", "remove",
    "calendar", "event", "schedule", "meeting", "appointment", "remind", "task",
    "note", "document", "report", "summary", "code",
    "workspace", "directory",
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
    logging.warning("zoneinfo module not found. Using UTC. Consider `pip install tzdata`.")
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception
from collections import defaultdict

import file_manager

DEFAULT_CONFIG_PATH = "config.yaml"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tuning_logger = logging.getLogger('TuningLogger')
tuning_logger.setLevel(logging.DEBUG)
tuning_logger.propagate = False

for handler in tuning_logger.handlers[:]:
    tuning_logger.removeHandler(handler)
    handler.close()
try:
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tuning_log_file = os.path.join(log_dir, 'tuning_log.jsonl')
    tuning_handler = logging.FileHandler(tuning_log_file, mode='a', encoding='utf-8')
    tuning_formatter = logging.Formatter('%(message)s')
    tuning_handler.setFormatter(tuning_formatter)
    tuning_logger.addHandler(tuning_handler)
    tuning_logger.info(json.dumps({"event_type": "TUNING_LOG_INIT", "timestamp": datetime.now(timezone.utc).isoformat()}))
except Exception as e:
    logger.error(f"!!! Failed to configure tuning logger: {e}. Tuning logs will not be saved. !!!", exc_info=True)
    tuning_logger.disabled = True

def log_tuning_event(event_type: str, data: dict):
    if tuning_logger.disabled: return
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data
        }
        tuning_logger.debug(json.dumps(log_entry))
    except Exception as e:
        logger.error(f"Error logging tuning event '{event_type}': {e}", exc_info=True)

def strip_emojis(text: str) -> str:
    if not isinstance(text, str): return text
    try:
        cleaned_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogatepass')
        return EMOJI_PATTERN.sub(r'', cleaned_text)
    except Exception:
        return EMOJI_PATTERN.sub(r'', text)

@dataclasses.dataclass
class InteractionResult:
    final_response_text: str
    inner_thoughts: str | None = None
    memories_used: list = dataclasses.field(default_factory=list)
    user_node_uuid: str | None = None
    ai_node_uuid: str | None = None
    needs_planning: bool = False
    # --- NEW: Add field for NLU-detected interaction type ---
    detected_interaction_type: str | None = None # e.g., "topic_change", "question", "statement"
    detected_intent_details: dict | None = None # e.g., {"new_topic_keywords": ["X"]}
    # --- ADDED THE MISSING FIELD ---
    extracted_ai_action_tag_json: str | None = None # Assuming it's a string (JSON string) or None
    


class GraphMemoryClient:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH, personality_name=None):
        logger.info(f"Initializing GraphMemoryClient (Personality: {personality_name or 'Default'})...")
        self._load_config(config_path)

        if personality_name is None:
            personality_name = self.config.get('default_personality', 'default')
            logger.info(f"No personality specified, using default: {personality_name}")
        self.personality = personality_name

        base_memory_path = self.config.get('base_memory_path', 'memory_sets')
        self.data_dir = os.path.join(base_memory_path, self.personality)
        logger.info(f"Using data directory for '{self.personality}': {os.path.abspath(self.data_dir)}")

        # Construct file paths relative to the specific data_dir
        self.graph_file = os.path.join(self.data_dir, "memory_graph.json")
        self.index_file = os.path.join(self.data_dir, "memory_index.faiss")
        self.embeddings_file = os.path.join(self.data_dir, "memory_embeddings.npy")
        self.mapping_file = os.path.join(self.data_dir, "memory_mapping.json")
        self.asm_file = os.path.join(self.data_dir, "asm.json")
        self.drives_file = os.path.join(self.data_dir, "drives.json")
        self.last_conversation_file = os.path.join(self.data_dir, "last_conversation.json")

        # API URLs
        self.kobold_api_url = self.config.get('kobold_api_url', "http://localhost:5001/api/v1/generate")
        base_kobold_url = self.kobold_api_url.rsplit('/api/', 1)[0] if '/api/' in self.kobold_api_url else self.kobold_api_url
        self.kobold_chat_api_url = self.config.get('kobold_chat_api_url', f"{base_kobold_url}/v1/chat/completions")
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
        self.embedder = None
        self.embedding_dim = 0
        self.nlp = None
        self.last_interaction_concept_uuids = set()
        self.last_interaction_mood = (0.0, 0.1)
        self.autobiographical_model = {}
        self.drive_state = {
            "short_term": {},
            "long_term": {},
            "dynamic_baselines": {} # This will be calculated, not stored in file
        }
        self.last_conversation_turns = []
        self.time_since_last_interaction_hours = 0.0
        self.pending_re_greeting = None
        self.high_impact_nodes_this_interaction = {}
        self.emotional_core = None
        self._consolidation_counter = 0
        # --- NEW: Track current interaction segment for priming ---
        self.current_conversational_segment_uuids = [] # Stores UUIDs of recent turns in current segment

        # --- NEW: Workspace File Paths & Settings ---
        self.workspace_dir_name = self.config.get('workspace_dir', 'Workspace') # Get name from config
        self.workspace_path = os.path.join(self.data_dir, self.workspace_dir_name) # Full path to workspace
        os.makedirs(self.workspace_path, exist_ok=True) # Ensure workspace dir exists

        self.calendar_filename = self.config.get('calendar_file', 'calendar.jsonl') # Relative to workspace
        self.calendar_filepath = os.path.join(self.workspace_path, self.calendar_filename)

        self.workspace_index_filename = self.config.get('workspace_index_file', '_workspace_index.jsonl') # Relative to workspace
        self.workspace_index_filepath = os.path.join(self.workspace_path, self.workspace_index_filename)

        self.workspace_archive_dir_name = self.config.get('workspace_archive_dir', '_archive') # Relative to workspace
        self.workspace_archive_path = os.path.join(self.workspace_path, self.workspace_archive_dir_name)
        # os.makedirs(self.workspace_archive_path, exist_ok=True) # Archive dir created by archive_file if needed

        self.protected_workspace_files = self.config.get('protected_workspace_files', [])
        self.protected_workspace_prefixes = self.config.get('protected_workspace_prefixes', [])

        self.workspace_persona_prefs = self.config.get('workspace_persona_settings', {}).get('default_preferences', {})
        # TODO (later): Add logic to load persona-specific overrides if full persona configs are implemented.
        # For now, default_preferences will be used.
        logger.info(f"Loaded workspace persona preferences: {self.workspace_persona_prefs}")

        os.makedirs(self.data_dir, exist_ok=True)
        embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        tokenizer_name = self.config.get('tokenizer_name')
        if not tokenizer_name:
            logger.error("Tokenizer name/path ('tokenizer_name') not found in config. Cannot load tokenizer.")
        else:
            logger.info(f"Using tokenizer path from config: {tokenizer_name}")

        spacy_model_name = self.config.get('spacy_model_name', 'en_core_web_sm')
        features_cfg = self.config.get('features', {})
        rich_assoc_enabled = features_cfg.get('enable_rich_associations', False)
        if rich_assoc_enabled:
            try:
                logger.info(f"Checking/Loading spaCy model: {spacy_model_name}")
                if not spacy.util.is_package(spacy_model_name):
                    logger.warning(f"spaCy model '{spacy_model_name}' not found. Attempting download...")
                    command = [sys.executable, "-m", "spacy", "download", spacy_model_name]
                    try:
                        result = subprocess.run(command, check=True, capture_output=True, text=True)
                        logger.info(f"Successfully downloaded spaCy model '{spacy_model_name}'.\nOutput:\n{result.stdout}")
                        self.nlp = True
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to download spaCy model '{spacy_model_name}'. RC: {e.returncode}\nErr:{e.stderr}\nOut:{e.stdout}")
                        self.nlp = False
                    except FileNotFoundError:
                        logger.error(f"Could not run spacy download command. Is '{sys.executable}' correct and spacy installed?")
                        self.nlp = False
                    except Exception as download_e:
                        logger.error(f"An unexpected error occurred during spacy model download: {download_e}", exc_info=True)
                        self.nlp = False
                else:
                    logger.info(f"spaCy model '{spacy_model_name}' already installed.")
                    self.nlp = True

                if self.nlp is True:
                    try:
                        self.nlp = spacy.load(spacy_model_name)
                        logger.info(f"spaCy model '{spacy_model_name}' loaded successfully.")
                    except OSError as e:
                        logger.error(f"Could not load spaCy model '{spacy_model_name}' even after download check/attempt. {e}")
                        self.nlp = None
                    except Exception as e:
                        logger.error(f"An unexpected error occurred loading the spaCy model '{spacy_model_name}': {e}", exc_info=True)
                        self.nlp = None
                else:
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

        try:
            logger.info(f"Loading embed model: {embedding_model_name}")
            self.embedder = SentenceTransformer(embedding_model_name, trust_remote_code=True)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Embed model loaded. Dim: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed loading embed model: {e}", exc_info=True)
            self.embedder = None
            self.embedding_dim = 0

        if tokenizer_name:
            try:
                logger.info(f"Loading tokenizer from: {tokenizer_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                logger.info("Tokenizer loaded.")
            except Exception as e:
                logger.error(f"Failed loading tokenizer from '{tokenizer_name}': {e}", exc_info=True)
                self.tokenizer = None
        else:
            self.tokenizer = None

        self._load_memory()

        try:
            if self.config.get('features', {}).get('enable_emotional_core', False):
                logger.info("Instantiating EmotionalCore...")
                self.emotional_core = EmotionalCore(self, self.config)
                if not self.emotional_core.is_enabled:
                    logger.warning("EmotionalCore instantiated but is disabled internally.")
                    self.emotional_core = None
                else:
                    logger.info("EmotionalCore instantiated successfully.")
            else:
                logger.info("EmotionalCore feature is disabled in main config.")
                self.emotional_core = None
        except Exception as e:
            logger.error(f"Failed to instantiate EmotionalCore: {e}", exc_info=True)
            self.emotional_core = None

        self._load_last_conversation()
        self._calculate_time_since_last_interaction()
        self._check_and_generate_re_greeting()

        if not self.last_added_node_uuid and self.graph.number_of_nodes() > 0:
            try:
                latest_node = self._find_latest_node_uuid()
                if latest_node:
                    self.last_added_node_uuid = latest_node
                    logger.info(f"Set last_added_node_uuid from loaded graph to: {self.last_added_node_uuid[:8]}")
                else:
                    logger.warning("Could not determine last added node from loaded graph.")
            except Exception as e:
                logger.error(f"Error finding latest node during init: {e}", exc_info=True)

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
                },
                "emotional_core_config": self.config.get('emotional_core', {}), # Log EmoCore config
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
        Accepts optional emotion values (valence/arousal) which can be derived from EmotionalCore.
        """
        # ... (logging, basic checks, timestamp generation remain same) ...
        node_uuid = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        # ...

        # --- Get config values safely ---
        features_cfg = self.config.get('features', {})
        saliency_enabled = features_cfg.get('enable_saliency', False)
        emotion_cfg = self.config.get('emotion_analysis', {})
        # Use provided emotions OR fall back to defaults from config if None is passed
        final_valence = emotion_valence if emotion_valence is not None else emotion_cfg.get('default_valence', 0.0)
        final_arousal = emotion_arousal if emotion_arousal is not None else emotion_cfg.get('default_arousal', 0.1)
        logger.debug(f"Node {node_uuid[:8]} Emotion (V/A): ({final_valence:.2f}, {final_arousal:.2f}) | Input: (V={emotion_valence}, A={emotion_arousal})") # Log source

        saliency_cfg = self.config.get('saliency', {})
        initial_scores = saliency_cfg.get('initial_scores', {})
        # --- NEW: Use a dedicated influence factor for EmoCore-derived arousal ---
        emotion_core_arousal_influence = saliency_cfg.get('emotion_core_arousal_saliency_influence', 0.2) # Example new config value
        default_arousal_influence = saliency_cfg.get('default_arousal_saliency_influence', 0.05) # Influence if using default arousal

        importance_keywords = saliency_cfg.get('importance_keywords', [])
        importance_boost = saliency_cfg.get('importance_saliency_boost', 0.0)
        flag_important_as_core = saliency_cfg.get('flag_important_as_core', False)

        # --- Calculate Initial Saliency ---
        initial_saliency = 0.0 # Default
        is_important_keyword_match = False

        if saliency_enabled:
            base_saliency = initial_scores.get('intention' if node_type == 'intention' else node_type, initial_scores.get('default', 0.5))

            # --- MODIFIED: Apply arousal influence based on whether it was provided (likely from EmoCore) ---
            arousal_influence_to_apply = 0.0
            if emotion_arousal is not None: # Check if specific arousal was PASSED IN
                arousal_influence_to_apply = final_arousal * emotion_core_arousal_influence
                logger.debug(f" Using EmoCore-derived arousal influence: {arousal_influence_to_apply:.3f}")
            else: # Use default arousal influence
                arousal_influence_to_apply = final_arousal * default_arousal_influence
                logger.debug(f" Using default arousal influence: {arousal_influence_to_apply:.3f}")

            initial_saliency = base_saliency + arousal_influence_to_apply
            # --- END MODIFICATION ---

            # --- Check for Importance Keywords (remains same) ---
            if importance_keywords and importance_boost > 0:
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in importance_keywords):
                    is_important_keyword_match = True
                    initial_saliency += importance_boost
                    logger.info(f"Importance keyword match in {node_uuid[:8]}. Boosting initial saliency by {importance_boost}.")

            initial_saliency = max(0.0, min(1.0, initial_saliency)) # Clamp 0-1
            logger.debug(f"Calculated initial saliency for {node_uuid[:8]} ({node_type}): {initial_saliency:.3f}")
        else:
            logger.debug(f"Saliency calculation disabled for {node_uuid[:8]}.")

        # --- Get embedding (remains same) ---
        embedding = self._get_embedding(text)
        if embedding is None: # Check return value
            logger.error(f"Failed to get embedding for node {node_uuid}. Node not added.")
            return None

        # --- Add to graph (uses final_valence, final_arousal, initial_saliency) ---
        try:
            self.graph.add_node(
                node_uuid,
                # ... (uuid, text, speaker, timestamp, node_type remain same) ...
                uuid=node_uuid, text=text, speaker=speaker, timestamp=timestamp, node_type=node_type,
                memory_strength=self.config.get('memory_strength', {}).get('initial_value', 1.0),
                access_count=0,
                emotion_valence=final_valence, # Use the emotion value passed in or default
                emotion_arousal=final_arousal, # Use the emotion value passed in or default
                saliency_score=initial_saliency, # Use calculated saliency
                base_strength=float(base_strength),
                activation_level=0.0,
                last_accessed_ts=time.time(),
                decay_resistance_factor=self.config.get('forgetting', {}).get('decay_resistance', {}).get(node_type, 1.0),
                user_feedback_score=0,
                is_core_memory=(is_important_keyword_match and flag_important_as_core)
            )
            # ... (rest of graph add logging, FAISS add, temporal linking, conversation update) ...

        except Exception as e:
            logger.error(f"Failed adding node {node_uuid} to graph: {e}", exc_info=True) # Added exc_info
            return None

        # --- Add embedding to dictionary ---
        self.embeddings[node_uuid] = embedding

        # --- Add to FAISS ---
        try:
            # ... (FAISS logic remains the same, including initialization check) ...
            if self.index is None:
                if hasattr(self, 'embedding_dim') and self.embedding_dim > 0:
                    logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                else:
                    logger.error("Cannot initialize FAISS index: embedding_dim not set.")
                    self._rollback_add(node_uuid)
                    return None

            self.index.add(np.array([embedding], dtype='float32'))
            new_faiss_id = self.index.ntotal - 1
            self.faiss_id_to_uuid[new_faiss_id] = node_uuid
            self.uuid_to_faiss_id[node_uuid] = new_faiss_id
            logger.debug(f"Embedding {node_uuid[:8]} added to FAISS ID {new_faiss_id}.")

        except Exception as e:
            logger.error(f"Failed adding embed {node_uuid} to FAISS: {e}", exc_info=True) # Added exc_info
            self._rollback_add(node_uuid)
            return None

        # --- Link temporally (remains same) ---
        if self.last_added_node_uuid and self.last_added_node_uuid in self.graph:
            try:
                self.graph.add_edge(
                    self.last_added_node_uuid, node_uuid,
                    type='TEMPORAL', base_strength=0.8, last_traversed_ts=time.time() # Use current time
                )
                logger.debug(f"Added T-edge {self.last_added_node_uuid[:8]}->{node_uuid[:8]}.")
            except Exception as e:
                logger.error(f"Failed adding T-edge: {e}", exc_info=True) # Added exc_info

        self.last_added_node_uuid = node_uuid

        # --- Update Last Conversation Turns (remains same) ---
        if node_type == 'turn':
            try:
                turn_data = {"speaker": speaker, "text": text, "timestamp": timestamp, "uuid": node_uuid}
                self.last_conversation_turns.append(turn_data)
                max_turns_to_keep = 6
                if len(self.last_conversation_turns) > max_turns_to_keep:
                    self.last_conversation_turns = self.last_conversation_turns[-max_turns_to_keep:]
                logger.debug(f"Updated last_conversation_turns. Current count: {len(self.last_conversation_turns)}")
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

    def _search_similar_nodes(self, query_text: str, k: int = None, node_type_filter: str = None, query_type: str = 'other',
                              apply_recency_bias: bool = False, recency_bias_factor: float = 0.1, recency_window_hours: float = 1.0) -> list[tuple[str, float]]:
        """
        Searches FAISS for nodes similar to query_text.
        Optionally filters by node_type_filter.
        Optionally biases search based on query_type ('episodic', 'semantic', 'other').
        Optionally applies a recency bias to scores (reduces distance for recent items).
        """
        act_cfg = self.config.get('activation', {})
        if k is None: k = act_cfg.get('max_initial_nodes', 7) # Default k from config

        if not query_text or self.index is None or self.index.ntotal == 0:
            return []
        try:
            q_embed = self._get_embedding(query_text)
            if q_embed is None or not hasattr(q_embed, 'shape') or q_embed.shape != (self.embedding_dim,): # Added hasattr check
                logger.error(f"Invalid query embedding for '{query_text[:30]}...' Shape: {getattr(q_embed, 'shape', 'N/A')}")
                return []

            q_embed_np = np.array([q_embed], dtype='float32')

            search_multiplier = 3 # Default search multiplier
            if query_type == 'episodic': search_multiplier = 5
            elif query_type == 'semantic': search_multiplier = 4

            search_k = k * search_multiplier # Search for more items initially
            actual_k_to_search = min(search_k, self.index.ntotal) # Don't search more than available

            if actual_k_to_search == 0: return [] # No items in index

            distances_from_faiss, indices_from_faiss = self.index.search(q_embed_np, actual_k_to_search)

            # Store tuples of (uuid, original_faiss_distance, final_adjusted_distance)
            results_with_distances = []

            logger.debug(f"FAISS Search: Query='{query_text[:30]}...', TargetK={k}, SearchedK={actual_k_to_search}, Filter='{node_type_filter}', QType='{query_type}', RecencyBias={apply_recency_bias}")

            if len(indices_from_faiss) > 0:
                for i_rank, faiss_id_val in enumerate(indices_from_faiss[0]):
                    faiss_id_int = int(faiss_id_val)
                    original_faiss_distance = float(distances_from_faiss[0][i_rank])

                    if faiss_id_int == -1: # Invalid FAISS ID
                        logger.debug(f"  Rank {i_rank+1}: Invalid FAISS ID -1 encountered. Skipping.")
                        continue

                    node_uuid_val = self.faiss_id_to_uuid.get(faiss_id_int)
                    if not node_uuid_val or node_uuid_val not in self.graph:
                        logger.debug(f"  Rank {i_rank+1}: FAISS ID {faiss_id_int} -> UUID {str(node_uuid_val)[:8]} not in graph/map. Skipping.")
                        continue

                    node_data_val = self.graph.nodes[node_uuid_val]
                    node_type_val = node_data_val.get('node_type')

                    # Apply explicit node type filter
                    if node_type_filter and node_type_val != node_type_filter:
                        logger.debug(f"  Rank {i_rank+1}: UUID={node_uuid_val[:8]} (Type {node_type_val}) FILTERED OUT (requested {node_type_filter}).")
                        continue

                    # Start with original distance, then apply penalties/boosts
                    current_adjusted_distance = original_faiss_distance
                    log_adjustments_str = ""

                    # Apply query type penalties (increases distance)
                    if query_type == 'episodic' and node_type_val != 'turn':
                        penalty_factor = self.config.get('retrieval_penalties',{}).get('episodic_non_turn_penalty', 1.1)
                        current_adjusted_distance *= penalty_factor
                        log_adjustments_str += f" EpisodicBias(x{penalty_factor:.1f})"
                    elif query_type == 'semantic' and node_type_val not in ['summary', 'concept']:
                        penalty_factor = self.config.get('retrieval_penalties',{}).get('semantic_non_summary_concept_penalty', 1.2)
                        current_adjusted_distance *= penalty_factor
                        log_adjustments_str += f" SemanticBias(x{penalty_factor:.1f})"

                    # Apply Recency Bias (reduces distance for recent items)
                    if apply_recency_bias and recency_bias_factor > 0:
                        node_ts_str_val = node_data_val.get('timestamp')
                        if node_ts_str_val:
                            try:
                                node_dt_val = datetime.fromisoformat(node_ts_str_val.replace('Z', '+00:00'))
                                age_hours_val = (datetime.now(timezone.utc) - node_dt_val).total_seconds() / 3600.0
                                if age_hours_val <= recency_window_hours and age_hours_val >= 0: # Ensure age is positive
                                    # Bias effect is stronger for more recent items within the window
                                    bias_strength_effect = (1.0 - (age_hours_val / recency_window_hours)) * recency_bias_factor
                                    current_adjusted_distance *= (1.0 - bias_strength_effect) # Lower distance is better
                                    log_adjustments_str += f" RecencyBoost(-{bias_strength_effect*100:.1f}%)"
                            except Exception as e_ts_val:
                                logger.warning(f"Error parsing timestamp for recency bias on {node_uuid_val[:8]}: {e_ts_val}")

                    logger.debug(f"  Rank {i_rank+1}: UUID={node_uuid_val[:8]} (Type {node_type_val}), OrigD:{original_faiss_distance:.3f}{log_adjustments_str} -> AdjD:{current_adjusted_distance:.3f}")
                    results_with_distances.append((node_uuid_val, original_faiss_distance, current_adjusted_distance))

            # Sort all collected candidates by their final adjusted distance (ascending)
            results_with_distances.sort(key=lambda item_val: item_val[2])

            # Select the top K results based on the target K
            final_top_k_results = []
            for uuid_res, _, adj_dist_res in results_with_distances[:k]: # Take top k from sorted list
                # Convert adjusted distance to a pseudo-similarity score (0-1, higher is better)
                # This is a heuristic. Max L2 distance for normalized embeddings is 2.0.
                # Closer to 0 distance = higher similarity.
                similarity_heuristic = max(0.0, 1.0 - (adj_dist_res / 2.0))
                final_top_k_results.append((uuid_res, similarity_heuristic))

            logger.info(f"Search for '{strip_emojis(query_text[:30])}...': Found {len(final_top_k_results)} relevant nodes (target_k={k}).")
            if final_top_k_results:
                logger.debug(f" Final top {k} (UUID, SimilarityScore): {[(u[:8], f'{s:.3f}') for u,s in final_top_k_results]}")
            return final_top_k_results

        except Exception as e_search:
            logger.error(f"FAISS search error for query '{query_text[:30]}...': {e_search}", exc_info=True)
            return []



    def retrieve_memory_chain(self, initial_node_uuids: list[str],
                              recent_concept_uuids: list[str] | None = None,
                              current_mood: tuple[float, float] | None = None) -> tuple[list[dict], tuple[float, float]]:
        act_cfg = self.config.get('activation', {})
        features_cfg = self.config.get('features', {})
        saliency_cfg = self.config.get('saliency', {})

        initial_activation_config = act_cfg.get('initial', 1.0)
        spreading_depth = act_cfg.get('spreading_depth', 3)
        activation_threshold = act_cfg.get('threshold', 0.1)
        prop_base = act_cfg.get('propagation_factor_base', 0.65)
        prop_factors = act_cfg.get('propagation_factors', {})

        prop_temporal_fwd = prop_factors.get('TEMPORAL_fwd', 1.0)
        prop_temporal_bwd = prop_factors.get('TEMPORAL_bwd', 0.8)
        prop_summary_fwd = prop_factors.get('SUMMARY_OF_fwd', 1.1)
        prop_summary_bwd = prop_factors.get('SUMMARY_OF_bwd', 0.4)
        prop_concept_fwd = prop_factors.get('MENTIONS_CONCEPT_fwd', 1.0)
        prop_concept_bwd = prop_factors.get('MENTIONS_CONCEPT_bwd', 0.9)
        prop_assoc = prop_factors.get('ASSOCIATIVE', 0.8)
        prop_hier_fwd = prop_factors.get('HIERARCHICAL_fwd', 1.1)
        prop_hier_bwd = prop_factors.get('HIERARCHICAL_bwd', 0.5)
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
        prop_spacy = prop_factors.get('SPACY_REL', 0.7)
        prop_unknown = prop_factors.get('UNKNOWN', 0.5)

        guaranteed_saliency_threshold = act_cfg.get('guaranteed_saliency_threshold', 0.88)
        intention_boost_factor = act_cfg.get('intention_boost_factor', 1.2)
        always_retrieve_core = act_cfg.get('always_retrieve_core', True)
        context_focus_boost = act_cfg.get('context_focus_boost', 0.0)

        conversational_segment_priming_boost = act_cfg.get('conversational_priming_boost_factor', 1.5) # Renamed
        segment_uuids_for_priming = set(self.current_conversational_segment_uuids) # Use current_conversational_segment_uuids
        logger.debug(f"Priming conversational segment UUIDs ({len(segment_uuids_for_priming)}): {list(segment_uuids_for_priming)[:5]}")


        saliency_enabled = features_cfg.get('enable_saliency', False)
        activation_influence_on_spread = saliency_cfg.get('activation_influence', 0.0) if saliency_enabled else 0.0

        recent_concept_uuids_set = set(recent_concept_uuids) if recent_concept_uuids else set()

        drive_cfg = self.config.get('subconscious_drives', {})
        mood_influence_cfg = drive_cfg.get('mood_influence', {})
        drives_enabled_for_mood = drive_cfg.get('enabled', False) and mood_influence_cfg

        base_mood_for_retrieval = current_mood if current_mood else (0.0, 0.1)
        effective_mood = base_mood_for_retrieval

        if drives_enabled_for_mood:
            current_st_drives = self.drive_state.get("short_term", {})
            valence_adjustment_from_drives = 0.0
            arousal_adjustment_from_drives = 0.0
            valence_factors = mood_influence_cfg.get('valence_factors', {})
            arousal_factors = mood_influence_cfg.get('arousal_factors', {})

            for drive_name, st_level in current_st_drives.items():
                deviation = st_level # ST is already deviation from 0 (LT-influenced baseline)
                valence_adjustment_from_drives += deviation * valence_factors.get(drive_name, 0.0)
                arousal_adjustment_from_drives += deviation * arousal_factors.get(drive_name, 0.0)

            max_adj_val = mood_influence_cfg.get('max_mood_adjustment', 0.3)
            valence_adjustment_from_drives = max(-max_adj_val, min(max_adj_val, valence_adjustment_from_drives))
            arousal_adjustment_from_drives = max(-max_adj_val, min(max_adj_val, arousal_adjustment_from_drives))

            adj_valence_val = max(-1.0, min(1.0, base_mood_for_retrieval[0] + valence_adjustment_from_drives))
            adj_arousal_val = max(0.0, min(1.0, base_mood_for_retrieval[1] + arousal_adjustment_from_drives))
            effective_mood = (adj_valence_val, adj_arousal_val)
            if abs(valence_adjustment_from_drives) > 1e-4 or abs(arousal_adjustment_from_drives) > 1e-4:
                logger.info(f"Mood for retrieval biased by ST drives: Base=({base_mood_for_retrieval[0]:.2f},{base_mood_for_retrieval[1]:.2f}) -> DriveAdjusted=({effective_mood[0]:.2f},{effective_mood[1]:.2f})")
                log_tuning_event("RETRIEVAL_MOOD_DRIVE_ADJUSTMENT", {
                    "personality": self.personality, "base_mood": base_mood_for_retrieval,
                    "valence_adj_drives": valence_adjustment_from_drives, "arousal_adj_drives": arousal_adjustment_from_drives,
                    "drive_adjusted_mood": effective_mood, "st_drive_state": current_st_drives,
                })

        if self.emotional_core and self.emotional_core.is_enabled:
            valence_hint_val = self.emotional_core.derived_mood_hints.get("valence", 0.0)
            arousal_hint_val = self.emotional_core.derived_mood_hints.get("arousal", 0.0)
            emocore_cfg_val = self.emotional_core.config
            valence_factor_val = emocore_cfg_val.get("mood_valence_factor", 0.3)
            arousal_factor_val = emocore_cfg_val.get("mood_arousal_factor", 0.2)

            if abs(valence_hint_val) > 1e-4 or abs(arousal_hint_val) > 1e-4:
                current_v_val, current_a_val = effective_mood
                new_v_val = current_v_val + (valence_hint_val * valence_factor_val)
                new_a_val = current_a_val + (arousal_hint_val * arousal_factor_val)
                effective_mood = (max(-1.0, min(1.0, new_v_val)), max(0.0, min(1.0, new_a_val)))
                logger.info(f"Mood for retrieval after EmotionalCore hints: ({effective_mood[0]:.2f}, {effective_mood[1]:.2f})")
                log_tuning_event("RETRIEVAL_MOOD_EMOCORE_ADJUSTMENT", {
                    "personality": self.personality, "mood_after_drives": (current_v_val, current_a_val),
                    "valence_hint": valence_hint_val, "arousal_hint": arousal_hint_val,
                    "valence_factor": valence_factor_val, "arousal_factor": arousal_factor_val,
                    "mood_after_emocore": effective_mood,
                })

        emo_ctx_cfg = act_cfg.get('emotional_context', {})
        emo_ctx_enabled = emo_ctx_cfg.get('enable', False) and effective_mood is not None
        emo_max_dist_val = emo_ctx_cfg.get('max_distance', 1.414)
        emo_boost_val = emo_ctx_cfg.get('boost_factor', 0.0)
        emo_penalty_val = emo_ctx_cfg.get('penalty_factor', 0.0)

        relevant_nodes_list_final = [] # Initialize here

        logger.info(f"Starting retrieval. Initial nodes: {len(initial_node_uuids)}. SalInf: {activation_influence_on_spread:.2f}, GuarSal>=: {guaranteed_saliency_threshold}, FocusBoost: {context_focus_boost}, RecentConcepts: {len(recent_concept_uuids_set)}, EmoCtx: {emo_ctx_enabled}, EffectiveMood: {effective_mood}")
        log_tuning_event("RETRIEVAL_START", {
            "personality": self.personality, "initial_node_uuids": initial_node_uuids,
            "recent_concept_uuids": list(recent_concept_uuids_set), "current_mood": effective_mood,
            "saliency_influence": activation_influence_on_spread, "guaranteed_saliency_threshold": guaranteed_saliency_threshold,
            "context_focus_boost": context_focus_boost, "emotional_context_enabled": emo_ctx_enabled,
        })

        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph empty. Returning empty list and effective_mood.")
            return [], effective_mood

        activation_levels_dict = defaultdict(float)
        current_time_sec = time.time()
        valid_initial_nodes_set = set()

        for uuid_init in initial_node_uuids:
            if uuid_init in self.graph:
                node_data_init = self.graph.nodes[uuid_init]
                initial_strength_val = node_data_init.get('memory_strength', 1.0)
                current_base_initial_act = initial_activation_config * initial_strength_val

                context_boost_mult = 1.0
                if context_focus_boost > 0 and recent_concept_uuids_set:
                    is_recent_concept_val = uuid_init in recent_concept_uuids_set
                    mentions_recent_concept_val = False
                    if not is_recent_concept_val:
                        try:
                            for succ_uuid_val in self.graph.successors(uuid_init):
                                if succ_uuid_val in recent_concept_uuids_set and self.graph.get_edge_data(uuid_init, succ_uuid_val, {}).get('type') == 'MENTIONS_CONCEPT':
                                    mentions_recent_concept_val = True; break
                        except Exception as e_ctx_init: logger.warning(f"Error checking concept links for focus boost on {uuid_init[:8]}: {e_ctx_init}")
                    if is_recent_concept_val or mentions_recent_concept_val: context_boost_mult = 1.0 + context_focus_boost

                current_base_initial_act *= context_boost_mult

                segment_priming_mult = 1.0
                if uuid_init in segment_uuids_for_priming: # Check against the set from self.current_conversational_segment_uuids
                    segment_priming_mult = conversational_segment_priming_boost

                current_base_initial_act *= segment_priming_mult

                intention_boost_mult = 1.0
                if node_data_init.get('node_type') == 'intention' and intention_boost_factor > 1.0:
                    intention_boost_mult = intention_boost_factor

                final_initial_act = current_base_initial_act * intention_boost_mult

                activation_levels_dict[uuid_init] = final_initial_act
                node_data_init['last_accessed_ts'] = current_time_sec
                valid_initial_nodes_set.add(uuid_init)
                logger.debug(f"Initialized node {uuid_init[:8]} Act:{final_initial_act:.3f} (Str:{initial_strength_val:.2f}, CtxB:{context_boost_mult:.2f}, SegP:{segment_priming_mult:.2f}, IntB:{intention_boost_mult:.2f})")
            else:
                logger.warning(f"Initial node {uuid_init} not in graph.")

        if not activation_levels_dict:
            logger.warning("No valid initial nodes found or activated in graph. Returning empty list and effective_mood.")
            return [], effective_mood

        logger.debug(f"Valid initial nodes for spreading: {len(valid_initial_nodes_set)}")
        active_nodes_set = set(activation_levels_dict.keys())

        for depth_val in range(spreading_depth):
            logger.debug(f"--- Spreading Step {depth_val + 1} ---")
            newly_activated_this_step = defaultdict(float)
            nodes_to_process_in_step = list(active_nodes_set) # Process nodes active at START of this step
            logger.debug(f" Processing {len(nodes_to_process_in_step)} nodes in step {depth_val + 1}.")

            for source_uuid_spread in nodes_to_process_in_step:
                source_data_spread = self.graph.nodes.get(source_uuid_spread)
                if not source_data_spread: continue
                source_act_spread = activation_levels_dict.get(source_uuid_spread, 0.0) # Get current activation
                raw_saliency_spread = source_data_spread.get('saliency_score', 0.0)
                source_saliency_spread = raw_saliency_spread if isinstance(raw_saliency_spread, (int, float)) else 0.0
                if source_act_spread < 1e-6: continue # Skip if source activation is negligible

                neighbors_spread = set(self.graph.successors(source_uuid_spread)) | set(self.graph.predecessors(source_uuid_spread))

                for neighbor_uuid_spread in neighbors_spread:
                    if neighbor_uuid_spread == source_uuid_spread: continue
                    neighbor_data_spread = self.graph.nodes.get(neighbor_uuid_spread)
                    if not neighbor_data_spread: continue

                    is_forward_edge = self.graph.has_edge(source_uuid_spread, neighbor_uuid_spread)
                    edge_data_spread = self.graph.get_edge_data(source_uuid_spread, neighbor_uuid_spread) if is_forward_edge else self.graph.get_edge_data(neighbor_uuid_spread, source_uuid_spread)
                    if not edge_data_spread: continue

                    edge_type_spread = edge_data_spread.get('type', 'UNKNOWN')
                    base_type_factor_val = prop_unknown
                    if edge_type_spread == 'TEMPORAL': base_type_factor_val = prop_temporal_fwd if is_forward_edge else prop_temporal_bwd
                    elif edge_type_spread == 'SUMMARY_OF': base_type_factor_val = prop_summary_fwd if is_forward_edge else prop_summary_bwd
                    elif edge_type_spread == 'MENTIONS_CONCEPT': base_type_factor_val = prop_concept_fwd if is_forward_edge else prop_concept_bwd
                    elif edge_type_spread == 'ASSOCIATIVE': base_type_factor_val = prop_assoc
                    elif edge_type_spread == 'HIERARCHICAL': base_type_factor_val = prop_hier_fwd if is_forward_edge else prop_hier_bwd
                    elif edge_type_spread == 'CAUSES': base_type_factor_val = prop_causes
                    elif edge_type_spread == 'PART_OF': base_type_factor_val = prop_part_of
                    elif edge_type_spread == 'HAS_PROPERTY': base_type_factor_val = prop_has_prop
                    elif edge_type_spread == 'ENABLES': base_type_factor_val = prop_enables
                    elif edge_type_spread == 'PREVENTS': base_type_factor_val = prop_prevents
                    elif edge_type_spread == 'CONTRADICTS': base_type_factor_val = prop_contradicts
                    elif edge_type_spread == 'SUPPORTS': base_type_factor_val = prop_supports
                    elif edge_type_spread == 'EXAMPLE_OF': base_type_factor_val = prop_example_of
                    elif edge_type_spread == 'MEASURES': base_type_factor_val = prop_measures
                    elif edge_type_spread == 'LOCATION_OF': base_type_factor_val = prop_location_of
                    elif edge_type_spread == 'ANALOGY': base_type_factor_val = prop_analogy
                    elif edge_type_spread == 'INFERRED_RELATED_TO': base_type_factor_val = prop_inferred
                    elif edge_type_spread.startswith('SPACY_'): base_type_factor_val = prop_spacy

                    drive_weight_mult = 1.0 # Placeholder for potential drive influence on edge traversal
                    type_factor_final = base_type_factor_val * drive_weight_mult
                    dyn_strength_val = self._calculate_dynamic_edge_strength(edge_data_spread, current_time_sec)
                    saliency_boost_val = 1.0 + (source_saliency_spread * activation_influence_on_spread) if saliency_enabled else 1.0
                    neighbor_strength_val = neighbor_data_spread.get('memory_strength', 1.0)
                    base_act_pass_val = source_act_spread * dyn_strength_val * prop_base * type_factor_final * saliency_boost_val * neighbor_strength_val

                    emo_adjustment_val = 0.0
                    if emo_ctx_enabled and base_act_pass_val > 1e-6:
                        try:
                            default_v_val = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                            default_a_val = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                            neighbor_v_val = neighbor_data_spread.get('emotion_valence', default_v_val)
                            neighbor_a_val = neighbor_data_spread.get('emotion_arousal', default_a_val)
                            mood_v_val, mood_a_val = effective_mood
                            dist_sq_val = (neighbor_v_val - mood_v_val)**2 + (neighbor_a_val - mood_a_val)**2
                            emo_dist_val = math.sqrt(dist_sq_val)
                            norm_dist_val = min(1.0, emo_dist_val / emo_max_dist_val) if emo_max_dist_val > 0 else 0.0
                            emo_adjustment_val = emo_boost_val * (1.0 - norm_dist_val) - emo_penalty_val * norm_dist_val
                        except Exception as e_emo_ctx:
                            logger.warning(f"Error calculating emotional context bias for {neighbor_uuid_spread[:8]}: {e_emo_ctx}")
                            emo_adjustment_val = 0.0

                    act_pass_final = max(0.0, base_act_pass_val + emo_adjustment_val)

                    if act_pass_final > 1e-6:
                        newly_activated_this_step[neighbor_uuid_spread] += act_pass_final
                        logger.debug(f"  Spread: {source_uuid_spread[:8]}(A:{source_act_spread:.2f},S:{source_saliency_spread:.2f}) -> {neighbor_uuid_spread[:8]}(Str:{neighbor_strength_val:.2f}) ({edge_type_spread},{'F' if is_forward_edge else 'B'}), DStr:{dyn_strength_val:.2f}, TypeF:{type_factor_final:.2f}, SalB:{saliency_boost_val:.2f}, EmoAdj:{emo_adjustment_val:.3f} => Pass:{act_pass_final:.3f}")
                        edge_key_val = (source_uuid_spread, neighbor_uuid_spread) if is_forward_edge else (neighbor_uuid_spread, source_uuid_spread)
                        if edge_key_val in self.graph.edges:
                            self.graph.edges[edge_key_val]['last_traversed_ts'] = current_time_sec

            # Update activation levels for ALL nodes involved in this step (decay old, add new)
            nodes_to_update_in_step = set(activation_levels_dict.keys()) | set(newly_activated_this_step.keys())
            next_active_nodes_set = set()

            for uuid_update in nodes_to_update_in_step:
                node_data_update = self.graph.nodes.get(uuid_update)
                if not node_data_update: continue

                current_activation_before_decay = activation_levels_dict.get(uuid_update, 0.0)
                decayed_activation = current_activation_before_decay
                if current_activation_before_decay > 0: # Apply decay only if it was active
                    decay_mult_val = self._calculate_node_decay(node_data_update, current_time_sec)
                    decayed_activation *= decay_mult_val

                # Add newly spread activation
                final_activation_for_node = decayed_activation + newly_activated_this_step.get(uuid_update, 0.0)

                if final_activation_for_node > 1e-6:
                    activation_levels_dict[uuid_update] = final_activation_for_node
                    self.graph.nodes[uuid_update]['last_accessed_ts'] = current_time_sec # Update access time
                    next_active_nodes_set.add(uuid_update)
                elif uuid_update in activation_levels_dict: # Remove if activation dropped to zero
                    del activation_levels_dict[uuid_update]

            active_nodes_set = next_active_nodes_set # Update active set for next iteration
            logger.debug(f" Step {depth_val+1} finished. Active Nodes: {len(active_nodes_set)}. Max Activation: {max(activation_levels_dict.values()) if activation_levels_dict else 0:.3f}")
            if not active_nodes_set: break

        interference_cfg = act_cfg.get('interference', {})
        interference_applied_count_val = 0
        penalized_nodes_set = set()
        if interference_cfg.get('enable', False) and self.index and self.index.ntotal > 0:
            logger.info("--- Applying Interference Simulation ---")
            check_threshold_interf = interference_cfg.get('check_threshold', 0.15)
            sim_threshold_interf = interference_cfg.get('similarity_threshold', 0.25)
            penalty_factor_interf = interference_cfg.get('penalty_factor', 0.90)
            k_neighbors_interf = interference_cfg.get('max_neighbors_check', 5)

            nodes_to_check_interf = sorted(activation_levels_dict.items(), key=lambda item_interf: item_interf[1], reverse=True)

            for source_uuid_interf, source_activation_interf in nodes_to_check_interf:
                if source_uuid_interf in penalized_nodes_set: continue
                if source_activation_interf < check_threshold_interf: continue
                source_embedding_interf = self.embeddings.get(source_uuid_interf)
                if source_embedding_interf is None: continue

                try:
                    source_embed_np_interf = np.array([source_embedding_interf], dtype='float32')
                    distances_interf, indices_interf = self.index.search(source_embed_np_interf, k_neighbors_interf + 1)
                    local_cluster_interf = []
                    if len(indices_interf) > 0 and len(indices_interf[0]) > 0:
                        for i_interf, faiss_id_interf in enumerate(indices_interf[0]):
                            neighbor_uuid_interf = self.faiss_id_to_uuid.get(int(faiss_id_interf))
                            if neighbor_uuid_interf is None or neighbor_uuid_interf == source_uuid_interf: continue
                            neighbor_activation_interf = activation_levels_dict.get(neighbor_uuid_interf)
                            distance_interf = distances_interf[0][i_interf]
                            if neighbor_activation_interf is not None and distance_interf <= sim_threshold_interf:
                                local_cluster_interf.append((neighbor_uuid_interf, neighbor_activation_interf, distance_interf))
                    if local_cluster_interf:
                        cluster_with_source_interf = [(source_uuid_interf, source_activation_interf, 0.0)] + local_cluster_interf
                        dominant_uuid_interf, max_act_interf, _ = max(cluster_with_source_interf, key=lambda item_dom: item_dom[1])
                        for neighbor_uuid_pen, neighbor_activation_pen, dist_pen in cluster_with_source_interf:
                            if neighbor_uuid_pen != dominant_uuid_interf and neighbor_uuid_pen not in penalized_nodes_set:
                                original_activation_pen = activation_levels_dict[neighbor_uuid_pen]
                                activation_levels_dict[neighbor_uuid_pen] *= penalty_factor_interf
                                penalized_nodes_set.add(neighbor_uuid_pen)
                                interference_applied_count_val += 1
                                logger.debug(f"  Interference: Dom '{dominant_uuid_interf[:8]}' ({max_act_interf:.3f}) penalized '{neighbor_uuid_pen[:8]}'. Act {original_activation_pen:.3f} -> {activation_levels_dict[neighbor_uuid_pen]:.3f} (Dist: {dist_pen:.3f})")
                except AttributeError: logger.warning("Interference check failed: Faiss index/map not initialized."); break
                except Exception as e_interf: logger.error(f"Error during interference check for {source_uuid_interf[:8]}: {e_interf}", exc_info=True)
            if interference_applied_count_val > 0: logger.info(f"Interference applied to {interference_applied_count_val} node activations.")
            else: logger.info("No interference applied in this retrieval.")
        else: logger.debug("Interference simulation disabled or index unavailable.")
        log_tuning_event("RETRIEVAL_INTERFERENCE_RESULT", {
            "personality": self.personality, "initial_node_uuids": initial_node_uuids,
            "interference_enabled": interference_cfg.get('enable', False),
            "interference_applied_count": interference_applied_count_val, "penalized_node_uuids": list(penalized_nodes_set),
        })

        relevant_nodes_final_dict = {}
        processed_uuids_for_access_count_set = set()

        default_v_val_final = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
        default_a_val_final = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)

        for uuid_sel, final_activation_sel in activation_levels_dict.items():
            if final_activation_sel >= activation_threshold:
                node_data_sel = self.graph.nodes.get(uuid_sel)
                if node_data_sel:
                    if uuid_sel not in processed_uuids_for_access_count_set:
                        node_data_sel['access_count'] = node_data_sel.get('access_count', 0) + 1
                        processed_uuids_for_access_count_set.add(uuid_sel)

                    node_info_sel = node_data_sel.copy()
                    node_info_sel['final_activation'] = final_activation_sel
                    node_info_sel['guaranteed_inclusion'] = False
                    relevant_nodes_final_dict[uuid_sel] = node_info_sel

                    if saliency_enabled:
                        recall_boost = saliency_cfg.get('recall_boost_factor', 0.05)
                        if recall_boost > 0:
                            current_saliency_sel = node_data_sel.get('saliency_score', 0.0)
                            if isinstance(current_saliency_sel, (int, float)):
                                new_saliency_sel = min(1.0, current_saliency_sel + recall_boost)
                                if new_saliency_sel > current_saliency_sel: self.graph.nodes[uuid_sel]['saliency_score'] = new_saliency_sel

                    if emo_ctx_enabled and emo_ctx_cfg.get('reconsolidation_enable', False):
                        recon_thresh = emo_ctx_cfg.get('reconsolidation_threshold', 0.5)
                        recon_factor = emo_ctx_cfg.get('reconsolidation_factor', 0.05)
                        if recon_factor > 0:
                            try:
                                node_v_sel = node_data_sel.get('emotion_valence', default_v_val_final)
                                node_a_sel = node_data_sel.get('emotion_arousal', default_a_val_final)
                                if isinstance(node_v_sel, (int, float)) and isinstance(node_a_sel, (int, float)):
                                    mood_v_sel, mood_a_sel = effective_mood
                                    dist_sq_sel = (node_v_sel - mood_v_sel)**2 + (node_a_sel - mood_a_sel)**2
                                    emo_dist_sel = math.sqrt(dist_sq_sel)
                                    if emo_dist_sel >= recon_thresh:
                                        new_v_sel = node_v_sel + (mood_v_sel - node_v_sel) * recon_factor
                                        new_a_sel = node_a_sel + (mood_a_sel - node_a_sel) * recon_factor
                                        self.graph.nodes[uuid_sel]['emotion_valence'] = max(-1.0, min(1.0, new_v_sel))
                                        self.graph.nodes[uuid_sel]['emotion_arousal'] = max(0.0, min(1.0, new_a_sel))
                            except Exception as e_recon: logger.warning(f"Error during emotional reconsolidation for {uuid_sel[:8]}: {e_recon}")

        logger.info(f"Found {len(relevant_nodes_final_dict)} active nodes above activation threshold ({activation_threshold}).")
        core_added_count_val = 0
        saliency_guaranteed_added_count_val = 0

        for uuid_guar, final_activation_guar in activation_levels_dict.items():
            if uuid_guar not in relevant_nodes_final_dict: # Only consider if not already added by threshold
                node_data_guar = self.graph.nodes.get(uuid_guar)
                if node_data_guar:
                    is_core_guar = node_data_guar.get('is_core_memory', False)
                    current_saliency_guar_raw = node_data_guar.get('saliency_score', 0.0) # Ensure it's a float
                    current_saliency_guar = current_saliency_guar_raw if isinstance(current_saliency_guar_raw, (int, float)) else 0.0

                    should_include_guar = False
                    inclusion_reason_guar = ""

                    if is_core_guar and always_retrieve_core:
                        should_include_guar = True; inclusion_reason_guar = "Core Memory"; core_added_count_val +=1
                    elif current_saliency_guar >= guaranteed_saliency_threshold:
                        should_include_guar = True; inclusion_reason_guar = f"High Saliency ({current_saliency_guar:.3f})"; saliency_guaranteed_added_count_val += 1

                    if should_include_guar:
                        logger.info(f"Guaranteed inclusion for {uuid_guar[:8]} ({inclusion_reason_guar}, Act: {final_activation_guar:.3f})")
                        if uuid_guar not in processed_uuids_for_access_count_set:
                            self.graph.nodes[uuid_guar]['access_count'] = self.graph.nodes[uuid_guar].get('access_count', 0) + 1
                            processed_uuids_for_access_count_set.add(uuid_guar)

                        node_info_guar = node_data_guar.copy()
                        node_info_guar['final_activation'] = final_activation_guar
                        node_info_guar['guaranteed_inclusion'] = inclusion_reason_guar.split(' ')[0].lower()
                        relevant_nodes_final_dict[uuid_guar] = node_info_guar

                        if saliency_enabled:
                            recall_boost_guar = saliency_cfg.get('recall_boost_factor', 0.05)
                            if recall_boost_guar > 0 and isinstance(current_saliency_guar, (int, float)):
                                new_saliency_guar = min(1.0, current_saliency_guar + recall_boost_guar)
                                if new_saliency_guar > current_saliency_guar: self.graph.nodes[uuid_guar]['saliency_score'] = new_saliency_guar

                        if emo_ctx_enabled and emo_ctx_cfg.get('reconsolidation_enable', False):
                            recon_thresh_guar = emo_ctx_cfg.get('reconsolidation_threshold', 0.5)
                            recon_factor_guar = emo_ctx_cfg.get('reconsolidation_factor', 0.05)
                            if recon_factor_guar > 0:
                                try:
                                    node_v_guar = node_data_guar.get('emotion_valence', default_v_val_final)
                                    node_a_guar = node_data_guar.get('emotion_arousal', default_a_val_final)
                                    if isinstance(node_v_guar, (int, float)) and isinstance(node_a_guar, (int, float)):
                                        mood_v_guar, mood_a_guar = effective_mood
                                        dist_sq_guar = (node_v_guar - mood_v_guar)**2 + (node_a_guar - mood_a_guar)**2
                                        emo_dist_guar = math.sqrt(dist_sq_guar)
                                        if emo_dist_guar >= recon_thresh_guar:
                                            new_v_guar = node_v_guar + (mood_v_guar - node_v_guar) * recon_factor_guar
                                            new_a_guar = node_a_guar + (mood_a_guar - node_a_guar) * recon_factor_guar
                                            self.graph.nodes[uuid_guar]['emotion_valence'] = max(-1.0, min(1.0, new_v_guar))
                                            self.graph.nodes[uuid_guar]['emotion_arousal'] = max(0.0, min(1.0, new_a_guar))
                                except Exception as e_recon_guar: logger.warning(f"Error during emo recon for guaranteed {uuid_guar[:8]}: {e_recon_guar}")

        if core_added_count_val > 0: logger.info(f"Added {core_added_count_val} nodes due to Core Memory guarantee.")
        if saliency_guaranteed_added_count_val > 0: logger.info(f"Added {saliency_guaranteed_added_count_val} nodes due to high saliency guarantee.")
        if len(processed_uuids_for_access_count_set) > 0: logger.info(f"Incremented access count for {len(processed_uuids_for_access_count_set)} retrieved nodes.")

        relevant_nodes_list_final = list(relevant_nodes_final_dict.values())
        relevant_nodes_list_final.sort(key=lambda x_sort: (x_sort.get('final_activation', 0.0), x_sort.get('timestamp', '')), reverse=True)

        log_parts_final = []
        for n_log in relevant_nodes_list_final:
            marker_log = ""
            if n_log.get('guaranteed_inclusion') == 'core': marker_log = "**"
            elif n_log.get('guaranteed_inclusion') == 'saliency': marker_log = "*"
            log_parts_final.append(f"{n_log['uuid'][:8]}({n_log['final_activation']:.3f}{marker_log})")
        logger.info(f"Final nodes ({len(relevant_nodes_list_final)} total): [{', '.join(log_parts_final)}]")

        logger.debug("--- Retrieved Node Details (Top 5) ---")
        for i_log, node_log in enumerate(relevant_nodes_list_final[:5]):
            saliency_log = node_log.get('saliency_score', '?'); strength_log = node_log.get('memory_strength', '?')
            saliency_str_log = f"{saliency_log:.2f}" if isinstance(saliency_log, (int, float)) else str(saliency_log)
            strength_str_log = f"{strength_log:.2f}" if isinstance(strength_log, (int, float)) else str(strength_log)
            guar_str_log = f" Guar:{node_log.get('guaranteed_inclusion')}" if node_log.get('guaranteed_inclusion') else ""
            logger.debug(f"  {i_log+1}. ({node_log['final_activation']:.3f}) UUID:{node_log['uuid'][:8]} Str:{strength_str_log} Count:{node_log.get('access_count','?')} Sal:{saliency_str_log}{guar_str_log} Text: '{strip_emojis(node_log.get('text', 'N/A')[:80])}...'")
        logger.debug("------------------------------------")

        final_retrieved_data_log = [{
            "uuid": n_data['uuid'], "type": n_data.get('node_type'), "final_activation": n_data.get('final_activation'),
            "saliency_score": n_data.get('saliency_score'), "memory_strength": n_data.get('memory_strength'),
            "access_count": n_data.get('access_count'), "guaranteed": n_data.get('guaranteed_inclusion'),
            "text_preview": n_data.get('text', '')[:50]
        } for n_data in relevant_nodes_list_final]
        log_tuning_event("RETRIEVAL_RESULT", {
            "personality": self.personality, "initial_node_uuids": initial_node_uuids,
            "activation_threshold": activation_threshold, "guaranteed_saliency_threshold": guaranteed_saliency_threshold,
            "final_retrieved_count": len(relevant_nodes_list_final), "final_retrieved_nodes": final_retrieved_data_log,
            "effective_mood": effective_mood,
        })

        asm_check_cfg = self.config.get('autobiographical_model', {}).get('dynamic_check', {})
        if asm_check_cfg.get('enable', False) and self.autobiographical_model:
            contradiction_threshold_asm = asm_check_cfg.get('contradiction_saliency_threshold', 0.8)
            contradiction_found_asm = False
            contradicting_node_uuid_asm = None
            node_text_asm = ""
            asm_summary_text_asm = ""
            node_saliency_asm = 0.0
            for node_info_asm in relevant_nodes_list_final:
                node_saliency_raw_asm = node_info_asm.get('saliency_score', 0.0)
                node_saliency_asm = node_saliency_raw_asm if isinstance(node_saliency_raw_asm, (int,float)) else 0.0

                if node_saliency_asm >= contradiction_threshold_asm:
                    node_text_asm = node_info_asm.get('text', '').lower()
                    asm_summary_text_asm = self.autobiographical_model.get('summary_statement', '').lower()
                    if (" not " in node_text_asm and " not " not in asm_summary_text_asm) or \
                            (" never " in node_text_asm and " always " in asm_summary_text_asm): # Basic check
                        logger.warning(f"Potential ASM contradiction! Node {node_info_asm['uuid'][:8]} (Sal: {node_saliency_asm:.2f}) vs ASM Summary.")
                        contradiction_found_asm = True
                        contradicting_node_uuid_asm = node_info_asm['uuid']
                        log_tuning_event("ASM_CONTRADICTION_DETECTED", {
                            "personality": self.personality, "node_uuid": contradicting_node_uuid_asm,
                            "node_text_preview": node_text_asm[:100], "node_saliency": node_saliency_asm,
                            "asm_summary_preview": asm_summary_text_asm[:100],
                        })
                        break
            if contradiction_found_asm and contradicting_node_uuid_asm:
                logger.warning(f"*** Potential ASM Contradiction Detected! *** Node {contradicting_node_uuid_asm[:8]} vs Current ASM.")
                self.autobiographical_model['needs_review'] = True
                self.autobiographical_model['last_contradiction_node'] = contradicting_node_uuid_asm
                self.autobiographical_model['last_contradiction_time'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"ASM flagged for review. Conflicting node: {contradicting_node_uuid_asm[:8]}")
                log_tuning_event("ASM_CONTRADICTION_FLAGGED", {
                    "personality": self.personality, "conflicting_node_uuid": contradicting_node_uuid_asm,
                    "conflicting_node_text_preview": node_text_asm[:100], "conflicting_node_saliency": node_saliency_asm,
                    "asm_summary_preview": asm_summary_text_asm[:100], "asm_state_at_detection": self.autobiographical_model.copy()
                })
        return relevant_nodes_list_final, effective_mood




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
        last_accessed = node_data.get('last_accessed_ts', 0)
        recency_sec = max(0, current_time - last_accessed)
        activation = node_data.get('activation_level', 0.0)
        node_type = node_data.get('node_type', 'default')
        saliency = node_data.get('saliency_score', 0.0)
        valence = node_data.get('emotion_valence', 0.0)
        arousal = node_data.get('emotion_arousal', 0.1)
        node_emotion_magnitude = math.sqrt(valence**2 + arousal**2)
        degree = self.graph.degree(node_uuid) if node_uuid in self.graph else 0
        access_count = node_data.get('access_count', 0)

        decay_constant = weights.get('recency_decay_constant', 0.000005)
        norm_recency_raw = 1.0 - math.exp(-decay_constant * recency_sec)
        max_norm_recency_cap = weights.get('max_norm_recency_cap', 0.95)
        norm_recency = min(norm_recency_raw, max_norm_recency_cap)
        
        norm_inv_activation = 1.0 - min(1.0, max(0.0, activation))
        
        type_map_forget = {'turn': 1.0, 'summary': 0.4, 'concept': 0.1, 'intention': 0.2, 'boundary': 0.05, 'default': 0.6}
        norm_type_forgettability = type_map_forget.get(node_type, 0.6)
        
        norm_inv_saliency = 1.0 - min(1.0, max(0.0, saliency))
        norm_inv_emotion_node = 1.0 - min(1.0, max(0.0, node_emotion_magnitude / 1.414))
        norm_inv_connectivity = 1.0 - min(1.0, math.log1p(degree) / math.log1p(weights.get('connectivity_log_base', 10)))
        norm_inv_access_count = 1.0 - min(1.0, math.log1p(access_count) / math.log1p(weights.get('access_count_log_base', 20)))

        base_forget_score_val = (norm_recency * weights.get('recency_factor', 0.4) +
                                 norm_inv_activation * weights.get('activation_factor', 0.3) +
                                 norm_type_forgettability * weights.get('node_type_factor', 0.2))
        base_forget_score_val = max(0.0, min(1.0, base_forget_score_val))

        saliency_resist_mult = 1.0 - (saliency * weights.get('saliency_resistance_factor', 0.3))
        emotion_resist_mult_node = 1.0 - ((node_emotion_magnitude / 1.414) * weights.get('emotion_magnitude_resistance_factor', 0.2))
        connectivity_resist_mult = 1.0 - ((min(1.0, math.log1p(degree) / math.log1p(weights.get('connectivity_log_base', 10)))) * weights.get('connectivity_resistance_factor', 0.15))
        access_count_resist_mult = 1.0 - ((min(1.0, math.log1p(access_count) / math.log1p(weights.get('access_count_log_base', 20)))) * weights.get('access_count_resistance_factor', 0.1))
        type_resist_mult = node_data.get('decay_resistance_factor', 1.0) 

        intermediate_score_val = base_forget_score_val * \
                                 max(0.01, saliency_resist_mult) * \
                                 max(0.01, emotion_resist_mult_node) * \
                                 max(0.01, connectivity_resist_mult) * \
                                 max(0.01, access_count_resist_mult) * \
                                 max(0.01, type_resist_mult)
        
        logger.debug(f"    Forget Factors {node_uuid[:8]}: BaseF:{base_forget_score_val:.2f}, SalR:{saliency_resist_mult:.2f}, EmoR:{emotion_resist_mult_node:.2f}, ConR:{connectivity_resist_mult:.2f}, AccR:{access_count_resist_mult:.2f}, TypR:{type_resist_mult:.2f} -> InterS: {intermediate_score_val:.3f}")
        final_adjusted_score_val = intermediate_score_val

        if self.emotional_core and self.emotional_core.is_enabled:
            global_mood_v = self.emotional_core.derived_mood_hints.get("valence", 0.0)
            global_mood_a = self.emotional_core.derived_mood_hints.get("arousal", 0.0)
            
            stress_arousal_thresh = weights.get('global_stress_threshold_arousal', 0.7)
            stress_valence_thresh = weights.get('global_stress_threshold_valence', -0.3)
            stress_forget_mult = weights.get('global_stress_forget_factor', 1.2)
            saliency_protection_stress = weights.get('saliency_for_stress_protection', 0.5)

            is_globally_stressed = (global_mood_a >= stress_arousal_thresh and 
                                    global_mood_v <= stress_valence_thresh)
            is_protected_node = (node_data.get('saliency_score', 0.0) >= saliency_protection_stress or 
                                 node_data.get('is_core_memory', False))

            if is_globally_stressed and not is_protected_node:
                original_score_before_stress = final_adjusted_score_val
                final_adjusted_score_val *= stress_forget_mult
                logger.debug(f"    Global AI stress. Increasing forgettability for {node_uuid[:8]} by x{stress_forget_mult:.2f}. Score {original_score_before_stress:.4f} -> {final_adjusted_score_val:.4f}")
                log_tuning_event("FORGETTABILITY_GLOBAL_STRESS_EFFECT", {
                    "personality": self.personality, "node_uuid": node_uuid,
                    "global_mood": (global_mood_v, global_mood_a),
                    "original_score_before_stress": original_score_before_stress,
                    "stress_factor_applied": stress_forget_mult,
                    "final_score_after_stress": final_adjusted_score_val
                })

        flashbulb_mag_thresh_val = weights.get('flashbulb_emotion_magnitude_threshold', 1.2) 
        if node_emotion_magnitude >= flashbulb_mag_thresh_val:
            logger.debug(f"    Node {node_uuid[:8]} is 'Flashbulb' (EmoMag: {node_emotion_magnitude:.3f}). Setting forgettability very low.")
            final_adjusted_score_val = 0.001 
            if self.config.get('features', {}).get('enable_core_memory', False) and not node_data.get('is_core_memory', False):
                self.graph.nodes[node_uuid]['is_core_memory'] = True
                logger.info(f"    Flagged flashbulb node {node_uuid[:8]} as CORE MEMORY.")
                log_tuning_event("CORE_MEMORY_FLAGGED", {"personality": self.personality, "node_uuid": node_uuid, "reason": "flashbulb_emotion"})
            node_data['is_core_memory'] = True 

        if node_data.get('is_core_memory', False) and final_adjusted_score_val > 0.001: 
            core_immunity_cfg = self.config.get('core_memory', {}).get('forget_immunity', True)
            if core_immunity_cfg:
                final_adjusted_score_val = 0.0
            else:
                core_resistance_mult = weights.get('core_memory_resistance_factor', 0.05) 
                final_adjusted_score_val *= core_resistance_mult
            logger.debug(f"    Node {node_uuid[:8]} is Core. Immunity:{core_immunity_cfg}. Score after core logic: {final_adjusted_score_val:.4f}")

        final_adjusted_score_val = max(0.0, min(1.0, final_adjusted_score_val))
        log_tuning_event("FORGETTABILITY_FINAL_SCORE", {
            "personality": self.personality, "node_uuid": node_uuid, "node_type": node_type,
            "intermediate_score_before_global_effects": intermediate_score_val, 
            "is_core_memory": node_data.get('is_core_memory', False),
            "final_forgettability_score": final_adjusted_score_val,
            "current_memory_strength": node_data.get('memory_strength', 1.0),
        })
        return final_adjusted_score_val

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

        purge_check_details = {} # For logging

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
        logger.debug(f"_call_kobold_api received prompt ('{strip_emojis(prompt[:80])}...'). Length: {len(prompt)}") # Strip emojis for logging
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

        # --- MOVED THIS LINE DOWN ---
        # Check if streaming is configured for this LLM task.
        # For a 'generate' API, streaming might involve parsing different response structures.
        # For simplicity, assuming non-streaming for direct _call_kobold_api,
        # but if a config flag existed:
        # if self.config.get('llm_models', {}).get(some_task_name, {}).get('stream', False):
        #    payload['stream'] = True
        # For now, let's assume the direct generate call might not use streaming by default
        # or that the API handles non-streaming if `stream` is not present.
        # If streaming is *always* desired for this specific API, uncomment:
        # payload['stream'] = True # Add to payload
        # For now, let's assume Kobold's generate API doesn't require this or handles it.
        # If you intend to use streaming with the /api/v1/generate endpoint,
        # you'd typically make the request and then iterate over response.iter_lines()
        # similar to how _call_kobold_multimodal_api (or a revised _call_kobold_api) would.
        # The current structure of _call_kobold_api parsing `response.json()` suggests
        # it's expecting a non-streaming, full JSON response.

        # Log payload (masking potentially long prompt)
        log_payload = payload.copy()
        log_payload['prompt'] = strip_emojis(log_payload['prompt'][:100]) + ("..." if len(log_payload['prompt']) > 100 else "") # Strip emojis for logging
        logger.debug(f"Payload sent to Kobold API ({api_url}): {log_payload}")
        logger.debug(f"Kobold API Raw Prompt ENDING WITH: ...{strip_emojis(prompt[-200:])}")


        try:
            response = requests.post(api_url, json=payload, timeout=180) # 3-minute timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- ADDED LOGGING: Raw response before JSON parsing ---
            raw_response_text_from_api = response.text
            logger.info(f"Kobold API RAW Full Response (before JSON parse/strip): ```{raw_response_text_from_api}```")
            # --- END ADDED LOGGING ---

            result = response.json()
            # Extract generated text (structure might vary slightly between Kobold versions)
            gen_txt = result.get('results', [{}])[0].get('text', '').strip()

            # --- ADDED LOGGING: Text field before stop sequence stripping ---
            logger.info(f"Kobold API 'text' field from JSON (before stop sequence strip): ```{gen_txt}```")
            # --- END ADDED LOGGING ---

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
                logger.debug(f"Kobold API cleaned response text: '{strip_emojis(cleaned_txt[:100])}...'") # Strip emojis for logging

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
        """
        logger.info(f"Analyzing for action request: '{strip_emojis(request_text[:100])}...'")
        
        prompt_template = self._load_prompt("action_analysis_prompt.txt") 
        
        if not prompt_template:
            logger.error("Failed to load action analysis prompt template. Cannot analyze action.")
            return {'action': 'error', 'reason': 'Action analysis prompt template missing.'}

        full_prompt = "" 
        try:
            full_prompt = prompt_template.format( 
                request_text=request_text
            )
        except IndexError as ie: 
            # This error should now be rare if the prompt file is correct.
            logger.error(f"INDEX_ERROR during .format() in analyze_action_request for request_text='{strip_emojis(request_text)}'. Prompt may still have issues.")
            logger.error(f"Problematic template snippet (if loaded):\n{prompt_template[:500]}...") # Log a snippet
            return {'action': 'error', 'reason': f'Prompt formatting IndexError: {ie}. Check prompt file and logs.'}
        except Exception as e_fmt: 
            logger.error(f"OTHER FORMATTING ERROR in analyze_action_request for request_text='{strip_emojis(request_text)}': {e_fmt}", exc_info=True)
            return {'action': 'error', 'reason': f'Prompt formatting error: {e_fmt}.'}
            
        logger.debug(f"Sending action analysis prompt (from file):\n{strip_emojis(full_prompt)[:500]}...")
        llm_response_str = self._call_configured_llm('action_analysis', prompt=full_prompt)

        if not llm_response_str or llm_response_str.startswith("Error:"):
            error_reason = llm_response_str if llm_response_str else "LLM call failed (empty response)"
            logger.error(f"Action analysis failed due to LLM API error: {error_reason}")
            return {'action': 'error', 'reason': error_reason}

        parsed_result = None
        json_str = "" 
        try:
            logger.debug(f"Raw action analysis response:  ```{llm_response_str}```")
            match = re.search(r'(\{.*\}|\[.*\])', llm_response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                # logger.debug(f"Extracted potential JSON string using regex: {json_str}") # Optional
                parsed_result = json.loads(json_str)
            else: # Fallback parsing
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()
                
                start_brace = cleaned_response.find('{')
                end_brace = cleaned_response.rfind('}')
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    json_str = cleaned_response[start_brace:end_brace + 1]
                    # logger.debug(f"Extracted JSON string using brace finding (fallback): {json_str}") # Optional
                    parsed_result = json.loads(json_str)
                else:
                    logger.error(f"Could not find valid JSON object in LLM response. Raw: '{llm_response_str}'")
                    return {'action': 'error', 'reason': 'Could not extract valid JSON object from LLM response.', 'raw_response': llm_response_str}

            if not isinstance(parsed_result, dict):
                 raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_result)}).")

            logger.info(f"LLM Parsed Action: {parsed_result}")
            action = parsed_result.get("action")
            
            tools = { # Define tools locally for this function's scope
                "create_file": ["filename", "content"], "append_file": ["filename", "content"],
                "list_files": [], "read_file": ["filename"], "delete_file": ["filename"],
                "add_calendar_event": ["date", "time", "description"], "read_calendar": []
            }
            valid_actions = ["none", "clarify", "error"] + list(tools.keys())

            if not action or not isinstance(action, str) or action not in valid_actions:
                logger.warning(f"LLM returned unknown or invalid action '{action}'. Treating as 'none'. Raw: {llm_response_str}")
                return {"action": "none"}

            if action == "none": return {"action": "none"}
            if action == "error": return parsed_result # Pass through LLM-reported error

            args = parsed_result.get("args", {})
            if not isinstance(args, dict):
                raise ValueError(f"Invalid 'args' format for action '{action}'. Expected dict, got {type(args)}.")

            if action == "clarify":
                missing_args_val = parsed_result.get("missing_args")
                original_action_val = parsed_result.get("original_action")
                if not (isinstance(missing_args_val, list) and all(isinstance(item, str) for item in missing_args_val)):
                    raise ValueError("Clarify: 'missing_args' must be a list of strings.")
                if not isinstance(original_action_val, str):
                    raise ValueError("Clarify: 'original_action' must be a string.")
                if original_action_val not in tools:
                    raise ValueError(f"Clarify: 'original_action' '{original_action_val}' not in defined tools.")
                logger.info(f"Clarification requested for '{original_action_val}', missing: {missing_args_val}")
                return parsed_result

            required_args_list = tools.get(action, [])
            missing = []
            validated_args = {}
            for req_arg in required_args_list:
                arg_value = args.get(req_arg)
                if arg_value is None or (isinstance(arg_value, str) and not arg_value.strip()):
                    if not (action == "read_calendar" and req_arg == "date"): # 'date' is optional for read_calendar
                        missing.append(req_arg)
                else:
                    validated_args[req_arg] = str(arg_value).strip() # Ensure string and strip
            
            if missing:
                logger.warning(f"Action '{action}' identified, but missing required args: {missing}. Requesting clarification.")
                return {"action": "clarify", "missing_args": missing, "original_action": action}

            if "filename" in validated_args:
                original_filename = validated_args["filename"]
                # Use file_manager's safety check if available, or a simplified one
                # Assuming file_manager._is_filename_safe exists and is imported or accessible
                # For simplicity here, using os.path.basename and basic char check
                safe_filename = os.path.basename(original_filename) 
                # Basic invalid char pattern (can be expanded or use file_manager._is_filename_safe)
                if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip() or \
                   any(char in safe_filename for char in '<>:"/\\|?*'): # Simplified check
                    logger.error(f"Invalid or unsafe filename after basename: '{original_filename}' -> '{safe_filename}'")
                    return {'action': 'error', 'reason': f"Invalid filename provided: '{original_filename}'", 'raw_response': llm_response_str, 'parsed': parsed_result}
                validated_args["filename"] = safe_filename
                logger.debug(f"Sanitized filename: '{original_filename}' -> '{safe_filename}'")
            
            logger.info(f"Action analysis successful: Action='{action}', Args={validated_args}")
            return {"action": action, "args": validated_args}

        except json.JSONDecodeError as e:
            logger.error(f"LLM Action Parse Error (JSONDecodeError): {e}. Extracted String: '{json_str}'. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM response JSON parsing failed: {e}', 'raw_response': llm_response_str}
        except ValueError as e: # Catches our explicit ValueErrors for validation
            logger.error(f"LLM Action Validation Error: {e}. Parsed JSON: {parsed_result if 'parsed_result' in locals() else 'N/A'}. Raw: '{llm_response_str}'")
            return {'action': 'error', 'reason': f'LLM response validation failed: {e}', 'raw_response': llm_response_str, 'parsed': parsed_result if 'parsed_result' in locals() else None}
        except Exception as e: # Generic catch-all
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
        logger.info(f"Analyzing *memory modification* request: '{strip_emojis(request[:100])}...'") # Strip emojis
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
                    logger.info(f"Extracted new text: {strip_emojis(new_text[:50])}...") # Strip emojis
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
        logger.debug(f"Classifying query type for: '{strip_emojis(query_text[:100])}...'") # Strip emojis
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
        logger.debug(f"Analyzing for intention request: '{strip_emojis(request_text[:100])}...'") # Strip emojis
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
                logger.info(f"Intention detected: Content='{strip_emojis(content[:50])}...', Trigger='{trigger}'") # Strip emojis
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


    def _update_next_interaction_context(self, 
                                         user_node_uuid: str | None, 
                                         ai_node_uuid: str | None, 
                                         mood_used_for_current_retrieval: tuple[float, float]):
        """
        Helper to calculate and store concept/mood context for the *next* interaction's bias.
        The mood_used_for_current_retrieval is the effective mood that influenced the just-completed retrieval.
        """
        # --- Concept Extraction (remains the same) ---
        current_turn_concept_uuids = set()
        nodes_to_check_for_concepts = [uuid_val for uuid_val in [user_node_uuid, ai_node_uuid] if uuid_val] # Renamed internal var
        for turn_uuid in nodes_to_check_for_concepts:
            if turn_uuid in self.graph:
                try:
                    for successor_uuid in self.graph.successors(turn_uuid):
                        edge_data = self.graph.get_edge_data(turn_uuid, successor_uuid)
                        if edge_data and edge_data.get('type') == 'MENTIONS_CONCEPT':
                            if successor_uuid in self.graph and self.graph.nodes[successor_uuid].get('node_type') == 'concept':
                                current_turn_concept_uuids.add(successor_uuid)
                except Exception as concept_find_e:
                    logger.warning(f"Error finding concepts linked from turn {turn_uuid[:8]} for next bias: {concept_find_e}")
        
        if current_turn_concept_uuids: # Only log if there are concepts
            logger.info(f"Storing {len(current_turn_concept_uuids)} concepts for next interaction's bias.")
        else:
            logger.debug("No new concepts identified from current turn to store for next interaction bias.")
        self.last_interaction_concept_uuids = current_turn_concept_uuids

        # --- Store the mood that was effectively used for the current retrieval cycle ---
        # This mood already incorporates ST drive deviations and EmotionalCore hints from the *current* interaction.
        self.last_interaction_mood = mood_used_for_current_retrieval # Use the mood passed in
        
        logger.info(f"Storing mood V={self.last_interaction_mood[0]:.2f}, A={self.last_interaction_mood[1]:.2f} as bias for *next* interaction's retrieval.")
        log_tuning_event("NEXT_MOOD_BIAS_SET", {
            "personality": self.personality,
            "mood_set_for_next_bias": self.last_interaction_mood,
            "trigger": "end_of_interaction_context_update"
        })

        # --- Calculate Average Mood of Current Turn (as a base) ---
        current_turn_mood_base = (0.0, 0.1) # Default
        mood_nodes_found = 0
        total_valence = 0.0
        total_arousal = 0.0
        node_moods_for_avg = {}
        for node_uuid in nodes_to_check_for_concepts:
            if node_uuid in self.graph:
                node_data = self.graph.nodes[node_uuid]
                default_v = self.config.get('emotion_analysis', {}).get('default_valence', 0.0)
                default_a = self.config.get('emotion_analysis', {}).get('default_arousal', 0.1)
                node_v = node_data.get('emotion_valence', default_v)
                node_a = node_data.get('emotion_arousal', default_a)
                node_moods_for_avg[node_uuid[:8]] = {"V": node_v, "A": node_a}
                total_valence += node_v
                total_arousal += node_a
                mood_nodes_found += 1
        if mood_nodes_found > 0:
            current_turn_mood_base = (total_valence / mood_nodes_found, total_arousal / mood_nodes_found)
        logger.debug(f"Base mood from turn nodes: V={current_turn_mood_base[0]:.2f}, A={current_turn_mood_base[1]:.2f}")

        # --- Incorporate EmotionalCore Hints ---
        final_mood_for_next_bias = current_turn_mood_base
        if self.emotional_core and self.emotional_core.is_enabled:
            valence_hint = self.emotional_core.derived_mood_hints.get("valence", 0.0)
            arousal_hint = self.emotional_core.derived_mood_hints.get("arousal", 0.0)
            valence_factor = self.emotional_core.config.get("mood_valence_factor", 0.3) # Factor from EmotionalCore config
            arousal_factor = self.emotional_core.config.get("mood_arousal_factor", 0.2)

            if abs(valence_hint) > 1e-4 or abs(arousal_hint) > 1e-4:
                current_v, current_a = current_turn_mood_base # Start with the averaged mood
                # Apply hints additively, scaled by factors
                new_v = current_v + (valence_hint * valence_factor)
                new_a = current_a + (arousal_hint * arousal_factor)
                # Clamp final mood
                final_mood_for_next_bias = (max(-1.0, min(1.0, new_v)), max(0.0, min(1.0, new_a)))
                logger.info(f"Mood for next bias adjusted by EmoCore hints: Base=({current_v:.2f},{current_a:.2f}) Hints=({valence_hint:.2f},{arousal_hint:.2f}) -> Final=({final_mood_for_next_bias[0]:.2f},{final_mood_for_next_bias[1]:.2f})")
                log_tuning_event("NEXT_MOOD_BIAS_UPDATE", {
                    "personality": self.personality,
                    "base_averaged_mood": current_turn_mood_base,
                    "emocore_valence_hint": valence_hint,
                    "emocore_arousal_hint": arousal_hint,
                    "final_mood_for_bias": final_mood_for_next_bias,
                })
            else:
                logger.debug("No significant EmotionalCore hints to apply to next mood bias.")
        else:
            logger.debug("EmotionalCore disabled or unavailable, using base averaged mood for next bias.")

        logger.info(f"Storing final mood (Avg V/A): {final_mood_for_next_bias[0]:.2f} / {final_mood_for_next_bias[1]:.2f} for next interaction's bias.")
        self.last_interaction_mood = final_mood_for_next_bias # Update state


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
            logger.debug(f"Sending re-greeting prompt (init):\n{strip_emojis(re_greeting_prompt)}") # Strip emojis
            # Call LLM using dedicated config
            ai_response = self._call_configured_llm('re_greeting_generation', prompt=re_greeting_prompt)
            parsed_response = ai_response.strip() if ai_response and not ai_response.startswith("Error:") else "Hello again! It's been a while." # Fallback greeting

            # Store the generated greeting
            self.pending_re_greeting = parsed_response
            logger.info(f"Generated and stored pending re-greeting: '{strip_emojis(self.pending_re_greeting[:50])}...'") # Strip emojis
        else:
            logger.debug("Time gap does not exceed threshold. No re-greeting needed on init.")

    def get_pending_re_greeting(self) -> str | None:
        """Returns the pending re-greeting message (if any) and clears it."""
        greeting = self.pending_re_greeting
        self.pending_re_greeting = None # Clear after retrieval
        return greeting

    # --- Drive State Management ---
    def _initialize_drive_state(self):
        logger.info("Initializing drive state from config definitions...")
        drive_cfg = self.config.get('subconscious_drives', {})
        drive_definitions = drive_cfg.get('definitions', {})

        if not drive_definitions:
            logger.warning("No drive definitions found in config. Drive system might not function correctly.")
            self.drive_state = {"short_term": {}, "long_term": {}, "dynamic_baselines": {}}
            return

        st_drives = {}
        lt_drives = {}
        # dynamic_baselines will be calculated during updates, not stored persistently

        for drive_name, definition in drive_definitions.items():
            st_drives[drive_name] = float(definition.get('initial_short_term_level', 0.0))
            lt_drives[drive_name] = float(definition.get('initial_long_term_level', 0.0))
            # dynamic_baselines[drive_name] = lt_drives[drive_name] # Initial baseline is LT level

        self.drive_state = {
            "short_term": st_drives,
            "long_term": lt_drives,
            "dynamic_baselines": {} # Calculated on the fly during _update_drive_state
        }
        logger.info(f"Drive state initialized. ST: {self.drive_state['short_term']}, LT: {self.drive_state['long_term']}")

    def _load_drive_state(self):
        """Loads combined drive state (short & long term) from JSON file or initializes it."""
        default_state = {"short_term": {}, "long_term": {}}
        if os.path.exists(self.drives_file):
            try:
                with open(self.drives_file, 'r') as f:
                    loaded_state_from_file = json.load(f)

                drive_cfg = self.config.get('subconscious_drives', {})
                drive_definitions = drive_cfg.get('definitions', {})

                # Initialize with empty dicts to ensure keys exist
                self.drive_state = {"short_term": {}, "long_term": {}, "dynamic_baselines": {}}

                if isinstance(loaded_state_from_file, dict):
                    # Load ST and LT from file if they exist
                    self.drive_state["short_term"] = loaded_state_from_file.get("short_term", {})
                    self.drive_state["long_term"] = loaded_state_from_file.get("long_term", {})
                else:
                    logger.warning(f"Loaded drive state from {self.drives_file} is not a dict. Re-initializing.")
                    # Fallthrough to initialize from definitions

                # Ensure all defined drives are present and initialize if missing
                for drive_name, definition in drive_definitions.items():
                    if drive_name not in self.drive_state["short_term"]:
                        st_init = float(definition.get('initial_short_term_level', 0.0))
                        self.drive_state["short_term"][drive_name] = st_init
                        logger.info(f"Initialized missing ST drive '{drive_name}' to {st_init}")
                    if drive_name not in self.drive_state["long_term"]:
                        lt_init = float(definition.get('initial_long_term_level', 0.0))
                        self.drive_state["long_term"][drive_name] = lt_init
                        logger.info(f"Initialized missing LT drive '{drive_name}' to {lt_init}")

                logger.info(f"Drive state loaded/validated. ST: {self.drive_state['short_term']}, LT: {self.drive_state['long_term']}")

            except Exception as e:
                logger.error(f"Error loading or validating drive state from {self.drives_file}: {e}. Re-initializing.", exc_info=True)
                self._initialize_drive_state() # Fallback to full initialization

        # Dynamic baselines are not stored, they are calculated in _update_drive_state
        self.drive_state["dynamic_baselines"] = {}

    def _save_drive_state(self):
        if not self.drive_state or "short_term" not in self.drive_state or "long_term" not in self.drive_state:
            logger.warning("Skipping drive state save: state is incomplete or not initialized.")
            return
        try:
            state_to_save = {
                "short_term": self.drive_state.get("short_term", {}),
                "long_term": self.drive_state.get("long_term", {})
            }
            with open(self.drives_file, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            logger.debug(f"Drive state (ST/LT) saved to {self.drives_file}.")
        except Exception as e:
            logger.error(f"Error saving drive state: {e}", exc_info=True)

    # --- Signature changed to accept context_text ---
    def _update_drive_state(self,
                            context_text: str = "",
                            user_node_uuid: str | None = None,
                            ai_node_uuid: str | None = None,
                            interaction_id_for_log: str | None = None): # <<< ADD THIS PARAMETER HERE

        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False):
            return
        
        # Use a fallback interaction_id if not provided, for direct calls or testing
        current_interaction_id_for_profiling = interaction_id_for_log if interaction_id_for_log else getattr(self, 'current_interaction_id', 'N/A_DriveUpdate')

        logger.info(f"Updating short-term drive state (Interaction ID for logs: {current_interaction_id_for_profiling[:8]})...")
        drive_definitions = drive_cfg.get('definitions', {})
        st_state_before_cycle = self.drive_state.get("short_term", {}).copy() # Ensure "short_term" exists
        overall_changed_in_cycle = False

        # Ensure drive_state components exist, initialize if not (should be handled by _load_drive_state/_initialize_drive_state)
        if "short_term" not in self.drive_state or "long_term" not in self.drive_state:
            logger.error("Drive state components (short_term or long_term) missing in _update_drive_state. Re-initializing.")
            self._initialize_drive_state()

        st_state_before_cycle = self.drive_state.get("short_term", {}).copy()
        lt_drives = self.drive_state.get("long_term", {}) # For sensitivity calculations
        overall_changed_in_cycle = False

        # B. Decay Short-Term Drives (towards 0.0, as ST is now deviation from LT-influenced baseline of 0)
        st_decay_rate = drive_cfg.get('short_term_decay_rate', 0.03)
        if st_decay_rate > 0:
            decay_details_log = {}
            for drive_name, current_st_level in list(self.drive_state.get("short_term", {}).items()):
                change = (0.0 - current_st_level) * st_decay_rate
                new_st_level = current_st_level + change
                if (current_st_level > 0 and new_st_level < 0) or \
                        (current_st_level < 0 and new_st_level > 0):
                    new_st_level = 0.0

                # Use the delta for _apply_st_drive_adjustment
                if self._apply_st_drive_adjustment(drive_name, (new_st_level - current_st_level), "decay"):
                    decay_details_log[drive_name] = {"before": current_st_level, "after": self.drive_state["short_term"][drive_name], "decay_target": 0.0}
                    overall_changed_in_cycle = True

            if decay_details_log: # Log only if actual decays happened
                log_profile_event(current_interaction_id_for_profiling, self.personality, "DriveST_Decay", 0.0, f"Details:{len(decay_details_log)}_items") # Duration not measured for this sub-part alone
                log_tuning_event("DRIVE_ST_DECAY", {
                    "personality": self.personality, "decay_rate": st_decay_rate, "details": decay_details_log
                })

        # C. Heuristic Adjustments (from EmotionalCore results of the *last* interaction)
        if self.emotional_core and self.emotional_core.is_enabled:
            heuristics_cfg = drive_cfg.get('heuristic_adjustment_factors', {})
            if heuristics_cfg:
                logger.debug("Applying heuristic ST drive adjustments from last EmoCore analysis...")
                emo_results = self.emotional_core.current_analysis_results
                heuristic_adjustments_log = {}

                sentiment_compound = emo_results.get("sentiment", {}).get("compound", 0.0)
                s_thresh = heuristics_cfg.get('sentiment_trigger_threshold', 0.1)
                if abs(sentiment_compound) > s_thresh:
                    conn_adj_val = sentiment_compound * heuristics_cfg.get('sentiment_connection_adjustment', 0.0)
                    if self._apply_st_drive_adjustment("Connection", conn_adj_val, "sentiment_connection"):
                        overall_changed_in_cycle = True

                    safe_adj_val = sentiment_compound * heuristics_cfg.get('sentiment_safety_adjustment', 0.0)
                    if self._apply_st_drive_adjustment("Safety", safe_adj_val, "sentiment_safety"):
                        heuristic_adjustments_log["Safety_Sentiment"] = safe_adj_val
                        overall_changed_in_cycle = True

                need_conf_thresh = heuristics_cfg.get('need_confidence_threshold', 0.5)
                for emo_need_name, emo_need_data in emo_results.get("triggered_needs", {}).items():
                    if emo_need_data.get("confidence", 0.0) > need_conf_thresh:
                        for system_drive_name in drive_definitions.keys():
                            cfg_key = f"need_{emo_need_name}_{system_drive_name}_adjustment"
                            adjustment_val = heuristics_cfg.get(cfg_key)
                            if adjustment_val is not None and isinstance(adjustment_val, (int,float)) and abs(adjustment_val) > 1e-5:
                                if self._apply_st_drive_adjustment(system_drive_name, adjustment_val, f"need_{emo_need_name}"):
                                    heuristic_adjustments_log[f"{system_drive_name}_Need_{emo_need_name}"] = adjustment_val
                                    overall_changed_in_cycle = True

                fear_conf_thresh = heuristics_cfg.get('fear_confidence_threshold', 0.5)
                for emo_fear_name, emo_fear_data in emo_results.get("triggered_fears", {}).items():
                    if emo_fear_data.get("confidence", 0.0) > fear_conf_thresh:
                        for system_drive_name in drive_definitions.keys():
                            cfg_key = f"fear_{emo_fear_name}_{system_drive_name}_adjustment"
                            adjustment_val = heuristics_cfg.get(cfg_key)
                            if adjustment_val is not None and isinstance(adjustment_val, (int,float)) and abs(adjustment_val) > 1e-5:
                                if self._apply_st_drive_adjustment(system_drive_name, adjustment_val, f"fear_{emo_fear_name}"):
                                    heuristic_adjustments_log[f"{system_drive_name}_Fear_{emo_fear_name}"] = adjustment_val
                                    overall_changed_in_cycle = True

                # Placeholder for Preferences Influence
                pref_conf_thresh = heuristics_cfg.get('preference_confidence_threshold', 0.6) # Add to config
                for pref_name, pref_data in emo_results.get("triggered_preferences", {}).items():
                    if pref_data.get("confidence", 0.0) > pref_conf_thresh:
                        pref_type = pref_data.get("type", "unknown") # 'positive' or 'negative'
                        # Example: if Clarity preference is violated, it might frustrate Autonomy or Understanding
                        if pref_name == "Clarity" and pref_type == "negative": # Violated
                            adj_val = heuristics_cfg.get("preference_violation_Clarity_Autonomy_adjustment", 0.0)
                            if self._apply_st_drive_adjustment("Autonomy", adj_val, f"pref_violation_{pref_name}"):
                                heuristic_adjustments_log[f"Autonomy_PrefViolation_{pref_name}"] = adj_val
                                overall_changed_in_cycle = True
                        # Add more mappings from preference to drive adjustments

                if heuristic_adjustments_log:
                    log_tuning_event("DRIVE_ST_HEURISTIC_ADJUSTMENTS", {
                        "personality": self.personality, "trigger_source": "emotional_core_last_interaction",
                        "emo_core_results_summary": {
                            "sentiment_compound": sentiment_compound,
                            "triggered_needs_count": len(emo_results.get("triggered_needs", {})),
                            "triggered_fears_count": len(emo_results.get("triggered_fears", {})),
                            "triggered_preferences_count": len(emo_results.get("triggered_preferences", {}))
                        },
                        "adjustments_applied": heuristic_adjustments_log,
                        "st_state_after_heuristics": self.drive_state.get("short_term",{}).copy()
                    })

        # D. LLM Analysis for ST Drive Changes (if context_text is provided)
        if context_text and context_text.strip():
            logger.info("Attempting LLM analysis for ST drive update using provided context...")
            drive_definitions = self.config.get('subconscious_drives', {}).get('definitions', {})
            prompt_template_llm = self._load_prompt("drive_analysis_prompt.txt")
            if not prompt_template_llm:
                logger.error("drive_analysis_prompt.txt not found. Skipping LLM ST drive update.")
            else:
                drive_state_str_parts = ["[Current Short-Term Drive States (Negative=Frustrated, Positive=Satisfied):]"]
                for drive_name, st_level in self.drive_state.get("short_term", {}).items():
                    state_desc = "Neutral";
                    if st_level > 0.3: state_desc = "Satisfied"
                    elif st_level < -0.3: state_desc = "Frustrated"
                    drive_state_str_parts.append(f"- {drive_name}: {st_level:+.2f} ({state_desc})")
                current_drive_state_for_prompt = "\n".join(drive_state_str_parts)

                drive_defs_for_prompt_parts = ["[Drive Definitions:]"]
                for drive_name, definition in drive_definitions.items():
                    drive_defs_for_prompt_parts.append(f"- {drive_name}: {definition.get('description', 'N/A')}")
                drive_definitions_for_prompt = "\n".join(drive_defs_for_prompt_parts)

                full_prompt_llm = prompt_template_llm.format(
                    context_text=context_text[:3000], # Limit context length
                    drive_definitions=drive_definitions_for_prompt,
                    current_drive_state=current_drive_state_for_prompt
                )
                log_tuning_event("DRIVE_ST_LLM_PROMPT", {"personality": self.personality, "prompt_preview": full_prompt_llm[:500]})
                llm_response_str = self._call_configured_llm('drive_analysis_short_term', prompt=full_prompt_llm)

                if llm_response_str and not llm_response_str.startswith("Error:"):
                    try:
                        match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
                        json_str_llm = ""
                        if match: json_str_llm = match.group(0)
                        else: # Fallback cleaning if no strict JSON object found
                            cleaned_resp_llm = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                            start_brace_llm = cleaned_resp_llm.find('{'); end_brace_llm = cleaned_resp_llm.rfind('}')
                            if start_brace_llm != -1 and end_brace_llm != -1: json_str_llm = cleaned_resp_llm[start_brace_llm:end_brace_llm+1]

                        if not json_str_llm:
                            logger.error(f"Could not extract JSON from ST drive analysis LLM response. Raw: '{llm_response_str}'")
                            raise ValueError("No JSON object found in LLM response for ST drive analysis.")

                        llm_drive_scores = json.loads(json_str_llm)
                        log_tuning_event("DRIVE_ST_LLM_PARSED", {
                            "personality": self.personality, "raw_response": llm_response_str, "parsed_scores": llm_drive_scores
                        })

                        base_adj_factor = drive_cfg.get('llm_score_adjustment_factor', 0.1)
                        amp_factor = 1.0
                        if self.high_impact_nodes_this_interaction: # Check if dict has items
                            max_mag = max(self.high_impact_nodes_this_interaction.values()) if self.high_impact_nodes_this_interaction else 0.0
                            amp_cfg_factor = drive_cfg.get('emotional_impact_amplification_factor', 1.5)
                            impact_thresh = drive_cfg.get('emotional_impact_threshold', 0.8)
                            if impact_thresh > 0 and max_mag > impact_thresh: # Only amplify if above threshold
                                mag_ratio = max(0.0, (max_mag - impact_thresh) / impact_thresh) # How much it exceeds
                                amp_factor = 1.0 + (mag_ratio * (amp_cfg_factor - 1.0))
                                amp_factor = min(amp_cfg_factor, amp_factor) # Cap amplification
                                logger.info(f"Amplifying LLM ST drive adjustments by {amp_factor:.2f} due to high emotional impact (MaxMag: {max_mag:.2f})")

                        effective_adj_factor = base_adj_factor * amp_factor
                        llm_adjustments_log_applied = {}

                        for drive_name, score in llm_drive_scores.items():
                            if drive_name in self.drive_state.get("short_term", {}):
                                if not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
                                    logger.warning(f"LLM returned invalid score '{score}' for ST drive '{drive_name}'. Skipping."); continue

                                adjustment = score * effective_adj_factor

                                lt_level_for_sens = lt_drives.get(drive_name, 0.0)
                                sensitivity_factor = 1.0 + (lt_level_for_sens * 0.5) # LT trait influences sensitivity
                                adjustment *= max(0.1, sensitivity_factor)

                                if self._apply_st_drive_adjustment(drive_name, adjustment, "llm_analysis"):
                                    llm_adjustments_log_applied[drive_name] = {"score": score, "base_adj": base_adj_factor, "amp": amp_factor, "sens": sensitivity_factor, "final_adj": adjustment}
                                    overall_changed_in_cycle = True
                        if llm_adjustments_log_applied:
                            log_tuning_event("DRIVE_ST_LLM_ADJUSTMENTS_APPLIED", {
                                "personality": self.personality,
                                "adjustments": llm_adjustments_log_applied, "st_state_after_llm": self.drive_state.get("short_term",{}).copy()
                            })
                    except json.JSONDecodeError as e: logger.error(f"Failed to parse JSON from ST drive analysis: {e}. JSON tried: '{json_str_llm if 'json_str_llm' in locals() else 'N/A'}'. Raw: '{llm_response_str}'")
                    except ValueError as e: logger.error(f"ValueError during ST drive LLM response processing: {e}. Raw: '{llm_response_str}'")
                    except Exception as e: logger.error(f"Error processing ST drive analysis LLM response: {e}", exc_info=True)
                else:
                    logger.error(f"LLM call failed or returned error for ST drive analysis: {llm_response_str}")

        # E. Inter-Drive Dynamics
        inter_drive_cfg = drive_cfg.get('inter_drive_interactions', {})
        if inter_drive_cfg:
            logger.debug("Applying ST inter-drive dynamics...")
            inter_drive_adjustments_log_applied = {}
            st_state_before_inter_drive = self.drive_state.get("short_term",{}).copy()

            for influencing_drive, targets in inter_drive_cfg.items():
                if influencing_drive in st_state_before_inter_drive:
                    influencer_st_level = st_state_before_inter_drive[influencing_drive]
                    for target_drive, params in targets.items():
                        if target_drive in self.drive_state.get("short_term",{}):
                            threshold = params.get('threshold', 0.0)
                            factor = params.get('factor', 0.0)
                            apply_influence = False
                            if threshold >= 0 and influencer_st_level > threshold: apply_influence = True
                            elif threshold < 0 and influencer_st_level < threshold: apply_influence = True

                            if apply_influence and abs(factor) > 1e-5:
                                deviation_from_threshold = influencer_st_level - threshold
                                adjustment = factor * deviation_from_threshold

                                if self._apply_st_drive_adjustment(target_drive, adjustment, f"inter_drive_from_{influencing_drive}"):
                                    inter_drive_adjustments_log_applied.setdefault(target_drive, []).append({
                                        "influencer": influencing_drive, "inf_level": influencer_st_level,
                                        "threshold": threshold, "factor": factor, "adjustment": adjustment
                                    })
                                    overall_changed_in_cycle = True
            if inter_drive_adjustments_log_applied:
                log_tuning_event("DRIVE_ST_INTER_INTERACTIONS", {
                    "personality": self.personality,
                    "st_state_before_inter": st_state_before_inter_drive,
                    "adjustments": inter_drive_adjustments_log_applied,
                    "st_state_after_inter": self.drive_state.get("short_term",{}).copy()
                })

        if overall_changed_in_cycle:
            logger.info(f"Short-term drive state updated. Before all ST updates: {st_state_before_cycle}, After all ST updates: {self.drive_state.get('short_term', {})}")
            self._save_drive_state() # Save if any ST drive changed during this cycle
        else:
            logger.debug("ST Drive state unchanged after full update cycle.")


    def _apply_st_drive_adjustment(self, drive_name: str, adjustment: float, reason: str):
        """
        Applies an adjustment to a short-term drive, clamps it, and logs the change.
        """
        if drive_name not in self.drive_state.get("short_term", {}): # Check if short_term exists
            logger.warning(f"Attempted to adjust unknown ST drive '{drive_name}' (or ST drives not init) for reason '{reason}'.")
            return False # Indicate no change

        current_level = self.drive_state["short_term"][drive_name]
        new_level = current_level + adjustment

        min_st_level = -1.0
        max_st_level = 1.0
        clamped_new_level = max(min_st_level, min(max_st_level, new_level))

        if abs(clamped_new_level - current_level) > 1e-5: # Check for significant change
            self.drive_state["short_term"][drive_name] = clamped_new_level
            logger.debug(f"  ST Drive '{drive_name}' adjusted by {adjustment:+.3f} (Reason: {reason}). From {current_level:.3f} -> {clamped_new_level:.3f}")
            return True # Indicate change occurred
        # else:
        #     logger.debug(f"  ST Drive '{drive_name}' adjustment {adjustment:+.3f} (Reason: {reason}) resulted in no significant change from {current_level:.3f}.")
        return False # No significant change


    def _update_long_term_drives(self, high_impact_memory_uuid: str | None = None):
        drive_cfg = self.config.get('subconscious_drives', {})
        if not drive_cfg.get('enabled', False):
            return
        logger.info("Updating long-term drive state...")
        drive_definitions = drive_cfg.get('definitions', {})
        lt_state_before_cycle = self.drive_state["long_term"].copy()
        overall_lt_changed = False

        # A. Gradual Shift from Persistent ST Deviations (ST levels *are* deviations from 0)
        lt_adjustment_factor = drive_cfg.get('long_term_adjustment_factor', 0.01)
        lt_adjustments_log = {}
        if lt_adjustment_factor > 0:
            logger.debug("Applying gradual LT shift based on ST levels...")
            for drive_name, current_st_level in self.drive_state["short_term"].items():
                if drive_name in self.drive_state["long_term"]:
                    current_lt_level = self.drive_state["long_term"][drive_name]

                    # How much the LT trait resists change
                    stability = float(drive_definitions.get(drive_name, {}).get('long_term_stability_factor', 0.95))
                    change_sensitivity = 1.0 - stability # Higher stability = lower sensitivity

                    # ST level itself is the "persistent deviation" from the LT-influenced baseline of 0.
                    # If ST is consistently positive (satisfied), LT baseline might be too low.
                    # If ST is consistently negative (frustrated), LT baseline might be too high.
                    # So, LT should move in the direction of ST's sign.
                    lt_change = current_st_level * lt_adjustment_factor * change_sensitivity

                    if abs(lt_change) > 1e-5:
                        new_lt_level = current_lt_level + lt_change
                        new_lt_level = max(-1.0, min(1.0, new_lt_level)) # Clamp LT

                        if abs(new_lt_level - current_lt_level) > 1e-5:
                            self.drive_state["long_term"][drive_name] = new_lt_level
                            overall_lt_changed = True
                            lt_adjustments_log[drive_name] = {
                                "from_st_level": current_st_level, "stability": stability,
                                "change": lt_change, "old_lt": current_lt_level, "new_lt": new_lt_level
                            }
                            logger.debug(f"  LT Drive '{drive_name}' gradually shifted by ST level {current_st_level:.2f}: {current_lt_level:.3f} -> {new_lt_level:.3f}")
            if lt_adjustments_log:
                log_tuning_event("DRIVE_LT_FROM_ST_DEVIATIONS", {
                    "personality": self.personality, "lt_adj_factor": lt_adjustment_factor,
                    "adjustments": lt_adjustments_log
                })

        # B. LLM Analysis of ASM (Optional, Less Frequent - if configured)
        # This part can remain similar to your existing logic for ASM-based LT update,
        # just ensure the prompt asks for target LT levels or +/- adjustments for each drive.
        # For brevity, I'll skip reimplementing the full LLM call here but outline the idea:
        # if self.config.get(... 'run_asm_lt_drive_analysis' ...):
        #    asm_summary_text = ... (generate from self.autobiographical_model)
        #    prompt_template_asm_lt = self._load_prompt("long_term_drive_analysis_prompt.txt")
        #    full_prompt_asm_lt = prompt_template_asm_lt.format(asm_summary_text=asm_summary_text, drive_definitions=...)
        #    llm_response_asm_lt = self._call_configured_llm('drive_analysis_long_term', prompt=full_prompt_asm_lt)
        #    # Parse llm_response_asm_lt for {"DriveName": target_lt_level, ...}
        #    # Nudge current LT levels towards these targets, scaled by lt_adjustment_factor & stability.
        #    # overall_lt_changed = True if changes made

        # C. High-Impact Memory Nudge
        if high_impact_memory_uuid and high_impact_memory_uuid in self.graph:
            logger.info(f"Applying LT drive nudge from high-impact memory: {high_impact_memory_uuid[:8]}")
            node_data_him = self.graph.nodes[high_impact_memory_uuid]
            valence_him = node_data_him.get('emotion_valence', 0.0)
            # arousal_him = node_data_him.get('emotion_arousal', 0.1) # Could use arousal too
            shift_factor_cfg = drive_cfg.get('high_impact_memory_baseline_shift_factor', 0.1) # Factor from config
            him_nudge_log = {}

            # Define specific nudges based on valence (can be expanded)
            # Example: Strong positive experience (high valence) reinforces Connection and Competence LT.
            if valence_him > 0.7: # Threshold for strong positive
                for target_drive in ["Connection", "Competence"]:
                    if target_drive in self.drive_state["long_term"]:
                        current_lt = self.drive_state["long_term"][target_drive]
                        # Nudge towards positive max (+1.0)
                        adjustment = (1.0 - current_lt) * shift_factor_cfg * valence_him # Scale by valence intensity
                        if abs(adjustment) > 1e-5:
                            new_lt = max(-1.0, min(1.0, current_lt + adjustment))
                            self.drive_state["long_term"][target_drive] = new_lt
                            overall_lt_changed = True
                            him_nudge_log[target_drive] = {"before": current_lt, "after": new_lt, "reason": f"high_valence_memory ({valence_him:.2f})"}
                            logger.debug(f"  LT Drive '{target_drive}' nudged by high valence memory: {current_lt:.3f} -> {new_lt:.3f}")

            # Example: Strong negative experience (low valence) might reinforce Safety LT or decrease Novelty LT.
            elif valence_him < -0.7: # Threshold for strong negative
                for target_drive in ["Safety"]: # Drive to increase
                    if target_drive in self.drive_state["long_term"]:
                        current_lt = self.drive_state["long_term"][target_drive]
                        adjustment = (1.0 - current_lt) * shift_factor_cfg * abs(valence_him) # Nudge towards positive max
                        if abs(adjustment) > 1e-5:
                            new_lt = max(-1.0, min(1.0, current_lt + adjustment))
                            self.drive_state["long_term"][target_drive] = new_lt
                            overall_lt_changed = True
                            him_nudge_log[target_drive] = {"before": current_lt, "after": new_lt, "reason": f"low_valence_memory ({valence_him:.2f})"}
                            logger.debug(f"  LT Drive '{target_drive}' nudged by low valence memory: {current_lt:.3f} -> {new_lt:.3f}")
                # Example for decreasing a drive
                for target_drive_decrease in ["Novelty"]:
                    if target_drive_decrease in self.drive_state["long_term"]:
                        current_lt_dec = self.drive_state["long_term"][target_drive_decrease]
                        adjustment_dec = (-1.0 - current_lt_dec) * shift_factor_cfg * abs(valence_him) # Nudge towards negative min
                        if abs(adjustment_dec) > 1e-5:
                            new_lt_dec = max(-1.0, min(1.0, current_lt_dec + adjustment_dec))
                            self.drive_state["long_term"][target_drive_decrease] = new_lt_dec
                            overall_lt_changed = True
                            him_nudge_log[target_drive_decrease] = {"before": current_lt_dec, "after": new_lt_dec, "reason": f"low_valence_memory_suppression ({valence_him:.2f})"}
                            logger.debug(f"  LT Drive '{target_drive_decrease}' (suppression) nudged by low valence memory: {current_lt_dec:.3f} -> {new_lt_dec:.3f}")


            if him_nudge_log:
                log_tuning_event("DRIVE_LT_HIGH_IMPACT_NUDGE", {
                    "personality": self.personality, "memory_uuid": high_impact_memory_uuid,
                    "valence_him": valence_him, "shift_factor": shift_factor_cfg,
                    "adjustments": him_nudge_log
                })

        if overall_lt_changed:
            logger.info(f"Long-term drive state updated. Before cycle: {lt_state_before_cycle}, After cycle: {self.drive_state['long_term']}")
            self._save_drive_state() # Save LT changes immediately as they are infrequent
        else:
            logger.debug("LT Drive state unchanged after full update cycle.")


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
        interaction_id_for_log = getattr(self, 'current_interaction_id', 'N/A_LLM_Call_Context')
        
        step_start_time = time.perf_counter()
        logger.debug(f"Calling configured LLM for task: '{task_name}' (Interaction: {interaction_id_for_log[:8]})")
        task_config = self.config.get('llm_models', {}).get(task_name)

        if not task_config:
            err_msg = f"Error: LLM configuration for task '{task_name}' not found in config.yaml."
            logger.error(err_msg)
            log_profile_event(interaction_id_for_log, self.personality, f"LLMCall_{task_name}_FailConfig", time.perf_counter() - step_start_time)
            return err_msg

        api_type = task_config.get('api_type')
        model_name = task_config.get('model_name', 'koboldcpp-default')
        
        # Default parameters structure - keep both max_length and max_tokens here for flexibility
        default_params_from_config = {
            'max_length': task_config.get('max_length', 512), # For generate API
            'max_tokens': task_config.get('max_tokens', 512), # For chat_completions API
            'temperature': task_config.get('temperature', 0.7),
            'top_p': task_config.get('top_p', 0.95),
            'top_k': task_config.get('top_k', 60),
            'min_p': task_config.get('min_p', 0.0),
        }
        
        # Merge overrides into a temporary copy for this call
        current_call_params = {**default_params_from_config, **overrides}
        logger.debug(f"  Task Config Base: {task_config}")
        logger.debug(f"  Effective Params for this call: {current_call_params}")

        result_text = f"Error: LLM task '{task_name}' did not execute due to API type mismatch."

        if api_type == 'generate':
            if prompt is None:
                err_msg = f"Error: Prompt is required for 'generate' API type (task: {task_name})."
                logger.error(err_msg)
                log_profile_event(interaction_id_for_log, self.personality, f"LLMCall_{task_name}_FailInput", time.perf_counter() - step_start_time, "PromptMissing")
                return err_msg
            
            # --- FIX: Explicitly pass parameters _call_kobold_api expects ---
            generate_api_params = {
                'model_name': model_name,
                'max_length': current_call_params['max_length'],
                'temperature': current_call_params['temperature'],
                'top_p': current_call_params['top_p'],
                'top_k': current_call_params['top_k'],
                'min_p': current_call_params['min_p']
            }
            result_text = self._call_kobold_api(prompt=prompt, **generate_api_params)

        elif api_type == 'chat_completions':
            if messages is None:
                err_msg = f"Error: Messages list is required for 'chat_completions' API type (task: {task_name})."
                logger.error(err_msg)
                log_profile_event(interaction_id_for_log, self.personality, f"LLMCall_{task_name}_FailInput", time.perf_counter() - step_start_time, "MessagesMissing")
                return err_msg

            # --- FIX: Explicitly pass parameters _call_kobold_multimodal_api expects ---
            chat_api_params = {
                'model_name': model_name,
                'max_tokens': current_call_params['max_tokens'], # Use max_tokens here
                'temperature': current_call_params['temperature'],
                'top_p': current_call_params['top_p']
                # Add top_k, min_p if your multimodal API supports them
            }
            result_text = self._call_kobold_multimodal_api(messages=messages, **chat_api_params)
        else:
            err_msg = f"Error: Unknown api_type '{api_type}' configured for task '{task_name}'."
            logger.error(err_msg)
            log_profile_event(interaction_id_for_log, self.personality, f"LLMCall_{task_name}_FailAPIType", time.perf_counter() - step_start_time)
            return err_msg
        
        duration = time.perf_counter() - step_start_time
        success_status = "Success" if not (result_text is None or result_text.startswith("Error:")) else "Fail"
        output_len = len(result_text) if result_text else 0
        log_profile_event(interaction_id_for_log, self.personality, f"LLMCall_{task_name}", duration, f"Status:{success_status},APIType:{api_type},OutLen:{output_len}")
        return result_text

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


    def _handle_input(self, interaction_id: str, user_input: str, conversation_history: list, attachment_data: dict | None) -> tuple[str | None, str, list, tuple | None, tuple | None, tuple[float,float] | None]: # ADDED 6th return type
        """
        Handles input processing, choosing between text or multimodal, and calls the LLM.
        Returns: (inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion, effective_mood_for_retrieval_if_any)
        """
        inner_thoughts = None
        raw_llm_response_text = "Error: LLM call failed." # Default
        user_emotion = None
        ai_emotion = None
        memories_retrieved = []
        effective_mood_for_retrieval_if_any = None # Initialize

        if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('data_url'):
            logger.info(f"Interaction {interaction_id[:8]}: Handling multimodal input.")
            inner_thoughts, raw_llm_response_text = self._handle_multimodal_input(user_input, attachment_data)
            # No specific mood calculated *for retrieval bias* in multimodal path currently
            # memories_retrieved will be empty
        else:
            logger.info(f"Interaction {interaction_id[:8]}: Handling text input.")
            # _handle_text_input now returns 6 items
            inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion, effective_mood_for_retrieval_if_any = self._handle_text_input(user_input, conversation_history)

        return inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion, effective_mood_for_retrieval_if_any


    def process_interaction(self, user_input: str, conversation_history: list, attachment_data: dict | None = None) -> InteractionResult:
        interaction_id = str(uuid.uuid4())
        logger.info(f"--- Processing Interaction START (ID: {interaction_id[:8]}) ---")
        logger.info(f"Input='{strip_emojis(user_input[:60])}...', Attachment: {attachment_data.get('type') if attachment_data else 'No'}")
        self.high_impact_nodes_this_interaction.clear() # Reset for this interaction

        # Default error result object
        error_result = InteractionResult(final_response_text="Error: Processing failed unexpectedly.")

        log_tuning_event("INTERACTION_START", {
            "interaction_id": interaction_id,
            "personality": self.personality,
            "user_input_preview": strip_emojis(user_input[:100]),
            "has_attachment": bool(attachment_data),
            "attachment_type": attachment_data.get('type') if attachment_data else None,
            "history_length": len(conversation_history)
        })

        if not hasattr(self, 'embedder') or self.embedder is None:
            logger.critical("PROCESS_INTERACTION CRITICAL ERROR: Embedder not initialized!")
            log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "embedder_check", "error": "Embedder not initialized" })
            return error_result

        try:
            # Determine text to save in the graph for the user turn (includes attachment placeholder)
            graph_user_input = user_input
            if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('filename'):
                placeholder = f" [Image Attached: {attachment_data['filename']}]"
                separator = " " if graph_user_input else ""
                graph_user_input += separator + placeholder

            # --- Step 0: NLU for Interaction Type (Topic Change, Memory Mod Intent, etc.) ---
            history_text_for_nlu = "\n".join([f"{turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}" for turn in conversation_history[-3:]])
            
            # Call NLU analysis and provide default values if it fails or returns None
            nlu_type_from_analysis, nlu_details_from_analysis = self._analyze_interaction_type(user_input, history_text_for_nlu)
            
            current_nlu_type = nlu_type_from_analysis if nlu_type_from_analysis is not None else "unknown"
            current_nlu_details = nlu_details_from_analysis # This can be None, and that's okay

            logger.info(f"Interaction {interaction_id[:8]}: NLU analysis result - Type: '{current_nlu_type}', Details: {current_nlu_details}")

            # --- Handle specific NLU-detected types BEFORE main chat flow (if needed for early exit) ---
            # For now, we are mostly passing this info along. A more complex system might fork flow here.
            if current_nlu_type == "memory_modification_request":
                logger.info(f"Interaction {interaction_id[:8]}: NLU identified memory modification intent. Details: {current_nlu_details}. Current design will proceed to standard chat flow; this intent should ideally be handled directly by the worker via a specific task if no chat response is desired.")
                # If you wanted to handle this without a chat response, you would:
                # 1. Call self.analyze_memory_modification_request(user_input)
                # 2. The worker would emit a `modification_response_ready` signal.
                # 3. You'd return an InteractionResult here that tells the GUI worker not to expect a chat response,
                #    or an InteractionResult with a system message like "Okay, I'm processing that memory request."
                # For this iteration, we'll let it flow through to chat generation.

            # --- Step 1: Handle Input & Call LLM (text or multimodal) ---
            # This function now returns: (inner_thoughts, raw_llm_response, memories_retrieved, user_emotion, ai_emotion)
            inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion, effective_mood_retrieval_for_this_turn = self._handle_input(
                interaction_id, user_input, conversation_history, attachment_data
            )
            
            # _handle_input already calls _parse_llm_response if it's handling text,
            # or _parse_llm_response directly if it's multimodal.
            # So, raw_llm_response_text IS the final_response_text from the LLM after thought stripping.
            final_response_text_from_llm = raw_llm_response_text # This is already the post-thought-stripping response

            # Check for critical LLM call failure
            if final_response_text_from_llm is None or "Error:" in final_response_text_from_llm[:20]:
                 logger.error(f"Interaction {interaction_id[:8]}: LLM call failed or returned error: '{final_response_text_from_llm}'")
                 log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "llm_call", "error": final_response_text_from_llm or "Empty LLM response" })
                 error_result.final_response_text = final_response_text_from_llm or "Error: LLM processing failed."
                 return error_result

            # --- Step 2: Check for [ACTION:] tag in AI's response (for workspace planning) ---
            # Also consider NLU hints and keywords for planning.
            final_response_text_cleaned_after_action_check, needs_planning_from_tag, extracted_tag_json = self._check_for_action_request(
                response_text=final_response_text_from_llm,
                user_input=user_input
            )
            
            needs_overall_planning = needs_planning_from_tag # Start with tag result
            if not needs_overall_planning: # Only check further if tag didn't already trigger it
                if current_nlu_type == "workspace_action_request":
                    logger.info(f"Interaction {interaction_id[:8]}: NLU detected 'workspace_action_request'. Setting needs_planning=True.")
                    needs_overall_planning = True
                elif any(keyword in user_input.lower() for keyword in WORKSPACE_KEYWORDS): # Fallback to keywords
                    logger.info(f"Interaction {interaction_id[:8]}: Workspace keywords found in user input. Setting needs_planning=True.")
                    needs_overall_planning = True
            
            logger.debug(f"Interaction {interaction_id[:8]}: Needs Planning Flag = {needs_overall_planning}. Cleaned Response: '{final_response_text_cleaned_after_action_check[:60]}...'")

            # --- Step 3: Update Graph & Context ---
            # Ensure all arguments are correctly passed
            user_node_uuid, ai_node_uuid = self._update_graph_and_context(
                graph_user_input=graph_user_input,
                user_input_for_emotional_core_context=user_input,
                ai_response_for_emotional_core_context=final_response_text_cleaned_after_action_check,
                parsed_response=final_response_text_cleaned_after_action_check, # <<< ADD THIS LINE
                user_emotion_values=user_emotion,
                ai_emotion_values=ai_emotion,
                mood_used_for_current_retrieval=effective_mood_retrieval_for_this_turn,
                detected_interaction_type=current_nlu_type,
                detected_intent_details=current_nlu_details
            )
            logger.debug(f"Interaction {interaction_id[:8]}: Graph updated. User Node: {user_node_uuid}, AI Node: {ai_node_uuid}")

            # --- Step 4: Assemble and Return Result ---
            final_result = InteractionResult(
                final_response_text=final_response_text_cleaned_after_action_check,
                inner_thoughts=inner_thoughts,
                memories_used=memories_retrieved,
                user_node_uuid=user_node_uuid, # Use returned UUIDs
                ai_node_uuid=ai_node_uuid,     # Use returned UUIDs
                needs_planning=needs_overall_planning,
                detected_interaction_type=current_nlu_type,
                detected_intent_details=current_nlu_details,
                extracted_ai_action_tag_json=extracted_tag_json
            )

            log_tuning_event("INTERACTION_END", {
                "interaction_id": interaction_id,
                "personality": self.personality,
                "final_response_preview": strip_emojis(final_result.final_response_text[:100]),
                "retrieved_memory_count": len(final_result.memories_used),
                "user_node_added": final_result.user_node_uuid[:8] if final_result.user_node_uuid else None,
                "ai_node_added": final_result.ai_node_uuid[:8] if final_result.ai_node_uuid else None,
                "needs_planning": final_result.needs_planning,
                "nlu_interaction_type": final_result.detected_interaction_type,
                "nlu_details_keys": list(final_result.detected_intent_details.keys()) if final_result.detected_intent_details else None
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
            error_result.final_response_text = error_message_for_user
            return error_result


    def _handle_text_input(self, user_input: str, conversation_history: list) -> tuple[str | None, str, list, tuple | None, tuple | None, tuple[float, float]]: # ADDED 6th return item
        """
        Handles text-based input, including emotional analysis, memory retrieval, and LLM call.
        Returns: (inner_thoughts, final_response, memories_retrieved, user_emotion, ai_emotion, effective_mood_for_retrieval)
        """
        logger.info("Handling text input...")
        user_emotion_result = None
        ai_emotion_result = None
        mood_bias_for_retrieval = self.last_interaction_mood # Mood from *previous* interaction's end
        # current_turn_emotional_hints is handled within retrieve_memory_chain via self.emotional_core
        emotional_instructions = ""

        # --- 1. Emotional Analysis (if enabled) - RUN THIS FIRST ---
        if self.emotional_core and self.emotional_core.is_enabled:
            try:
                history_context_str = "\n".join([f"{strip_emojis(turn.get('text', ''))}" for turn in conversation_history[-3:]]) # Simpler history
                kg_context_str = self._get_kg_context_for_emotion(user_input)
                self.emotional_core.analyze_input(user_input, history_context_str, kg_context_str)
                # aggregate_and_combine stores results in self.emotional_core.derived_mood_hints
                # and returns tendency, derived_mood_hints
                _, _ = self.emotional_core.aggregate_and_combine() # We only need the side effect for now
                emotional_instructions = self.emotional_core.craft_prompt_instructions()

                user_sentiment = self.emotional_core.current_analysis_results.get("sentiment", {})
                user_valence = user_sentiment.get("compound", 0.0)
                # Simplified arousal from VADER for user turn
                user_arousal_simple = (user_sentiment.get("pos", 0.0) + user_sentiment.get("neg", 0.0)) * 0.5
                user_emotion_result = (user_valence, max(0.0, min(1.0, user_arousal_simple))) # Clamp arousal 0-1
                logger.info(f"Derived user input emotion (VADER via EmoCore): V={user_valence:.2f}, A={user_emotion_result[1]:.2f}")
            except Exception as emo_e:
                logger.error(f"Error during emotional analysis step in _handle_text_input: {emo_e}", exc_info=True)
                # Defaults will be used if these are None

        # --- 2. Memory Retrieval ---
        query_type = self._classify_query_type(user_input)
        max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7)
        initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes, query_type=query_type)
        initial_uuids = [uid for uid, score in initial_nodes]

        # retrieve_memory_chain now calculates and returns the 'effective_mood' that was used.
        memories_retrieved, effective_mood_for_retrieval = self.retrieve_memory_chain(
            initial_node_uuids=initial_uuids,
            recent_concept_uuids=list(self.last_interaction_concept_uuids),
            current_mood=mood_bias_for_retrieval # This is mood from *previous* interaction
        )
        logger.info(f"Retrieval used effective mood: V={effective_mood_for_retrieval[0]:.2f}, A={effective_mood_for_retrieval[1]:.2f}")

        # --- 3. Construct Prompt ---
        prompt = self._construct_prompt(
            user_input=user_input,
            conversation_history=conversation_history,
            memory_chain=memories_retrieved,
            tokenizer=self.tokenizer,
            max_context_tokens=self.config.get('prompting', {}).get('max_context_tokens', 4096),
            current_mood=effective_mood_for_retrieval, # Mood for the LLM prompt context
            emotional_instructions=emotional_instructions
        )

        # --- 4. Call LLM ---
        raw_llm_response = self._call_configured_llm('main_chat_text', prompt=prompt)

        # --- 5. Parse LLM Response ---
        inner_thoughts, final_response = self._parse_llm_response(raw_llm_response)

        # --- 6. Analyze AI Response Emotion (using EmotionalCore's VADER) ---
        if self.emotional_core and self.emotional_core.sentiment_analyzer and final_response:
            try:
                ai_scores = self.emotional_core._analyze_sentiment_vader(final_response) # Directly use EmoCore's method
                ai_valence = ai_scores.get("compound", 0.0)
                ai_arousal_simple = (ai_scores.get("pos", 0.0) + ai_scores.get("neg", 0.0)) * 0.5
                ai_emotion_result = (ai_valence, max(0.0, min(1.0, ai_arousal_simple))) # Clamp arousal
                logger.info(f"Derived AI response emotion (VADER via EmoCore): V={ai_valence:.2f}, A={ai_emotion_result[1]:.2f}")
            except Exception as ai_emo_e:
                logger.error(f"Error analyzing AI response emotion: {ai_emo_e}", exc_info=True)
                ai_emotion_result = None

        return inner_thoughts, final_response, memories_retrieved, user_emotion_result, ai_emotion_result, effective_mood_for_retrieval

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

    def _update_graph_and_context(self,
                                  graph_user_input_text_for_node: str,
                                  user_input_for_analysis: str,
                                  ai_response_text_for_node: str,
                                  user_emotion_values: tuple | None,
                                  ai_emotion_values: tuple | None,
                                  mood_used_for_current_retrieval: tuple | None,
                                  interaction_id_for_log: str,
                                  detected_interaction_type: str | None = None,
                                  detected_intent_details: dict | None = None,
                                  existing_user_node_uuid_if_retry: str | None = None
                                  ) -> tuple[str | None, str | None]:

        drive_cfg = self.config.get('subconscious_drives', {})
        logger.debug(f"_update_graph_and_context (ID: {interaction_id_for_log[:8]}): ExistingUserNode='{existing_user_node_uuid_if_retry[:8] if existing_user_node_uuid_if_retry else 'New'}'")

        user_node_uuid_to_use_for_linking = None # Initialize
        newly_created_user_node_uuid = None
        final_ai_node_uuid = None
        max_segment_len = self.config.get('activation', {}).get('conversational_segment_size', 5)

        # --- 1. Determine or Create User Node ---
        step_start_time_user_node = time.perf_counter()
        if existing_user_node_uuid_if_retry and existing_user_node_uuid_if_retry in self.graph:
            # IS a retry for a valid existing node.
            user_node_uuid_to_use_for_linking = existing_user_node_uuid_if_retry
            self.graph.nodes[user_node_uuid_to_use_for_linking]['last_accessed_ts'] = time.time()
            logger.info(f"Retry: Re-using user node {user_node_uuid_to_use_for_linking[:8]} and updating its access time.")
            log_profile_event(interaction_id_for_log, self.personality, "UGAC_AddUserNode_RetryReuse", 0.0, f"UserNode:{user_node_uuid_to_use_for_linking[:8]}")
        else:
            # NOT a retry OR the provided retry UUID was invalid/not found. Create a new user node.
            if existing_user_node_uuid_if_retry:
                logger.warning(f"Retry requested for node {existing_user_node_uuid_if_retry} but node not found in graph. Creating new user node instead.")

            user_v, user_a = user_emotion_values if user_emotion_values else (None, None)
            new_user_node_uuid = self.add_memory_node(
                text=graph_user_input_text_for_node,
                speaker="User",
                emotion_valence=user_v, emotion_arousal=user_a
            )
            user_node_uuid_to_use_for_linking = new_user_node_uuid # Use this new node
            newly_created_user_node_uuid = new_user_node_uuid # Mark it as newly created
            if user_node_uuid_to_use_for_linking:
                logger.debug(f"Added new user node {user_node_uuid_to_use_for_linking[:8]}")
            log_profile_event(interaction_id_for_log, self.personality, "UGAC_AddUserNode_New", time.perf_counter() - step_start_time_user_node, f"UserNode:{user_node_uuid_to_use_for_linking[:8] if user_node_uuid_to_use_for_linking else 'Fail'}")

        # --- Now user_node_uuid_to_use_for_linking is guaranteed to be assigned (or None if add_memory_node failed) ---
        # --- Perform actions associated with the user node ---
        if user_node_uuid_to_use_for_linking:
            # High impact check on the user node used for linking
            self._check_and_log_high_impact(user_node_uuid_to_use_for_linking)
            # Add to conversation segment
            self.current_conversational_segment_uuids.append(user_node_uuid_to_use_for_linking)
            if len(self.current_conversational_segment_uuids) > max_segment_len:
                self.current_conversational_segment_uuids.pop(0)

            # --- NLU Handling & Intention Storing (associated with the user node) ---
            step_start_time_nlu = time.perf_counter()
            boundary_node_added = False
            if detected_interaction_type == "topic_change":
                user_node_ts = self.graph.nodes[user_node_uuid_to_use_for_linking].get('timestamp', datetime.now(timezone.utc).isoformat())
                boundary_text = "CONVERSATION_BOUNDARY: Topic change detected by NLU."
                if detected_intent_details and "new_topic_keywords" in detected_intent_details:
                    keywords = detected_intent_details['new_topic_keywords']
                    if isinstance(keywords, list) and keywords: boundary_text += f" New topic hints: {', '.join(keywords)}"
                boundary_node_uuid = self.add_memory_node(boundary_text, "System", 'boundary', timestamp=user_node_ts)
                if boundary_node_uuid:
                    boundary_node_added = True
                    try:
                        self.graph.add_edge(user_node_uuid_to_use_for_linking, boundary_node_uuid, type='PRECEDES_BOUNDARY', base_strength=0.95, last_traversed_ts=time.time())
                    except Exception as link_e: logger.error(f"Failed to link user turn to boundary node: {link_e}")
                    self.current_conversational_segment_uuids = [] # Reset segment on topic change
            log_profile_event(interaction_id_for_log, self.personality, "UGAC_NLUHandling", time.perf_counter() - step_start_time_nlu, f"BoundaryAdded:{boundary_node_added}")

            step_start_time_intent = time.perf_counter()
            intention_node_added = False
            intention_analysis_result = self._analyze_intention_request(user_input_for_analysis)
            if intention_analysis_result.get("action") == "store_intention":
                intention_content = f"Remember: {intention_analysis_result['content']} (Trigger: {intention_analysis_result['trigger']})"
                intention_ts = self.graph.nodes[user_node_uuid_to_use_for_linking].get('timestamp', datetime.now(timezone.utc).isoformat())
                intention_node_uuid = self.add_memory_node(intention_content, "System", 'intention', timestamp=intention_ts)
                if intention_node_uuid:
                    intention_node_added = True
                    try:
                        self.graph.add_edge(user_node_uuid_to_use_for_linking, intention_node_uuid, type='GENERATED_INTENTION', base_strength=0.9, last_traversed_ts=time.time())
                        logger.info(f"Linked user turn {user_node_uuid_to_use_for_linking[:8]} to intention {intention_node_uuid[:8]}")
                    except Exception as link_e: logger.error(f"Failed to link user turn to intention node: {link_e}")
            log_profile_event(interaction_id_for_log, self.personality, "UGAC_StoreIntention", time.perf_counter() - step_start_time_intent, f"IntentionAdded:{intention_node_added}")

        # --- Add AI Node ---
        step_start_time_ai_node = time.perf_counter()
        if ai_response_text_for_node:
            ai_v, ai_a = ai_emotion_values if ai_emotion_values else (None, None)
            temp_last_added_node_for_linking = self.last_added_node_uuid # Store original last added

            # Set the node to link *from* for the temporal edge
            if user_node_uuid_to_use_for_linking: # Make sure we have a user node
                self.last_added_node_uuid = user_node_uuid_to_use_for_linking

            # Add the new AI node
            final_ai_node_uuid = self.add_memory_node(
                text=ai_response_text_for_node, speaker="AI",
                emotion_valence=ai_v, emotion_arousal=ai_a
            )

            # Restore last_added_node_uuid OR set it to the new AI node if successful
            if final_ai_node_uuid:
                self.last_added_node_uuid = final_ai_node_uuid # The new AI node is now the latest
                logger.debug(f"Added AI node {final_ai_node_uuid[:8]} linked from user node {user_node_uuid_to_use_for_linking[:8] if user_node_uuid_to_use_for_linking else 'N/A'}")
                self._check_and_log_high_impact(final_ai_node_uuid)
                self.current_conversational_segment_uuids.append(final_ai_node_uuid)
                if len(self.current_conversational_segment_uuids) > max_segment_len:
                    self.current_conversational_segment_uuids.pop(0)
            else:
                logger.warning("Failed to add AI node to graph.")
                self.last_added_node_uuid = temp_last_added_node_for_linking # Restore if AI add failed

        log_profile_event(interaction_id_for_log, self.personality, "UGAC_AddAiNode", time.perf_counter() - step_start_time_ai_node, f"AiNode:{final_ai_node_uuid[:8] if final_ai_node_uuid else 'Fail/None'}")

        # --- Update Context for Next Interaction ---
        step_start_time_next_ctx = time.perf_counter()
        mood_to_set_for_next_bias = mood_used_for_current_retrieval if mood_used_for_current_retrieval is not None else self.last_interaction_mood
        self._update_next_interaction_context(user_node_uuid_to_use_for_linking, final_ai_node_uuid, mood_to_set_for_next_bias)
        log_profile_event(interaction_id_for_log, self.personality, "UGAC_UpdateNextInteractionContext", time.perf_counter() - step_start_time_next_ctx)

        # --- Apply Heuristics ---
        # ... (heuristic logic as before, using user_input_for_analysis) ...
        step_start_time_heuristics = time.perf_counter()
        correction_keywords = self.config.get('conversation_heuristics_keywords', {}).get('correction', [])
        heuristic_applied = False
        if any(keyword in user_input_for_analysis.lower() for keyword in correction_keywords):
            if self._apply_st_drive_adjustment("Understanding", -0.05, "user_correction"):
                heuristic_applied = True
        log_profile_event(interaction_id_for_log, self.personality, "UGAC_ApplyHeuristics", time.perf_counter() - step_start_time_heuristics, f"CorrectionApplied:{heuristic_applied}")


        # --- Trigger Drive Update ---
        step_start_time_drive = time.perf_counter()
        context_for_drive_llm_update = ""
        if (drive_cfg.get('trigger_drive_update_on_high_impact', False) and self.high_impact_nodes_this_interaction) or \
                drive_cfg.get('llm_drive_update_every_interaction', False):
            user_part = user_input_for_analysis # Use the original user input
            ai_part = ai_response_text_for_node if final_ai_node_uuid else ""
            if user_part or ai_part:
                context_for_drive_llm_update = f"User: {user_part}\nAI: {ai_part}".strip()

        self._update_drive_state(context_text=context_for_drive_llm_update,
                                 user_node_uuid=user_node_uuid_to_use_for_linking, # User node associated with this interaction
                                 ai_node_uuid=final_ai_node_uuid, # AI node created in this interaction
                                 interaction_id_for_log=interaction_id_for_log)
        log_profile_event(interaction_id_for_log, self.personality, "UGAC_TriggerDriveUpdate", time.perf_counter() - step_start_time_drive, f"ContextLen:{len(context_for_drive_llm_update)}")

        # --- Update EmotionalCore Memory ---
        step_start_time_emocore = time.perf_counter()
        emocore_insight_updated = False
        if self.emotional_core and self.emotional_core.is_enabled:
            self.emotional_core.update_memory_with_emotional_insight()
            emocore_insight_updated = True
        log_profile_event(interaction_id_for_log, self.personality, "UGAC_EmoCoreUpdateMemory", time.perf_counter() - step_start_time_emocore, f"Updated:{emocore_insight_updated}")

        # --- Save Memory ---
        step_start_time_save = time.perf_counter()
        self._save_memory()
        log_profile_event(interaction_id_for_log, self.personality, "UGAC_SaveMemory", time.perf_counter() - step_start_time_save)

        # --- Return the correct UUIDs ---
        # User UUID: If a new one was created this turn, return that. Otherwise (retry), return the original one used.
        # AI UUID: Return the UUID of the AI node created in this turn (or None).
        return user_node_uuid_to_use_for_linking, final_ai_node_uuid

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
        logger.critical(f"@@@ _parse_llm_response INPUT: >>>{raw_response_text}<<<")
        inner_thoughts = None
        final_response = "" # Initialize to empty string

        if not raw_response_text:
            logger.warning("LLM response was empty in _parse_llm_response.")
            logger.critical(f"@@@ _parse_llm_response OUTPUT: thoughts=None, final_response='' (empty input)")
            return None, ""
        if not isinstance(raw_response_text, str):
            logger.error(f"LLM response not a string in _parse_llm_response! Type: {type(raw_response_text)}. Coercing.")
            raw_response_text = str(raw_response_text)

        logger.debug(f"PARSING LLM RESPONSE (coerced if needed): '''{raw_response_text}'''")

        try:
            # Find the last occurrence of "<thought>"
            last_thought_open_tag_match = None
            for match in re.finditer(r"<thought>", raw_response_text, re.IGNORECASE):
                last_thought_open_tag_match = match

            if last_thought_open_tag_match:
                # Search for the corresponding "</thought>" *after* this last opening tag
                start_search_for_closing_tag_from = last_thought_open_tag_match.end()
                closing_thought_match = re.search(r"(.*?)(</thought>)", raw_response_text[start_search_for_closing_tag_from:], re.DOTALL | re.IGNORECASE)

                if closing_thought_match:
                    # Thoughts are what's between the last <thought> and its </thought>
                    inner_thoughts = closing_thought_match.group(1).strip()
                    # Final response is everything after this specific </thought> block
                    final_response_start_index = start_search_for_closing_tag_from + closing_thought_match.end(2)
                    final_response = raw_response_text[final_response_start_index:].strip()
                    logger.debug(f"Extracted thoughts: '''{inner_thoughts}'''")
                    logger.debug(f"Initial final_response (after thought block): '''{final_response}'''")
                else:
                    # Found <thought> but no subsequent </thought>.
                    # This could mean the thought block is the rest of the string, or it's malformed.
                    # Safest to assume the AI intended the rest as thoughts if an open tag is present but unclosed.
                    # Or, if the model is *supposed* to always give a response after thoughts, treat it as an error.
                    # For now, let's assume if <thought> is present but unclosed, the rest is thought, and response is empty.
                    # This might need adjustment based on typical LLM failure modes.
                    # A more robust approach: if only <thought> is found, maybe the content after it IS the thought.
                    logger.warning(f"Found opening <thought> at {last_thought_open_tag_match.start()} but no closing </thought> found *after* it. Assuming text after open tag is thoughts, and no final response follows.")
                    inner_thoughts = raw_response_text[start_search_for_closing_tag_from:].strip()
                    final_response = "" # No clear final response if thought block isn't properly closed
            else:
                # No <thought> tags found at all, so everything is the final response
                logger.debug("No <thought> tags found. Entire input is final response.")
                final_response = raw_response_text.strip()
                inner_thoughts = None

        except Exception as parse_err:
            logger.error(f"Error parsing thoughts in _parse_llm_response: {parse_err}", exc_info=True)
            # Fallback: assume no thoughts, everything is response (or error message)
            inner_thoughts = None
            final_response = raw_response_text # Keep raw text if parsing fails badly

        # Ensure final_response is a string
        if final_response is None:
            final_response = ""

        # --- Aggressive Cleanup for both inner_thoughts and final_response ---
        # Remove any lingering thought tags from inner_thoughts
        if inner_thoughts:
            # Iteratively remove to handle nested or multiple stray tags within thoughts
            temp_thoughts = inner_thoughts
            while True:
                cleaned = re.sub(r"</?thought>", "", temp_thoughts, flags=re.IGNORECASE).strip()
                if cleaned == temp_thoughts:
                    break
                temp_thoughts = cleaned
            inner_thoughts = temp_thoughts
            if "<thought>" in inner_thoughts.lower() or "</thought>" in inner_thoughts.lower():
                 logger.warning(f"Post-cleanup, thought tags still detected in inner_thoughts: '''{inner_thoughts}'''")


        # Specifically remove any stray <thought> or </thought> tags from the final_response
        # This is crucial if the LLM mistakenly includes them.
        temp_final_response = final_response
        while True:
            # Remove a leading <thought> if it's not part of a proper block that was missed
            # Remove any </thought> tags, especially trailing ones
            cleaned = re.sub(r"^\s*<thought>\s*", "", temp_final_response, flags=re.IGNORECASE) # Leading open
            cleaned = re.sub(r"\s*</thought>\s*$", "", cleaned, flags=re.IGNORECASE) # Trailing close
            cleaned = re.sub(r"</?thought>", "", cleaned, flags=re.IGNORECASE).strip() # Any other stray tags

            if cleaned == temp_final_response:
                break
            temp_final_response = cleaned
        final_response = temp_final_response
        
        if "<thought>" in final_response.lower() or "</thought>" in final_response.lower():
            logger.warning(f"Post-cleanup, thought tags still detected in final_response: '''{final_response}'''")


        logger.critical(f"@@@ _parse_llm_response OUTPUT: thoughts='''{inner_thoughts}''', final_response='''{final_response}'''")
        return inner_thoughts, final_response

    def _check_for_action_request(self, response_text: str, user_input: str) -> tuple[str, bool, str | None]: # Modified return
        needs_planning = False
        cleaned_response_text = response_text 
        extracted_action_json_str = None # NEW

        if not isinstance(cleaned_response_text, str):
            cleaned_response_text = str(cleaned_response_text) if cleaned_response_text is not None else ""

        try:
            action_match = re.search(r'\[ACTION:\s*(\{.*?\})\s*\]$', cleaned_response_text, re.DOTALL | re.IGNORECASE)
            if action_match:
                extracted_action_json_str = action_match.group(1) # Capture the JSON
                logger.info(f"AI requested action detected via tag: {extracted_action_json_str}")
                cleaned_response_text = cleaned_response_text[:action_match.start()].strip()
                try:
                    action_data = json.loads(extracted_action_json_str)
                    if isinstance(action_data, dict) and "action" in action_data:
                        needs_planning = True # Still set needs_planning for the worker
                        logger.info("Setting needs_planning=True due to valid [ACTION:] tag.")
                    else:
                        logger.warning(f"ACTION tag found but content invalid: {extracted_action_json_str}")
                        extracted_action_json_str = None # Invalidate if not good JSON
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON in ACTION tag: {extracted_action_json_str}")
                    extracted_action_json_str = None # Invalidate
            # Keyword check for planning (can still run even if tag is present, as tag might be for direct exec)
            if not needs_planning: # Only set via keywords if tag didn't already
                user_input_lower = user_input.lower() if isinstance(user_input, str) else ""
                if any(keyword in user_input_lower for keyword in WORKSPACE_KEYWORDS):
                    needs_planning = True
                    logger.info(f"Potential workspace action detected via keywords in user input. Setting needs_planning=True.")
        except Exception as search_err:
            logger.error(f"Unexpected error during ACTION tag search: {search_err}", exc_info=True)
        
        return cleaned_response_text, needs_planning, extracted_action_json_str # Return the extracted JSON


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

    def plan_and_execute(self, user_input: str, conversation_history: list,
                         context_data: dict | None = None) -> list[tuple[bool, str, str, bool]]:
        """
        Plans and executes workspace actions based on user input and conversation context.
        Includes emotional state in the planning prompt.
        Can accept a predefined plan to bypass LLM planning.
        Returns list of tuples: (success, message, action_suffix, silent_and_successful)
        """
        task_id = str(uuid.uuid4())[:8] # Ensure uuid is imported
        logger.info(f"--- Starting Workspace Planning & Execution [ID: {task_id}] for input: '{strip_emojis(user_input[:50])}...' ---")
        workspace_action_results = []
        
        predefined_plan_json_str = None
        if context_data and isinstance(context_data, dict):
            predefined_plan_json_str = context_data.get('predefined_plan_json')

        parsed_plan = None 

        if predefined_plan_json_str:
            logger.info(f"[{task_id}] Using predefined plan from AI's previous action tag: {predefined_plan_json_str}")
            try:
                # Attempt to clean common LLM trailing comma issues in lists/objects
                cleaned_predefined_json_str = re.sub(r',\s*([\}\]])', r'\1', predefined_plan_json_str)
                if cleaned_predefined_json_str != predefined_plan_json_str:
                    logger.warning(f"[{task_id}] Cleaned trailing commas from predefined plan JSON.")
                
                action_data_from_tag = json.loads(cleaned_predefined_json_str)
                
                if isinstance(action_data_from_tag, dict) and "action" in action_data_from_tag:
                    parsed_plan = [action_data_from_tag] 
                    logger.info(f"[{task_id}] Predefined single action plan parsed: {parsed_plan}")
                elif isinstance(action_data_from_tag, list): 
                    parsed_plan = action_data_from_tag
                    logger.info(f"[{task_id}] Predefined plan (list) parsed: {parsed_plan}")
                else:
                    logger.error(f"[{task_id}] Predefined plan JSON was not a valid action object or list: {predefined_plan_json_str}")
                    workspace_action_results.append((False, "Internal Error: Invalid predefined plan format.", "planning_predefined_error", False))
                    return workspace_action_results
            except json.JSONDecodeError as e:
                logger.error(f"[{task_id}] Failed to parse predefined plan JSON: {e}. Raw: '{predefined_plan_json_str}'")
                workspace_action_results.append((False, f"Internal Error: Predefined plan JSON parse error: {e}", "planning_predefined_json_error", False))
                return workspace_action_results
        
        if parsed_plan is None: # Only call planning LLM if no valid predefined plan was used
            logger.info(f"[{task_id}] No valid predefined plan. Proceeding with LLM-based planning.")
            try:
                # 1. Retrieve Relevant Memories
                logger.debug(f"[{task_id}] Retrieving memories for planning context...")
                query_type = self._classify_query_type(user_input)
                max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7)
                initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes, query_type=query_type)
                initial_uuids = [uid for uid, score in initial_nodes]
                mood_for_retrieval = self.last_interaction_mood
                retrieved_nodes, effective_mood_for_retrieval = self.retrieve_memory_chain(
                    initial_node_uuids=initial_uuids,
                    recent_concept_uuids=list(self.last_interaction_concept_uuids),
                    current_mood=mood_for_retrieval
                )
                memory_chain_data = retrieved_nodes
                logger.info(f"[{task_id}] Retrieved {len(memory_chain_data)} memories for planning (Effective Mood: {effective_mood_for_retrieval}).")

                # 2. Prepare Planning Prompt Context
                logger.debug(f"[{task_id}] Preparing context for planning prompt...")
                planning_history_text = "\n".join([f"{turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}" for turn in conversation_history[-5:]])
                planning_memory_text = "\n".join([f"- {mem.get('speaker', '?')} ({self._get_relative_time_desc(mem.get('timestamp',''))}): {strip_emojis(mem.get('text', ''))}" for mem in memory_chain_data])
                if not planning_memory_text: planning_memory_text = "[No relevant memories retrieved]"

                workspace_files_list_str = "[Workspace State: Not Indexed or Empty]"
                if hasattr(self, 'workspace_index_filepath') and os.path.exists(self.workspace_index_filepath):
                    summaries_for_prompt = []
                    try:
                        with open(self.workspace_index_filepath, 'r', encoding='utf-8') as f_idx:
                            # Sort entries by modified_ts (desc) to get recent files first.
                            # This requires loading all entries, which might be inefficient for huge indexes.
                            # For now, let's just read top N lines for simplicity if index is large.
                            temp_entries = []
                            for line in f_idx:
                                try: temp_entries.append(json.loads(line))
                                except: pass # Skip malformed lines
                            
                            # Sort by modified_ts if present, otherwise by created_ts
                            temp_entries.sort(key=lambda x: x.get('modified_ts', x.get('created_ts', '')), reverse=True)

                            max_files_in_prompt = self.config.get('prompting', {}).get('max_files_to_summarize_in_context', 5)
                            for entry in temp_entries[:max_files_in_prompt]:
                                summaries_for_prompt.append(
                                    f"Filename: {entry.get('filename', '?')}\n  Modified: {self._get_relative_time_desc(entry.get('modified_ts', ''))}\n  Summary: {entry.get('llm_summary', '[No Summary]')}\n  Keywords: {', '.join(entry.get('keywords',[]))}"
                                )
                        if summaries_for_prompt:
                            workspace_files_list_str = "\n---\n".join(summaries_for_prompt)
                        elif os.path.exists(self.workspace_path): # Fallback if index is empty but workspace exists
                            raw_files, _ = file_manager.list_files(self.config, self.personality)
                            if raw_files: workspace_files_list_str = "[Workspace Files (No Summaries from Index)]:\n" + "\n".join([f"- {fname}" for fname in sorted(raw_files)])
                            else: workspace_files_list_str = "[Workspace is empty (checked file system)]"

                    except Exception as e_idx:
                        logger.error(f"Error reading workspace index for planning: {e_idx}")
                        workspace_files_list_str = "[Error reading workspace index]"
                elif os.path.exists(self.workspace_path): # Fallback if index file path itself doesn't exist
                    raw_files, _ = file_manager.list_files(self.config, self.personality)
                    if raw_files: workspace_files_list_str = "[Workspace Files (Index Missing)]:\n" + "\n".join([f"- {fname}" for fname in sorted(raw_files)])
                    else: workspace_files_list_str = "[Workspace is empty (Index Missing)]"


                asm_context_str = "[AI Self-Model: Not Available]"
                if self.autobiographical_model:
                    try:
                        asm_parts = ["[My Self-Perception:]"]
                        if self.autobiographical_model.get("summary_statement"): asm_parts.append(f"- Summary: {self.autobiographical_model['summary_statement']}")
                        if self.autobiographical_model.get("core_traits"): asm_parts.append(f"- Traits: {', '.join(self.autobiographical_model['core_traits'])}")
                        if self.autobiographical_model.get("goals_motivations"): asm_parts.append(f"- Goals/Motivations: {', '.join(self.autobiographical_model['goals_motivations'])}")
                        if self.autobiographical_model.get("relational_stance"): asm_parts.append(f"- My Role: {self.autobiographical_model['relational_stance']}")
                        if self.autobiographical_model.get("emotional_profile"): asm_parts.append(f"- Emotional Profile: {self.autobiographical_model['emotional_profile']}")
                        if len(asm_parts) > 1: asm_context_str = "\n".join(asm_parts)
                    except Exception as asm_fmt_e: logger.error(f"Error formatting ASM for planning prompt: {asm_fmt_e}"); asm_context_str = "[AI Self-Model: Error Formatting]"

                emotional_context_str = "[AI Emotional State: Neutral]"
                if self.emotional_core and self.emotional_core.is_enabled:
                    tendency = self.emotional_core.derived_tendency
                    mood_hints = self.emotional_core.derived_mood_hints 
                    drive_deviations_parts = []
                    base_drives = self.config.get('subconscious_drives', {}).get('base_drives', {})
                    long_term_influence = self.config.get('subconscious_drives', {}).get('long_term_influence_on_baseline', 1.0)
                    st_drives = self.drive_state.get("short_term", {})
                    lt_drives = self.drive_state.get("long_term", {})
                    for drive, st_level in st_drives.items():
                        config_baseline = base_drives.get(drive, 0.0)
                        lt_level = lt_drives.get(drive, 0.0)
                        dynamic_baseline = config_baseline + (lt_level * long_term_influence)
                        deviation = st_level - dynamic_baseline
                        if abs(deviation) > 0.15: 
                            drive_deviations_parts.append(f"{drive} dev: {deviation:+.1f}")
                    drive_dev_str = f"; Drives: ({', '.join(drive_deviations_parts)})" if drive_deviations_parts else ""
                    emo_parts = [f"Tendency: {tendency} (Mood V: {mood_hints.get('valence', 0):.1f}, A: {mood_hints.get('arousal', 0):.1f}{drive_dev_str})"]
                    triggered_fears_strong = {f: d.get('rationale', '') for f, d in self.emotional_core.current_analysis_results.get("triggered_fears", {}).items() if d.get("confidence", 0) > 0.65}
                    triggered_needs_strong = {n: d.get('rationale', '') for n, d in self.emotional_core.current_analysis_results.get("triggered_needs", {}).items() if d.get("confidence", 0) > 0.65}
                    if triggered_fears_strong: emo_parts.append(f"Strong Fears: {', '.join(triggered_fears_strong.keys())}")
                    if triggered_needs_strong: emo_parts.append(f"Strong Needs: {', '.join(triggered_needs_strong.keys())}")
                    emotional_context_str = f"[AI Emotional State: {'; '.join(emo_parts)}]"

                # Populate ai_suggested_action_json and persona_workspace_preferences
                ai_suggested_action_json_value = predefined_plan_json_str if predefined_plan_json_str else "None" 
                
                persona_prefs_string_value = "Default Preferences"
                if hasattr(self, 'workspace_persona_prefs') and self.workspace_persona_prefs:
                    prefs_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in self.workspace_persona_prefs.items()]
                    persona_prefs_string_value = ", ".join(prefs_parts) if prefs_parts else "Default"
                
                planning_prompt_template = self._load_prompt("workspace_planning_prompt.txt")
                if not planning_prompt_template:
                    logger.error(f"[{task_id}] Workspace planning prompt template missing.")
                    workspace_action_results.append((False, "Internal Error: Planning prompt missing.", "planning_error", False))
                    return workspace_action_results

                try:
                    planning_prompt = planning_prompt_template 
                    replacements = {
                        "{user_request}": user_input, "{history_text}": planning_history_text,
                        "{memory_text}": planning_memory_text, "{workspace_files_list}": workspace_files_list_str,
                        "{asm_context}": asm_context_str, "{emotional_context}": emotional_context_str,
                        "{ai_suggested_action_json}": ai_suggested_action_json_value,
                        "{persona_workspace_preferences}": persona_prefs_string_value
                    }
                    for placeholder, value in replacements.items():
                        str_value = str(value) if value is not None else "" 
                        planning_prompt = planning_prompt.replace(placeholder, str_value)
                    
                    remaining_placeholders = re.findall(r'\{[a-zA-Z0-9_]+\}', planning_prompt)
                    if remaining_placeholders: 
                        raise ValueError(f"Unreplaced placeholders found: {remaining_placeholders}")

                    logger.info(f"[{task_id}] Sending workspace planning prompt to LLM...")
                    logger.debug(f"[{task_id}] Planning Prompt Preview (First 500 chars):\n{strip_emojis(planning_prompt[:500])}...")

                except Exception as replace_e: 
                    logger.error(f"[{task_id}] Error during planning prompt construction: {replace_e}", exc_info=True)
                    workspace_action_results.append((False, f"Internal Error: Planning prompt construction: {replace_e}", "planning_format_error", False))
                    return workspace_action_results

                plan_response_str = self._call_configured_llm('workspace_planning', prompt=planning_prompt)
                
                logger.debug(f"[{task_id}] Raw planning LLM response: ```{plan_response_str}```")
                if plan_response_str and not plan_response_str.startswith("Error:"):
                    try:
                        md_json_match = re.search(r"```json\s*(\[.*?\])\s*```", plan_response_str, re.DOTALL | re.MULTILINE)
                        json_str_to_parse = None
                        if md_json_match:
                            json_str_to_parse = md_json_match.group(1)
                        else:
                            start_bracket = plan_response_str.find('[')
                            end_bracket = plan_response_str.rfind(']')
                            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                                json_str_to_parse = plan_response_str[start_bracket : end_bracket + 1]
                        
                        if json_str_to_parse:
                            logger.debug(f"[{task_id}] Extracted plan JSON string: {json_str_to_parse}")
                            cleaned_json_str = re.sub(r',\s*([\}\]])', r'\1', json_str_to_parse)
                            if cleaned_json_str != json_str_to_parse:
                                logger.warning(f"[{task_id}] Cleaned trailing commas from plan JSON.")
                            
                            parsed_plan = json.loads(cleaned_json_str)
                            if not isinstance(parsed_plan, list):
                                logger.error(f"[{task_id}] Parsed plan is not a list: {type(parsed_plan)}")
                                parsed_plan = None 
                            else:
                                logger.info(f"[{task_id}] Successfully parsed plan with {len(parsed_plan)} steps.")
                        else:
                            logger.warning(f"[{task_id}] Could not extract JSON list '[]' from planning response. Assuming no plan.")
                            parsed_plan = [] 
                    except json.JSONDecodeError as e:
                        json_context_for_error = cleaned_json_str if 'cleaned_json_str' in locals() else \
                                                 json_str_to_parse if 'json_str_to_parse' in locals() else \
                                                 plan_response_str
                        logger.error(f"[{task_id}] Failed to parse JSON plan: {e}. String tried: '{strip_emojis(json_context_for_error[:200])}...'. Raw: '{strip_emojis(plan_response_str[:200])}...'")
                        parsed_plan = None
                elif plan_response_str.startswith("Error:"):
                    logger.error(f"[{task_id}] Workspace planning LLM call failed: {plan_response_str}")
                    workspace_action_results.append((False, f"Planning Error: {plan_response_str}", "planning_llm_error", False))
                    return workspace_action_results
                else:
                    logger.warning(f"[{task_id}] Workspace planning LLM returned empty response. Assuming no plan.")
                    parsed_plan = []
            
            except Exception as planning_llm_phase_e:
                logger.error(f"[{task_id}] Error during LLM planning phase: {planning_llm_phase_e}", exc_info=True)
                workspace_action_results.append((False, f"Internal Error during planning: {planning_llm_phase_e}", "planning_phase_exception", False))
                return workspace_action_results
        
        # 5. Execute Plan
        if parsed_plan is not None:
            if isinstance(parsed_plan, list) and len(parsed_plan) > 0:
                logger.info(f"[{task_id}] Plan contains {len(parsed_plan)} step(s). Instantiating WorkspaceAgent...")
                try:
                    agent = WorkspaceAgent(self) 
                    logger.info(f"[{task_id}] Calling WorkspaceAgent.execute_plan...")
                    workspace_action_results = agent.execute_plan(parsed_plan)
                    logger.info(f"[{task_id}] WorkspaceAgent execution finished. Results count: {len(workspace_action_results)}")
                except Exception as agent_exec_e:
                    logger.error(f"[{task_id}] Error during WorkspaceAgent execution: {agent_exec_e}", exc_info=True)
                    workspace_action_results.append((False, f"Internal Error: Agent execution failed: {agent_exec_e}", "agent_execution_error", False))
            elif isinstance(parsed_plan, list) and len(parsed_plan) == 0:
                logger.info(f"[{task_id}] Plan was empty. No workspace actions executed.")
            else: 
                logger.error(f"[{task_id}] Parsed plan was invalid (not a list). No actions executed.")
                workspace_action_results.append((False, "Internal Error: Invalid plan structure after processing.", "planning_invalid_final_plan", False))
        else: 
            logger.warning(f"[{task_id}] No valid workspace plan available. No actions executed.")
            # Only add error if LLM planning was attempted and failed, and no predefined plan was used.
            if not predefined_plan_json_str and (not 'plan_response_str' in locals() or (plan_response_str and not plan_response_str.startswith("Error:"))):
                 workspace_action_results.append((False, "Planning Error: Could not parse plan from LLM.", "planning_parse_fail", False))
        
        logger.info(f"--- Workspace Planning & Execution [ID: {task_id}] Finished. Results: {len(workspace_action_results)} ---")
        return workspace_action_results
    
    def _analyze_interaction_type(self, user_input: str, conversation_history_text: str, interaction_id_for_log: str) -> tuple[str | None, dict | None]:
        """
        Uses LLM to determine the type of user interaction.
        interaction_id_for_log is passed for profiling.
        """
        step_start_time = time.perf_counter()
        detected_type = "unknown_error" # Default in case of early exit
        details = None

        if not self.config.get('features', {}).get('enable_nlu_interaction_typing', False):
            logger.debug("NLU interaction typing disabled. Defaulting type.")
            duration = time.perf_counter() - step_start_time
            log_profile_event(interaction_id_for_log, self.personality, "NLU_InteractionType_Disabled", duration)
            return "unknown_disabled", None

        logger.debug(f"Analyzing interaction type for: '{strip_emojis(user_input[:100])}...'")
        prompt_template = self._load_prompt("interaction_type_prompt.txt")
        if not prompt_template:
            logger.error("Failed to load interaction_type_prompt.txt. Cannot determine interaction type.")
            duration = time.perf_counter() - step_start_time
            log_profile_event(interaction_id_for_log, self.personality, "NLU_InteractionType_PromptFail", duration)
            return None, {"error_detail": "Prompt template missing"}

        history_snippet = "\n".join(conversation_history_text.splitlines()[-5:])
        try:
            full_prompt = prompt_template.format(user_input=user_input, history_snippet=history_snippet)
        except KeyError as e:
            logger.error(f"Missing placeholder in interaction_type_prompt.txt: {e}.")
            duration = time.perf_counter() - step_start_time
            log_profile_event(interaction_id_for_log, self.personality, "NLU_InteractionType_FormatError", duration, f"KeyError:{e}")
            return "unknown_prompt_error", {"error_detail": f"Missing placeholder: {e}"}

        llm_task_name = "interaction_type_analysis"
        llm_response_str = self._call_configured_llm(llm_task_name, prompt=full_prompt) # _call_configured_llm will do its own profiling

        if not llm_response_str or llm_response_str.startswith("Error:"):
            logger.error(f"Interaction type analysis LLM call failed or returned error: {llm_response_str}")
            duration = time.perf_counter() - step_start_time
            log_profile_event(interaction_id_for_log, self.personality, "NLU_InteractionType_LLMError", duration, f"LLMError:{llm_response_str[:50]}")
            return None, {"error_detail": llm_response_str or "Empty LLM response"}

        try:
            # ... (your existing robust JSON parsing logic for interaction type) ...
            # For brevity, assuming it results in:
            # detected_type = parsed_data.get("interaction_type", "unknown_parse")
            # details = parsed_data.get("details")
            # Replace with your actual parsing logic:
            match = re.search(r'(\{.*\})', llm_response_str, re.DOTALL) 
            json_str = ""
            if match:
                md_json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_str, re.DOTALL)
                if md_json_match: json_str = md_json_match.group(1)
                else: json_str = match.group(0)
            
            if json_str:
                parsed_data = json.loads(json_str)
                detected_type = parsed_data.get("interaction_type", "unknown_json_key")
                details = parsed_data.get("details")
                valid_types = ["question", "statement", "greeting", "farewell", 
                               "topic_change", "memory_modification_request", 
                               "workspace_action_request", "clarification_response", 
                               "feedback", "command", "other", "unknown",
                               "unknown_prompt_error", "unknown_parse_fail", "unknown_json_error", "unknown_json_key", "unknown_exception"]
                if not (detected_type and isinstance(detected_type, str) and detected_type in valid_types):
                    logger.warning(f"LLM response for interaction type '{detected_type}' not in expected. Parsed: {parsed_data}")
                    details = {"original_llm_type": detected_type, **(details if isinstance(details, dict) else {})}
                    detected_type = "other_unexpected_type"

            else: # No JSON object found
                simple_class = llm_response_str.strip().lower().replace(" ", "_")
                valid_simple_types = ["question", "statement", "topic_change", "greeting", "farewell", "command", "other"]
                if simple_class in valid_simple_types and len(simple_class) < 30:
                    detected_type = simple_class
                    details = None
                else:
                    detected_type = "unknown_parse_fail"
                    details = {"error_detail": "No JSON object found in response", "raw_response": llm_response_str[:100]}

            logger.info(f"NLU detected interaction type: '{detected_type}', Details: {details}")
            
        except Exception as e:
            logger.error(f"Error parsing/validating interaction type LLM response: {e}", exc_info=True)
            detected_type = "unknown_exception_parsing"
            details = {"error_detail": str(e), "raw_response": llm_response_str[:100]}
        
        duration = time.perf_counter() - step_start_time
        log_profile_event(interaction_id_for_log, self.personality, "NLU_InteractionTypeAnalysis_Total", duration, f"FinalType:{detected_type}")
        return detected_type, details

    def process_interaction(self, user_input: str, conversation_history: list,
                        attachment_data: dict | None = None,
                        is_retry: bool = False, # NEW
                        ai_node_to_replace: str | None = None, # NEW
                        user_node_uuid_for_retry: str | None = None # NEW - Added user UUID for retry context
                       ) -> InteractionResult: #<<< Return Type Hint added
        interaction_id = str(uuid.uuid4())
        self.current_interaction_id = interaction_id # Store for use by sub-methods for profiling

        overall_start_time = time.perf_counter()
        logger.info(f"--- Processing Interaction START (ID: {interaction_id[:8]}) ---")
        logger.info(f"Input='{strip_emojis(user_input[:60])}...', Attachment: {attachment_data.get('type') if attachment_data else 'No'}, Retry={is_retry}, ReplaceNode={ai_node_to_replace[:8] if ai_node_to_replace else 'N/A'}, UserNodeForRetry={user_node_uuid_for_retry[:8] if user_node_uuid_for_retry else 'N/A'}") # Log retry info
        self.high_impact_nodes_this_interaction.clear() # Reset for this interaction

        # Default error result object
        error_result = InteractionResult(final_response_text="Error: Processing failed unexpectedly.")

        log_tuning_event("INTERACTION_START", {
            "interaction_id": interaction_id,
            "personality": self.personality,
            "user_input_preview": strip_emojis(user_input[:100]),
            "has_attachment": bool(attachment_data),
            "attachment_type": attachment_data.get('type') if attachment_data else None,
            "history_length": len(conversation_history),
            "is_retry": is_retry,
            "ai_node_to_replace": ai_node_to_replace,
            "user_node_uuid_for_retry": user_node_uuid_for_retry
        })

        if not hasattr(self, 'embedder') or self.embedder is None:
            logger.critical("PROCESS_INTERACTION CRITICAL ERROR: Embedder not initialized!")
            log_profile_event(interaction_id, self.personality, "ProcessInteraction_Fail_Embedder", time.perf_counter() - overall_start_time, "Embedder not initialized")
            log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "embedder_check", "error": "Embedder not initialized" })
            return error_result # Returns InteractionResult

        try:
            # Determine text to save in the graph for the user turn (includes attachment placeholder)
            graph_user_input = user_input
            if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('filename'):
                placeholder = f" [Image Attached: {attachment_data['filename']}]"
                separator = " " if graph_user_input else ""
                graph_user_input += separator + placeholder

            # --- Step 0: NLU for Interaction Type ---
            history_text_for_nlu = "\n".join([f"{turn.get('speaker', '?')}: {strip_emojis(turn.get('text', ''))}"
                                             for turn in conversation_history[-3:]])
            # _analyze_interaction_type does its own internal profiling now, including the interaction_id
            nlu_type_from_analysis, nlu_details_from_analysis = self._analyze_interaction_type(
                user_input, history_text_for_nlu, interaction_id)
            current_nlu_type = nlu_type_from_analysis if nlu_type_from_analysis is not None else "unknown"
            current_nlu_details = nlu_details_from_analysis
            logger.info(f"Interaction {interaction_id[:8]}: NLU analysis result - Type: '{current_nlu_type}', Details: {current_nlu_details}")


            # --- Handle Retry Deletion ---
            deleted_node_for_retry = False
            if is_retry and ai_node_to_replace:
                logger.info(f"Retry requested. Attempting to delete previous AI node: {ai_node_to_replace}")
                delete_success = self.delete_memory_entry(ai_node_to_replace)
                if delete_success:
                    logger.info(f"Successfully deleted previous AI node {ai_node_to_replace} for retry.")
                    deleted_node_for_retry = True
                    # NOTE: conversation_history is a *copy* passed in. Modifying it here
                    # only affects the copy used within this function. The worker thread
                    # needs to handle removing it from *its* history separately if needed
                    # before passing it to the LLM (which handle_chat_task now does).
                else:
                    logger.warning(f"Failed to delete previous AI node {ai_node_to_replace} for retry. Proceeding with retry anyway.")
            # --------------------------

            # --- Step 1: Handle Input & Call LLM ---
            # This function now returns: (inner_thoughts, raw_llm_response, memories_retrieved, user_emotion, ai_emotion, effective_mood_for_retrieval)
            # We expect _handle_input to always return the tuple, even if response is an error string.
            handle_input_result = self._handle_input(
                interaction_id, user_input, conversation_history, attachment_data
            )

            # <<< Defensive Check: Ensure _handle_input returned the expected tuple >>>
            if not isinstance(handle_input_result, tuple) or len(handle_input_result) != 6:
                logger.critical(f"Interaction {interaction_id[:8]}: _handle_input returned unexpected type/length! Got: {type(handle_input_result)}. Expected 6-tuple.")
                error_result.final_response_text = "Error: Internal failure during input handling."
                log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "handle_input_return_check", "error": "Invalid return type from _handle_input" })
                log_profile_event(interaction_id, self.personality, "ProcessInteraction_Fail_HandleInputReturn", time.perf_counter() - overall_start_time, f"ReturnType:{type(handle_input_result)}")
                return error_result

            inner_thoughts, raw_llm_response_text, memories_retrieved, user_emotion, ai_emotion, effective_mood_retrieval_for_this_turn = handle_input_result

            # Check for critical LLM call failure indicated by the response text
            if raw_llm_response_text is None or "Error:" in raw_llm_response_text[:20]:
                 logger.error(f"Interaction {interaction_id[:8]}: LLM call failed or returned error: '{raw_llm_response_text}'")
                 log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "llm_call", "error": raw_llm_response_text or "Empty LLM response" })
                 error_result.final_response_text = raw_llm_response_text or "Error: LLM processing failed."
                 log_profile_event(interaction_id, self.personality, "ProcessInteraction_Fail_LLM", time.perf_counter() - overall_start_time, f"LLMError:{str(raw_llm_response_text)[:50]}")
                 return error_result # Return InteractionResult object

            final_response_text_from_llm = raw_llm_response_text # It's already parsed/cleaned

            # --- Step 2: Check for [ACTION:] tag ---
            final_response_text_cleaned_after_action_check, needs_overall_planning, extracted_tag_json = self._check_for_action_request(
                response_text=final_response_text_from_llm,
                user_input=user_input
            )
            # Determine overall planning need (incorporating NLU, keywords if needed)
            if not needs_overall_planning: # Only check further if tag didn't already trigger it
                if current_nlu_type == "workspace_action_request":
                    logger.info(f"Interaction {interaction_id[:8]}: NLU detected 'workspace_action_request'. Setting needs_planning=True.")
                    needs_overall_planning = True
                elif any(keyword in user_input.lower() for keyword in WORKSPACE_KEYWORDS): # Fallback to keywords
                    logger.info(f"Interaction {interaction_id[:8]}: Workspace keywords found in user input. Setting needs_planning=True.")
                    needs_overall_planning = True

            logger.debug(f"Interaction {interaction_id[:8]}: Needs Planning Flag = {needs_overall_planning}. Cleaned Response: '{final_response_text_cleaned_after_action_check[:60]}...'")


            # --- Step 3: Update Graph & Context ---
            # Pass the user node UUID if this is a retry attempt for that specific user input
            user_node_uuid, ai_node_uuid = self._update_graph_and_context(
                graph_user_input_text_for_node=graph_user_input, # Use the version with attachment placeholder
                user_input_for_analysis=user_input, # Original input for analysis
                ai_response_text_for_node=final_response_text_cleaned_after_action_check, # Cleaned response for node
                user_emotion_values=user_emotion,
                ai_emotion_values=ai_emotion,
                mood_used_for_current_retrieval=effective_mood_retrieval_for_this_turn,
                interaction_id_for_log=interaction_id,
                detected_interaction_type=current_nlu_type,
                detected_intent_details=current_nlu_details,
                existing_user_node_uuid_if_retry=user_node_uuid_for_retry # Pass the user node for retry case
            )

            # --- CRITICAL CHECK: Graph Update Failure ---
            # If it's NOT a retry AND we failed to get/create a user node UUID, it's a critical failure.
            if user_node_uuid is None and not is_retry:
                logger.error(f"Interaction {interaction_id[:8]}: Failed to add/find user node during graph update. Cannot proceed.")
                error_result.final_response_text = "Error: Failed to store user input in memory."
                log_tuning_event("INTERACTION_ERROR", { "interaction_id": interaction_id, "personality": self.personality, "stage": "graph_update_user_node", "error": "Failed to add/find user node" })
                log_profile_event(interaction_id, self.personality, "ProcessInteraction_Fail_UserNodeAdd", time.perf_counter() - overall_start_time)
                return error_result # Return InteractionResult object

            # If AI node add failed, log it but proceed (AI UUID will be None in result)
            if ai_node_uuid is None and final_response_text_cleaned_after_action_check:
                 logger.warning(f"Interaction {interaction_id[:8]}: AI response generated but failed to add AI node to graph.")
                 # The InteractionResult will correctly have ai_node_uuid=None

            # --- Step 4: Assemble and Return Result ---
            final_result = InteractionResult(
                final_response_text=final_response_text_cleaned_after_action_check,
                inner_thoughts=inner_thoughts,
                memories_used=memories_retrieved,
                user_node_uuid=user_node_uuid, # Use potentially None values if add failed
                ai_node_uuid=ai_node_uuid,
                needs_planning=needs_overall_planning,
                detected_interaction_type=current_nlu_type,
                detected_intent_details=current_nlu_details,
                extracted_ai_action_tag_json=extracted_tag_json
            )

            # ... (log tuning event for end) ...
            log_profile_event(interaction_id, self.personality, "ProcessInteraction_Success", time.perf_counter() - overall_start_time)
            logger.info(f"--- Processing Interaction END (ID: {interaction_id[:8]}) ---")
            return final_result # Returns InteractionResult object

        except Exception as e:
            logger.error(f"--- CRITICAL Outer Error during process_interaction (ID: {interaction_id[:8]}) ---", exc_info=True)
            # Log tuning event for error
            log_tuning_event("INTERACTION_ERROR", {
                "interaction_id": interaction_id,
                "personality": self.personality,
                "stage": "outer_exception_handler",
                "error": str(e),
                "error_type": type(e).__name__
            })
            log_profile_event(interaction_id, self.personality, "ProcessInteraction_OuterException", time.perf_counter() - overall_start_time, f"ErrorType:{type(e).__name__}")
            error_result.final_response_text = f"Error: Processing failed unexpectedly in main loop. Details: {type(e).__name__}"
            return error_result # Ensure InteractionResult is returned on outer exception
        finally:
            # Clean up self.current_interaction_id if you set it
            if hasattr(self, 'current_interaction_id'):
                del self.current_interaction_id
    
    
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

    def _update_workspace_index(self, filename: str, content_snippet: str, is_append: bool = False):
        """
        Generates metadata for a file and appends/updates it in the workspace index.
        """
        if not self.workspace_index_filepath:
            logger.error("Workspace index filepath not configured. Cannot update index.")
            return

        logger.info(f"Updating workspace index for file: {filename} (Append: {is_append})")

        # 1. Get basic file stats
        full_file_path = os.path.join(self.workspace_path, filename)
        file_format = os.path.splitext(filename)[1].lower()
        created_ts_iso = ""
        modified_ts_iso = ""
        size_kb = 0.0

        if os.path.exists(full_file_path):
            try:
                stat_info = os.stat(full_file_path)
                created_ts_iso = datetime.fromtimestamp(stat_info.st_ctime, tz=timezone.utc).isoformat()
                modified_ts_iso = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat()
                size_kb = round(stat_info.st_size / 1024, 2)
            except Exception as e:
                logger.error(f"Could not get stats for file {filename}: {e}")
        else:
            # File might have just been created, use current time
            now_iso = datetime.now(timezone.utc).isoformat()
            created_ts_iso = now_iso
            modified_ts_iso = now_iso
            # Size might be 0 if just created and empty, or we use len of snippet as rough guide
            size_kb = round(len(content_snippet) / 1024, 2) if content_snippet else 0.0


        # 2. Call LLM to generate summary, keywords, relevance
        # Prepare prompt for metadata generation
        prompt_template = self._load_prompt("generate_file_metadata_prompt.txt")
        if not prompt_template:
            logger.error("generate_file_metadata_prompt.txt not found. Skipping LLM metadata.")
            llm_summary = "[Summary generation failed: prompt missing]"
            keywords = []
            estimated_relevance_score = 0.1 # Low default
        else:
            metadata_prompt = prompt_template.format(
                filename=filename,
                content_snippet=content_snippet[:1000] # Limit snippet length for LLM
            )
            logger.debug(f"Sending metadata generation prompt for {filename}...")
            llm_response = self._call_configured_llm('generate_file_metadata', prompt=metadata_prompt)

            if llm_response and not llm_response.startswith("Error:"):
                try:
                    # Expecting JSON: {"summary": "...", "keywords": ["k1", "k2"], "relevance": 0.X}
                    parsed_meta = json.loads(llm_response)
                    llm_summary = parsed_meta.get("summary", "").strip()
                    keywords = parsed_meta.get("keywords", [])
                    if isinstance(keywords, str): # Handle if LLM returns comma-sep string
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    elif not isinstance(keywords, list):
                        keywords = []
                    keywords = [k for k in keywords if isinstance(k, str)] # Ensure all are strings

                    relevance_score_raw = parsed_meta.get("relevance", 0.1)
                    if isinstance(relevance_score_raw, (int, float)):
                        estimated_relevance_score = max(0.0, min(1.0, float(relevance_score_raw)))
                    else:
                        estimated_relevance_score = 0.1 # Default if invalid
                    logger.info(f"LLM Metadata for {filename}: Summary='{llm_summary[:50]}...', Keywords={keywords}, Relevance={estimated_relevance_score}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON metadata from LLM for {filename}. Raw: {llm_response}")
                    llm_summary = "[Summary generation failed: parse error]"
                    keywords = []
                    estimated_relevance_score = 0.1
                except Exception as e:
                    logger.error(f"Error processing LLM metadata for {filename}: {e}")
                    llm_summary = f"[Summary generation error: {type(e).__name__}]"
                    keywords = []
                    estimated_relevance_score = 0.1
            else:
                logger.error(f"LLM call failed for metadata generation for {filename}: {llm_response}")
                llm_summary = "[Summary generation failed: LLM error]"
                keywords = []
                estimated_relevance_score = 0.1

        # 3. Construct metadata entry
        metadata_entry = {
            "filename": filename,
            "created_ts": created_ts_iso,
            "modified_ts": modified_ts_iso,
            "size_kb": size_kb,
            "file_format": file_format,
            "llm_summary": llm_summary,
            "keywords": keywords,
            "estimated_relevance_score": estimated_relevance_score,
            "archived": False # New files are not archived
        }

        # 4. Append/Update index file
        try:
            updated_entries = []
            entry_found_and_updated = False
            if os.path.exists(self.workspace_index_filepath):
                with open(self.workspace_index_filepath, 'r', encoding='utf-8') as f_read:
                    for line in f_read:
                        try:
                            existing_entry = json.loads(line)
                            if existing_entry.get("filename") == filename:
                                # Update existing entry (especially for append or overwrite)
                                logger.debug(f"Updating existing index entry for {filename}")
                                existing_entry.update(metadata_entry) # Overwrite with new metadata
                                updated_entries.append(existing_entry)
                                entry_found_and_updated = True
                            else:
                                updated_entries.append(existing_entry)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping corrupt line in workspace index: {line.strip()}")

            if not entry_found_and_updated:
                updated_entries.append(metadata_entry) # Add as new if not found

            # Write all entries back (overwrite the file)
            with open(self.workspace_index_filepath, 'w', encoding='utf-8') as f_write:
                for entry in updated_entries:
                    json.dump(entry, f_write)
                    f_write.write('\n')
            logger.info(f"Workspace index successfully updated for {filename}.")

        except Exception as e:
            logger.error(f"Failed to write to workspace index file {self.workspace_index_filepath}: {e}", exc_info=True)
    
    # Inside GraphMemoryClient class
def _generate_file_content_for_persona(self, filename: str, content_description: str, requested_format: str | None) -> str | None:
    """
    Generates file content using an LLM, influenced by persona and current state.
    """
    logger.info(f"Generating file content for '{filename}' (Format: {requested_format}) based on description: '{content_description[:100]}...'")

    prompt_template = self._load_prompt("generate_structured_file_content_prompt.txt")
    if not prompt_template:
        logger.error("generate_structured_file_content_prompt.txt not found. Cannot generate file content.")
        return f"Error: Prompt missing for content generation of {filename}."

    # --- Gather Context for the Prompt ---
    asm_snippet_str = "[ASM: Not available]"
    if self.autobiographical_model:
        asm_parts = []
        if self.autobiographical_model.get("summary_statement"): asm_parts.append(f"Summary: {self.autobiographical_model['summary_statement']}")
        if self.autobiographical_model.get("core_traits"): asm_parts.append(f"Traits: {', '.join(self.autobiographical_model['core_traits'])}")
        # Add more relevant ASM fields if needed
        if asm_parts: asm_snippet_str = "\n".join(asm_parts)
    
    emotional_context_str = "[Emotional State: Neutral]"
    if self.emotional_core and self.emotional_core.is_enabled:
        tendency = self.emotional_core.derived_tendency
        mood_hints = self.emotional_core.derived_mood_hints
        emotional_context_str = f"Tendency: {tendency} (Mood V: {mood_hints.get('valence', 0):.1f}, A: {mood_hints.get('arousal', 0):.1f})"
        # Optionally add strong needs/fears if relevant for content generation style

    # Convert persona_workspace_prefs dict to a string
    prefs_str_parts = []
    if self.workspace_persona_prefs: # Check if it's loaded
        for key, value in self.workspace_persona_prefs.items():
            prefs_str_parts.append(f"{key.replace('_', ' ').title()}: {value}")
    persona_workspace_preferences_string = ", ".join(prefs_str_parts) if prefs_str_parts else "Default"


    actual_requested_format = requested_format
    if not actual_requested_format: # Fallback to persona default if not specified in plan
        if filename.endswith(".txt"): actual_requested_format = "plain_text_narrative"
        elif filename.endswith(".md"): actual_requested_format = "markdown_bullet_list"
        elif filename.endswith(".json"): actual_requested_format = "json_object_or_list" # Be generic for JSON
        elif filename.endswith(".csv"): actual_requested_format = "csv_data"
        else: # Fallback to persona's default note format if extension doesn't give a clear hint
            actual_requested_format = self.workspace_persona_prefs.get('default_note_format', 'plain_text_narrative')
        logger.info(f"No specific format requested for '{filename}', defaulted to '{actual_requested_format}' based on extension/persona preference.")


    try:
        full_prompt = prompt_template.format(
            asm_context_snippet=asm_snippet_str,
            emotional_context_string=emotional_context_str,
            persona_workspace_preferences_string=persona_workspace_preferences_string,
            filename=filename,
            content_description=content_description,
            requested_format=actual_requested_format
        )
    except KeyError as e:
        logger.error(f"Missing placeholder in generate_structured_file_content_prompt.txt: {e}")
        return f"Error: Prompt formatting error for {filename} ({e})."

    logger.debug(f"Sending file content generation prompt for '{filename}':\n{full_prompt[:500]}...")
    
    # Use the new LLM task 'file_content_generation' from config.yaml
    # (or 'generate_structured_file_content' if you named it that - ensure consistency)
    generated_content = self._call_configured_llm('file_content_generation', prompt=full_prompt)

    if generated_content and not generated_content.startswith("Error:"):
        logger.info(f"Successfully generated content for '{filename}' (Length: {len(generated_content)}).")
        return generated_content.strip() # Strip leading/trailing whitespace from LLM output
    else:
        logger.error(f"LLM call failed or returned error for file content generation of '{filename}': {generated_content}")
        return f"Error: LLM failed to generate content for {filename}. Details: {generated_content}"
    
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
