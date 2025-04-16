# persistent_backend_graph.py
import os
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

# *** Import file manager ***
import file_manager # Assuming file_manager.py exists in the same directory

# --- Configuration ---
DEFAULT_CONFIG_PATH = "config.yaml"

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Define logger for this module

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

        # API URLs
        self.kobold_api_url = self.config.get('kobold_api_url', "http://localhost:5001/api/v1/generate")
        base_kobold_url = self.kobold_api_url.rsplit('/api/', 1)[0]
        self.kobold_chat_api_url = self.config.get('kobold_chat_api_url', f"{base_kobold_url}/v1/chat/completions")
        logger.info(f"Using Kobold Generate API URL: {self.kobold_api_url}")
        logger.info(f"Using Kobold Chat Completions API URL: {self.kobold_chat_api_url}")

        # Initialize attributes
        self.graph = nx.DiGraph(); self.index = None; self.embeddings = {}; self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}; self.last_added_node_uuid = None; self.tokenizer = None
        self.embedder = None # Initialize embedder attribute explicitly
        self.embedding_dim = 0 # Initialize embedding_dim

        os.makedirs(self.data_dir, exist_ok=True)
        embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2'); tokenizer_name = self.config.get('tokenizer_name', 'google/gemma-7b-it')

        # Load Embedder
        try:
             logger.info(f"Loading embed model: {embedding_model_name}")
             # --- Assign to self.embedder ---
             self.embedder = SentenceTransformer(embedding_model_name, trust_remote_code=True)
             self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
             logger.info(f"Embed model loaded. Dim: {self.embedding_dim}")
             # --- DEBUG Check right after assignment ---
             logger.debug(f"INIT Check 1: Has embedder? {hasattr(self, 'embedder')}, Type: {type(self.embedder)}, Dim: {self.embedding_dim}")
        except Exception as e:
             logger.error(f"Failed loading embed model: {e}", exc_info=True)
             # Ensure embedder is None if loading fails
             self.embedder = None
             self.embedding_dim = 0
             raise # Re-raise exception to prevent client initialization? Or handle more gracefully?

        # Load Tokenizer
        try:
             logger.info(f"Loading tokenizer: {tokenizer_name}")
             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
             logger.info("Tokenizer loaded.")
        except Exception as e:
             logger.error(f"Failed loading tokenizer '{tokenizer_name}': {e}", exc_info=True)
             self.tokenizer = None # Ensure tokenizer is None if loading fails
             # Decide if this is fatal - maybe raise? For now, allow continuation.
             # raise

        self._load_memory() # Loads data from self.data_dir

        # Set last added node UUID (logic remains the same)
        if self.graph.number_of_nodes() > 0:
            # ... (existing logic to find last node) ...
             pass

        # --- FINAL DEBUG check at end of __init__ ---
        logger.debug(f"INIT END: Has embedder? {hasattr(self, 'embedder')}")
        if hasattr(self, 'embedder') and self.embedder:
             logger.debug(f"INIT END: Embedder type: {type(self.embedder)}, Dim: {getattr(self, 'embedding_dim', 'Not Set')}")
        else:
             logger.error("INIT END: EMBEDDER ATTRIBUTE IS MISSING OR NONE!")

        logger.info(f"GraphMemoryClient initialized for personality '{self.personality}'.")


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
        if not loaded_something: logger.info("No existing memory data found.")
        else: logger.info("Memory loading complete.")

    def _rebuild_index_from_graph_embeddings(self):
        """Rebuilds FAISS index based on current graph nodes/embeddings."""
        # (Keep implementation from previous version)
        logger.info(f"Rebuilding FAISS index from {self.graph.number_of_nodes()} graph nodes...")
        if self.graph.number_of_nodes() == 0: logger.warning("Graph empty, init empty index."); self.index = faiss.IndexFlatL2(self.embedding_dim); self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}; return
        try:
            new_index = faiss.IndexFlatL2(self.embedding_dim); new_map = {}; new_inv_map = {}; emb_list = []; current_id = 0
            nodes_in_graph = list(self.graph.nodes())
            for node_uuid in nodes_in_graph:
                embedding = self.embeddings.get(node_uuid)
                if embedding is not None and embedding.shape == (self.embedding_dim,):
                    emb_list.append(embedding.astype('float32')); new_map[current_id] = node_uuid; new_inv_map[node_uuid] = current_id; current_id += 1
                else: logger.warning(f"Skipping node {node_uuid[:8]} in rebuild (bad embed).")
            if emb_list: new_index.add(np.vstack(emb_list)); logger.info(f"Added {new_index.ntotal} vectors to new index.")
            else: logger.warning("No valid embeddings for rebuild.")
            self.index = new_index; self.faiss_id_to_uuid = new_map; self.uuid_to_faiss_id = new_inv_map
            logger.info("FAISS index rebuild based on graph complete.")
        except Exception as e: logger.error(f"FAISS rebuild error: {e}", exc_info=True); self.index = faiss.IndexFlatL2(self.embedding_dim); self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}

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
            logger.info(f"Memory saving done ({time.time() - start_time:.2f}s).")
        except Exception as e: logger.error(f"Unexpected save error: {e}", exc_info=True)

    # --- Memory Node Management ---
    # (Keep add_memory_node, _rollback_add, delete_memory_entry, _find_latest_node_uuid, edit_memory_entry, forget_topic)
    # ... (methods unchanged) ...
    def add_memory_node(self, text: str, speaker: str, node_type: str = 'turn', timestamp: str = None, base_strength: float = 0.5) -> str | None:
        """Adds a new memory node to the graph and index."""
        # --- DEBUG Check at start of function ---
        logger.debug(f"ADD_MEMORY_NODE START: Has embedder? {hasattr(self, 'embedder')}")

        if not text: logger.warning("Skip adding empty node."); return None
        log_text = text[:80] + '...' if len(text) > 80 else text
        logger.info(f"Adding node: Spk={speaker}, Typ={node_type}, Txt='{log_text}'")
        current_time = time.time()
        node_uuid = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        # --- Get embedding ---
        embedding = self._get_embedding(text)
        # --- Check if embedding failed ---
        if embedding is None:
             logger.error(f"Failed to get embedding for node {node_uuid}. Node not added.")
             return None # Stop if embedding failed

        # --- Add to graph ---
        try:
            self.graph.add_node(
                node_uuid,
                uuid=node_uuid, text=text, speaker=speaker, timestamp=timestamp,
                node_type=node_type, base_strength=float(base_strength),
                activation_level=0.0, last_accessed_ts=current_time
            )
            logger.debug(f"Node {node_uuid[:8]} added to graph.")
        except Exception as e:
             logger.error(f"Failed adding node {node_uuid} to graph: {e}")
             return None # Stop if graph add fails

        # --- Add embedding to dictionary ---
        self.embeddings[node_uuid] = embedding

        # --- Add to FAISS ---
        try:
            if self.index is None:
                # Check embedding_dim existence before creating index
                if hasattr(self, 'embedding_dim') and self.embedding_dim > 0:
                     logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
                     self.index = faiss.IndexFlatL2(self.embedding_dim)
                else:
                     logger.error("Cannot initialize FAISS index: embedding_dim not set.")
                     self._rollback_add(node_uuid) # Rollback graph/embedding dict changes
                     return None

            self.index.add(np.array([embedding], dtype='float32'))
            new_faiss_id = self.index.ntotal - 1
            self.faiss_id_to_uuid[new_faiss_id] = node_uuid
            self.uuid_to_faiss_id[node_uuid] = new_faiss_id
            logger.debug(f"Embedding {node_uuid[:8]} added to FAISS ID {new_faiss_id}.")
        except Exception as e:
             logger.error(f"Failed adding embed {node_uuid} to FAISS: {e}")
             self._rollback_add(node_uuid) # Rollback changes
             return None

        # --- Link temporally ---
        if self.last_added_node_uuid and self.last_added_node_uuid in self.graph:
            try:
                self.graph.add_edge(
                    self.last_added_node_uuid, node_uuid,
                    type='TEMPORAL', base_strength=0.8, last_traversed_ts=current_time
                )
                logger.debug(f"Added T-edge {self.last_added_node_uuid[:8]}->{node_uuid[:8]}.")
            except Exception as e:
                 logger.error(f"Failed adding T-edge: {e}") # Log error but don't fail the whole add

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
    def _search_similar_nodes(self, query_text: str, k: int = None, node_type_filter: str = None) -> list[tuple[str, float]]:
        """Searches FAISS for nodes similar to query_text, optionally filtering by type."""
        if k is None: k = self.config.get('activation', {}).get('max_initial_nodes', 7)
        if not query_text or self.index is None or self.index.ntotal == 0: return []
        try:
            q_embed = self._get_embedding(query_text)
            if q_embed is None or q_embed.shape != (self.embedding_dim,): return []
            q_embed_np = np.array([q_embed], dtype='float32'); search_k = k * 2 if node_type_filter else k; actual_k = min(search_k, self.index.ntotal);
            if actual_k == 0: return []
            dists, idxs = self.index.search(q_embed_np, actual_k); results = []
            logger.debug(f"FAISS Search Results (Top {actual_k}, filter='{node_type_filter}'):")
            if len(idxs) > 0:
                for i, faiss_id in enumerate(idxs[0]):
                    fid_int = int(faiss_id); dist = float(dists[0][i]); logger.debug(f"  Rank {i+1}: ID={fid_int}, Dist={dist:.4f}")
                    if fid_int != -1:
                        uuid = self.faiss_id_to_uuid.get(fid_int)
                        if uuid and uuid in self.graph:
                             if node_type_filter:
                                 if self.graph.nodes[uuid].get('node_type') == node_type_filter: results.append((uuid, dist)); logger.debug(f"    -> Valid UUID: {uuid[:8]} (Type MATCH)")
                                 else: logger.debug(f"    -> Valid UUID: {uuid[:8]} (Type MISMATCH)")
                             else: results.append((uuid, dist)); logger.debug(f"    -> Valid UUID: {uuid[:8]} (No filter)")
                        else: logger.debug(f"    -> UUID {uuid} not in graph/map.")
                    if len(results) >= k: break
            logger.info(f"Found {len(results)} similar nodes (type='{node_type_filter or 'any'}') for query '{query_text[:30]}...'")
            results.sort(key=lambda item: item[1]); return results
        except Exception as e: logger.error(f"FAISS search error: {e}", exc_info=True); return []
    def retrieve_memory_chain(self, initial_node_uuids: list[str]) -> list[dict]:
        """Retrieves relevant memories using activation spreading, considering edge types."""
        # (Keep implementation from previous version)
        act_cfg = self.config.get('activation', {}); initial_activation = act_cfg.get('initial', 1.0); spreading_depth = act_cfg.get('spreading_depth', 3); activation_threshold = act_cfg.get('threshold', 0.1); prop_base = act_cfg.get('propagation_factor_base', 0.65); prop_factors = act_cfg.get('propagation_factors', {}); prop_temporal_fwd = prop_factors.get('TEMPORAL_fwd', 1.0); prop_temporal_bwd = prop_factors.get('TEMPORAL_bwd', 0.8); prop_summary_fwd = prop_factors.get('SUMMARY_OF_fwd', 1.1); prop_summary_bwd = prop_factors.get('SUMMARY_OF_bwd', 0.4); prop_concept_fwd = prop_factors.get('MENTIONS_CONCEPT_fwd', 1.0); prop_concept_bwd = prop_factors.get('MENTIONS_CONCEPT_bwd', 0.9); prop_assoc = prop_factors.get('ASSOCIATIVE', 0.8); prop_hier_fwd = prop_factors.get('HIERARCHICAL_fwd', 1.1); prop_hier_bwd = prop_factors.get('HIERARCHICAL_bwd', 0.5); prop_unknown = prop_factors.get('UNKNOWN', 0.5)
        logger.info(f"Starting retrieval. Initial nodes: {initial_node_uuids}")
        if self.graph.number_of_nodes() == 0: logger.warning("Graph empty."); return []
        activation_levels = defaultdict(float); current_time = time.time(); valid_initial = 0
        for uuid in initial_node_uuids:
            if uuid in self.graph: activation_levels[uuid] = initial_activation; self.graph.nodes[uuid]['last_accessed_ts'] = current_time; valid_initial += 1;
            else: logger.warning(f"Initial node {uuid} not in graph.")
        if not activation_levels: logger.warning("No valid initial nodes."); return []
        logger.debug(f"Valid initial nodes: {valid_initial}"); active_nodes = set(activation_levels.keys())
        for depth in range(spreading_depth):
            logger.debug(f"--- Spreading Step {depth + 1} ---"); newly_activated = defaultdict(float)
            nodes_to_proc = list(active_nodes); logger.debug(f" Processing {len(nodes_to_proc)} nodes.")
            for source_uuid in nodes_to_proc:
                source_act = activation_levels.get(source_uuid, 0);
                if source_act < 1e-6: continue
                neighbors = set(self.graph.successors(source_uuid)) | set(self.graph.predecessors(source_uuid))
                for neighbor_uuid in neighbors:
                    if neighbor_uuid == source_uuid or neighbor_uuid not in self.graph: continue
                    is_forward = self.graph.has_edge(source_uuid, neighbor_uuid)
                    edge_data = self.graph.get_edge_data(source_uuid, neighbor_uuid) if is_forward else self.graph.get_edge_data(neighbor_uuid, source_uuid)
                    if not edge_data: continue
                    edge_type = edge_data.get('type', 'TEMPORAL'); type_factor = prop_unknown
                    if edge_type == 'TEMPORAL': type_factor = prop_temporal_fwd if is_forward else prop_temporal_bwd
                    elif edge_type == 'SUMMARY_OF': type_factor = prop_summary_fwd if is_forward else prop_summary_bwd
                    elif edge_type == 'MENTIONS_CONCEPT': type_factor = prop_concept_fwd if is_forward else prop_concept_bwd
                    elif edge_type == 'ASSOCIATIVE': type_factor = prop_assoc
                    elif edge_type == 'HIERARCHICAL': type_factor = prop_hier_fwd if is_forward else prop_hier_bwd
                    dyn_str = self._calculate_dynamic_edge_strength(edge_data, current_time); act_pass = source_act * dyn_str * prop_base * type_factor
                    if act_pass > 1e-6:
                        newly_activated[neighbor_uuid] += act_pass
                        edge_key = (source_uuid, neighbor_uuid) if is_forward else (neighbor_uuid, source_uuid)
                        if edge_key in self.graph.edges: self.graph.edges[edge_key]['last_traversed_ts'] = current_time
            nodes_to_decay = list(activation_levels.keys())
            for uuid in nodes_to_decay:
                if uuid in self.graph: node_data = self.graph.nodes[uuid]; decay_mult = self._calculate_node_decay(node_data, current_time); activation_levels[uuid] *= decay_mult; self.graph.nodes[uuid]['last_accessed_ts'] = current_time
                else:
                    if uuid in activation_levels: del activation_levels[uuid]
            active_nodes.clear()
            all_involved = set(activation_levels.keys()) | set(newly_activated.keys())
            for uuid in all_involved:
                 if uuid not in self.graph: continue
                 activation_levels[uuid] += newly_activated.get(uuid, 0.0)
                 if newly_activated.get(uuid, 0.0) > 1e-6 or uuid in nodes_to_decay: self.graph.nodes[uuid]['last_accessed_ts'] = current_time
                 if activation_levels[uuid] > 1e-6: active_nodes.add(uuid)
                 elif uuid in activation_levels: del activation_levels[uuid]
            logger.debug(f" Step {depth+1} finished. Active: {len(active_nodes)}. Max Act: {max(activation_levels.values()) if activation_levels else 0:.3f}")
        relevant_nodes = []
        for uuid, act in activation_levels.items():
            if act >= activation_threshold and uuid in self.graph: node_data = self.graph.nodes[uuid].copy(); node_data['final_activation'] = act; relevant_nodes.append(node_data)
        logger.info(f"Found {len(relevant_nodes)} nodes above threshold ({activation_threshold}).")
        relevant_nodes.sort(key=lambda x: (x['final_activation'], x.get('timestamp', '')), reverse=True)
        if relevant_nodes: logger.info(f"Final nodes: [{', '.join([n['uuid'][:8]+f'({n['final_activation']:.3f})' for n in relevant_nodes])}]")
        else: logger.info("No relevant nodes found.")
        logger.debug("--- Retrieved Node Texts (Top 5) ---")
        for i, node in enumerate(relevant_nodes[:5]): logger.debug(f"  {i+1}. ({node['final_activation']:.3f}) UUID:{node['uuid'][:8]} Text: '{node.get('text', 'N/A')[:80]}...'")
        logger.debug("------------------------------------")
        return relevant_nodes

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

        # --- Token Budget Calculation ---
        prompt_cfg = self.config.get('prompting', {})
        context_headroom = prompt_cfg.get('context_headroom', 250)
        mem_budget_ratio = prompt_cfg.get('memory_budget_ratio', 0.45)
        hist_budget_ratio = prompt_cfg.get('history_budget_ratio', 0.55)
        try:
            fixed_tokens = (len(tokenizer.encode(time_info_block)) +
                            len(tokenizer.encode(user_input_fmt)) +
                            len(tokenizer.encode(final_model_tag)))
        except Exception as e:
            logger.error(f"Tokenization error for fixed prompt parts: {e}") # Corrected logger name
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
                td = "TS?" # Default timestamp description
                try:
                    dt=datetime.fromisoformat(ts.replace('Z','+00:00'))
                    diff=datetime.now(timezone.utc)-dt
                    if diff < timedelta(hours=1): td=f"{int(diff.total_seconds()/60)}m ago"
                    elif diff < timedelta(days=1): td=f"{int(diff.total_seconds()/3600)}h ago"
                    else: td=dt.strftime('%Y-%m-%d')
                except Exception as ts_e:
                    logger.debug(f"Could not parse memory timestamp '{ts}': {ts_e}") # Corrected logger name

                fmt_mem = f"{spk} ({td}): {txt}\n"
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

        return final_prompt


    def _call_kobold_api(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Sends prompt to KoboldCpp API, returns generated text."""
        # Use the correct logger name: 'logger'
        logger.debug(f"_call_kobold_api received prompt ('{prompt[:80]}...'). Length: {len(prompt)}") # Corrected logger name

        # Calculate max_context_length
        try: prompt_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else len(prompt) // 3
        except Exception as e: logger.warning(f"Tokenizer error: {e}. Estimating prompt tokens."); prompt_tokens = len(prompt) // 3 # Corrected logger name
        model_max_ctx = self.config.get('prompting',{}).get('max_context_tokens', 4096)
        desired_total_tokens = prompt_tokens + max_length + 50
        max_ctx_len = min(model_max_ctx, desired_total_tokens)
        logger.debug(f"Prompt tokens: ~{prompt_tokens}. Max new: {max_length}. Max context length for API call: {max_ctx_len}") # Corrected logger name

        api_url = self.kobold_api_url
        if not api_url: logger.error("Kobold API URL is not configured."); return "Error: Kobold API URL not configured." # Corrected logger name

        payload = {
            'prompt': prompt, # Should contain [Image:...] tag if sent
            'max_context_length': max_ctx_len,
            'max_length': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'stop_sequence': ["<end_of_turn>", "<start_of_turn>user\n", "User:", "\nUser:"],
            'use_memory': False, 'use_story': False, 'use_authors_note': False, 'use_world_info': False,
        }
        log_payload = payload.copy()
        log_payload['prompt'] = log_payload['prompt'][:100] + ("..." if len(log_payload['prompt']) > 100 else "")
        logger.debug(f"Payload sent to Kobold API ({api_url}): {log_payload}") # Corrected logger name

        try:
            response = requests.post(api_url, json=payload, timeout=180) # Increased timeout slightly
            response.raise_for_status()
            result = response.json()
            gen_txt = result.get('results', [{}])[0].get('text', '').strip()
            for seq in payload['stop_sequence']:
                if gen_txt.endswith(seq): gen_txt = gen_txt[:-len(seq)].rstrip()
            if not gen_txt: logger.warning("Kobold API returned empty text.") # Corrected logger name
            else: logger.debug(f"Kobold API raw response text: '{gen_txt[:100]}...'") # Corrected logger name
            return gen_txt
        except requests.exceptions.RequestException as e:
            logger.error(f"Kobold API connection/request error: {e}", exc_info=True) # Corrected logger name
            return f"Error: Could not connect to Kobold API at {api_url}."
        except Exception as e:
            logger.error(f"Kobold API call unexpected error: {e}", exc_info=True) # Corrected logger name
            return f"Error: Unexpected issue during Kobold API call."

    # --- Memory Modification & Action Analysis ---

    # *** NEW: Analyze for File/Calendar/Other Actions ***
    def analyze_action_request(self, request_text: str) -> dict:
        """
        Uses LLM to detect non-memory action intents (e.g., file, calendar)
        and extract arguments. Returns structured action data or {"action": "none"}.
        """
        logger.info(f"Analyzing for action request: '{request_text[:100]}...'")
        # Define tools and their required arguments
        tools = {
            "create_file": ["filename", "content"],
            "append_file": ["filename", "content"],
            "add_calendar_event": ["date", "time", "description"],
            "read_calendar": []  # Date is optional
        }
        # *** REFINED Prompt Instructions ***
        tool_descriptions = """AVAILABLE ACTIONS:
- create_file: Creates/overwrites a file with given content. Requires: 'filename', 'content'.
- append_file: Appends content to a file. Requires: 'filename', 'content'.
- add_calendar_event: Adds an event to the calendar. Requires: 'date' (YYYY-MM-DD or relative like 'tomorrow'), 'time' (HH:MM or description like 'afternoon'), 'description'.
- read_calendar: Reads events for a date. Optional: 'date' (YYYY-MM-DD or relative like 'today', defaults to today)."""

        system_prompt = f"""SYSTEM: You can perform specific actions related to file management and calendar scheduling. Analyze the user's request below.
Determine if the user is asking to perform one of the AVAILABLE ACTIONS. Pay close attention to the required arguments.
{tool_descriptions}

**IMPORTANT:** Only identify an action if the user's intent CLEARLY matches an AVAILABLE ACTION. Do NOT map requests like "delete item from list" or "remove todo" to file actions like 'append_file'. If the request is about modifying memory (delete, edit, forget memory entries/nodes/topics), output {{"action": "none"}}. If the request doesn't match an AVAILABLE ACTION or memory modification, output {{"action": "none"}}.

- If the request matches an action AND provides ALL required arguments, respond ONLY with JSON: {{"action": "action_name", "args": {{"arg1": "value1", ...}}}}.
- If the request matches an action but is MISSING required arguments, respond ONLY with JSON: {{"action": "clarify", "missing_args": ["arg1", ...], "original_action": "action_name"}}.
- For all other requests (including memory modifications or unsupported actions), respond ONLY with JSON: {{"action": "none"}}.

Examples:
User Request: save this as my_file.txt: The content.
JSON Response: {{"action": "create_file", "args": {{"filename": "my_file.txt", "content": "The content."}}}}

User Request: add Dr. Smith Appt 2025-05-10 9am to calendar
JSON Response: {{"action": "add_calendar_event", "args": {{"date": "2025-05-10", "time": "9am", "description": "Dr. Smith Appt"}}}}

User Request: append status report to project_log.txt
JSON Response: {{"action": "clarify", "missing_args": ["content"], "original_action": "append_file"}}

User Request: what's happening today?
JSON Response: {{"action": "read_calendar", "args": {{}}}} # Assumes default to today

User Request: delete the previous memory node
JSON Response: {{"action": "none"}}

User Request: remove the first item from my shopping list file
JSON Response: {{"action": "none"}}

Analyze:"""
        full_prompt = f"{system_prompt}\nUser Request: {request_text}\nJSON Response:"

        logger.debug(f"Sending action analysis prompt:\n{full_prompt}")
        llm_response_str = self._call_kobold_api(full_prompt, max_length=250, temperature=0.1)

        if not llm_response_str: return {'action': 'error', 'reason': 'LLM call failed for action analysis'}
        try:
            logger.debug(f"Raw action analysis response: {llm_response_str}")
            # Try to extract JSON even if there's trailing text
            match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = llm_response_str.strip()  # Fallback if no {} found

            parsed = json.loads(json_str)
            logger.info(f"LLM Parsed Action: {parsed}")

            action = parsed.get("action")
            if not action: raise ValueError("Missing 'action' key")
            if action not in ["none", "clarify", "error", "create_file", "append_file", "add_calendar_event",
                              "read_calendar"]:
                logger.warning(f"LLM returned unknown action '{action}'. Treating as 'none'.")
                return {"action": "none"}
            if action == "none": return {"action": "none"}
            if action == "clarify":
                if "missing_args" not in parsed or "original_action" not in parsed: raise ValueError(
                    "Clarify missing keys")
                return parsed
            if action == "error": return parsed

            # Validate required args
            args = parsed.get("args", {})
            required_args = tools.get(action, [])
            missing = [arg for arg in required_args if arg not in args or not args[arg]]
            if action == "read_calendar" and "date" in missing: missing.remove("date")

            if missing:
                logger.warning(f"Action '{action}' missing args: {missing}. Requesting clarification.")
                return {"action": "clarify", "missing_args": missing, "original_action": action}

            # Sanitize filename
            if "filename" in args:
                args["filename"] = os.path.basename(str(args.get("filename", "default.txt")))
                if not args["filename"] or args["filename"] in ['.', '..']: raise ValueError("Invalid filename")

            return {"action": action, "args": args}

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM Action Parse/Validation Error: {e}. Raw: '{llm_response_str}'"); return {
                'action': 'error', 'reason': f'LLM Parse/Validation Fail: {e}', 'raw_response': llm_response_str}
        except Exception as e:
            logger.error(f"Unexpected error parsing action response: {e}", exc_info=True); return {
                'action': 'error', 'reason': f'Unexpected action parsing error: {e}',
                'raw_response': llm_response_str}

        # (Keep all other methods unchanged - __init__, _load_config, helpers, memory management, retrieval, prompting, execute_action, consolidation, reset, file wrappers etc.)
        # ... REST OF THE GraphMemoryClient Class ...

    # (Keep Example Usage Block unchanged)
    if __name__ == "__main__":
        # ...
        logger.info("Basic test finished.")

    def analyze_memory_modification_request(self, request: str) -> dict:
        """Analyzes user request for **memory modification only** (delete, edit, forget)."""
        # (Keep implementation from previous version)
        logger.info(f"Analyzing *memory modification* request: '{request[:100]}...'"); uuid_pattern=r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'; found_uuids=re.findall(uuid_pattern, request, re.IGNORECASE); target_uuid=found_uuids[0] if found_uuids else None; request_lower=request.lower(); detected_action=None
        if any(kw in request_lower for kw in self.config.get('modification_keywords',[])):
            if any(kw in request_lower for kw in ['delete', 'remove', 'forget']): detected_action = 'delete'
            elif any(kw in request_lower for kw in ['edit', 'change', 'correct', 'update']): detected_action = 'edit'
        if detected_action and target_uuid:
             logger.info(f"Direct extract: Action={detected_action}, UUID={target_uuid}"); result={'action': detected_action, 'target_uuid': target_uuid}
             if detected_action == 'edit':
                  parts = request.split(target_uuid); new_text=parts[1].strip() if len(parts)>1 and parts[1].strip() else None
                  if new_text:
                      for prefix in ["to ", "say ", "is "]:
                          if new_text.lower().startswith(prefix): new_text = new_text[len(prefix):].strip()
                      result['new_text']=new_text; logger.info(f"Extracted new text: {new_text[:50]}...")
                  else: logger.warning("Edit UUID found, but no new text extracted."); result['new_text'] = None
             return result
        logger.info("Falling back to LLM analysis for memory mod request.")
        system_prompt="""SYSTEM: Analyze user request for memory modification ('delete', 'edit', 'forget'). Respond ONLY with JSON.
- 'delete'/'forget': use "target_uuid" if UUID present, else use "target" description. For 'forget', identify "topic".
- 'edit': Identify "target_uuid" or "target" AND "new_text".
- Not a command: {"action": "none"}.
Example Request: edit 123e4567-e89b-12d3-a456-426614174000 to say "new"
Example JSON: {"action": "edit", "target_uuid": "123e4567-e89b-12d3-a456-426614174000", "new_text": "new"}
Analyze:"""
        full_prompt=f"{system_prompt}\nUser Request: {request}\nJSON Response:"
        llm_response_str = self._call_kobold_api(full_prompt, max_length=150, temperature=0.2)
        if not llm_response_str: return {'action': 'error', 'reason': 'LLM call failed'}
        try:
            llm_response_str = llm_response_str.strip().replace("```json", "").replace("```", "").strip(); parsed_response = json.loads(llm_response_str); logger.info(f"LLM Parsed Mod: {parsed_response}")
            if 'action' not in parsed_response or parsed_response['action'] not in ['delete','edit','forget','none','error']: raise ValueError("Missing/Invalid action")
            if target_uuid and 'target_uuid' not in parsed_response and parsed_response['action'] in ['delete', 'edit']: logger.info(f"Adding regex UUID {target_uuid} to LLM result."); parsed_response['target_uuid'] = target_uuid; parsed_response.pop('target', None)
            return parsed_response
        except Exception as e: logger.error(f"LLM Mod Parse Error: {e}. Raw: '{llm_response_str}'"); return {'action': 'error', 'reason': f'LLM Parse Fail: {e}', 'raw_response': llm_response_str}

    # *** NEW: Action Dispatcher ***
    def execute_action(self, action_data: dict) -> tuple[bool, str, str]:
        """Executes a validated action based on the action_data dictionary."""
        action = action_data.get("action")
        args = action_data.get("args", {})
        # *** ADDED: Log arguments before execution ***
        logger.debug(f"Attempting to execute action '{action}' with args: {args}")
        success = False; message = f"Action '{action}' failed."; action_suffix = f"{action}_fail"
        try:
            if action == "create_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content is not None:
                    # Log exact values passed
                    logger.debug(f"Calling file_manager.create_or_overwrite_file(config, personality='{self.personality}', filename='{filename}', content='{str(content)[:50]}...')")
                    success = file_manager.create_or_overwrite_file(self.config, self.personality, filename, str(content)) # Ensure content is string
                    message = f"File '{filename}' created/overwritten." if success else f"Failed create/overwrite '{filename}'."
                else: message = "Missing filename or content for create_file."
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
                logger.debug(f"Calling file_manager.read_calendar_events(config, personality='{self.personality}', target_date='{date}')")
                # TODO: Date parsing
                events = file_manager.read_calendar_events(self.config, self.personality, date)
                success = True; action_suffix = "cal_read_success"
                date_str = f" for {date}" if date else ""
                if events: message = f"Found {len(events)} event(s){date_str}:\n" + "\n".join([f"- {e.get('time', '?')}: {e.get('description', '?')} ({e.get('event_date', '?')})" for e in events])
                else: message = f"No events found{date_str}."
            else: message = f"Unknown action '{action}'."; action_suffix = "unknown_action_fail"

            if success and not action_suffix.endswith("_success"): # Ensure success suffix is set if action succeeded
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
        # --- DEBUG Check at start of function ---
        logger.debug(f"PROCESS_INTERACTION START: Has embedder? {hasattr(self, 'embedder')}")
        if not hasattr(self, 'embedder') or self.embedder is None:
             logger.error("PROCESS_INTERACTION ERROR: Cannot proceed without embedder!")
             return "Error: Backend embedder not initialized correctly.", []

        # (Rest of the process_interaction logic remains the same as previous version)
        logger.info(f"Processing interaction: Input='{user_input[:50]}...' Has Attachment: {bool(attachment_data)}")
        if attachment_data: logger.debug(f"Attachment details: type={attachment_data.get('type')}, filename={attachment_data.get('filename')}")
        ai_response = "Error: Processing failed."; memory_chain_data = []
        try:
            if attachment_data and attachment_data.get('type') == 'image' and attachment_data.get('data_url'):
                logger.info("Image attachment detected. Using Chat Completions API."); logger.info("Skipping memory retrieval for image prompt."); memory_chain_data = []
                messages = []; history_limit = 5; relevant_history = conversation_history[-history_limit:]
                for turn in relevant_history:
                    role = "user" if turn.get("speaker") == "User" else "assistant"; text_content = re.sub(r'\s*\[Image:\s*.*?\s*\]\s*', '', turn.get("text","")).strip()
                    if text_content: messages.append({"role": role, "content": text_content})
                user_content = [];
                if user_input: user_content.append({"type": "text", "text": user_input})
                user_content.append({"type": "image_url", "image_url": {"url": attachment_data['data_url']}}); messages.append({"role": "user", "content": user_content})
                max_gen_tokens = self.config.get('prompting', {}).get('max_generation_tokens', 512); ai_response = self._call_kobold_multimodal_api(messages=messages, max_tokens=max_gen_tokens)
                graph_user_input = user_input;
                if attachment_data.get('filename'): placeholder = f" [Image Attached: {attachment_data['filename']}]"; separator = " " if graph_user_input else ""; graph_user_input += separator + placeholder
            else:
                logger.info("No valid image attachment. Using standard Generate API."); memory_chain_data = []
                if not user_input.strip().startswith("[Image:"):
                    logger.info("Searching initial nodes..."); max_initial_nodes = self.config.get('activation', {}).get('max_initial_nodes', 7); initial_nodes = self._search_similar_nodes(user_input, k=max_initial_nodes); initial_uuids = [uid for uid, score in initial_nodes]; logger.info(f"Initial UUIDs: {initial_uuids}")
                    if initial_uuids: logger.info("Retrieving memory chain..."); memory_chain_data = self.retrieve_memory_chain(initial_uuids); logger.info(f"Retrieved memory chain size: {len(memory_chain_data)}")
                    else: logger.info("No relevant initial nodes found.")
                else: logger.warning("Input started with [Image:] but wasn't handled as attachment? Proceeding without memory.")
                logger.info("Constructing prompt string..."); max_tokens = self.config.get('prompting', {}).get('max_context_tokens', 4096); prompt = self._construct_prompt(user_input, conversation_history, memory_chain_data, self.tokenizer, max_tokens)
                logger.info("Calling standard LLM Generate API..."); max_gen_tokens = self.config.get('prompting', {}).get('max_generation_tokens', 512); ai_response = self._call_kobold_api(prompt=prompt, max_length=max_gen_tokens)
                graph_user_input = user_input
            if not ai_response: logger.error("LLM returned empty response."); ai_response = "Error: Received empty response from language model."
            parsed_response = ai_response.strip()
            logger.info("Adding user input node to graph..."); logger.debug(f"Adding user node with text: '{graph_user_input[:100]}...'")
            user_node_uuid = self.add_memory_node(graph_user_input, "User") # ERROR originates here if embedder missing
            logger.info("Adding AI response node to graph..."); logger.debug(f"Adding AI node with text: '{parsed_response[:100]}...'")
            ai_node_uuid = self.add_memory_node(parsed_response, "AI") # Or here
        except Exception as e: logger.error(f"Error during process_interaction: {e}", exc_info=True); ai_response = f"Error during processing: {e}"
        return parsed_response, memory_chain_data

    # --- Consolidation ---
    # (Keep _select_nodes_for_consolidation and run_consolidation from previous version)
    def _select_nodes_for_consolidation(self, count: int = None) -> list[str]:
        """Selects recent 'turn' nodes for consolidation."""
        # (Keep implementation from previous version)
        if count is None: count = self.config.get('consolidation', {}).get('turn_count', 10)
        turn_nodes = [(u, d['timestamp']) for u, d in self.graph.nodes(data=True) if d.get('node_type') == 'turn' and d.get('timestamp')]
        turn_nodes.sort(key=lambda x: x[1], reverse=True)
        return [uuid for uuid, ts in turn_nodes[:count]]

    def run_consolidation(self, min_nodes: int = None):
        """Performs consolidation: summary, concepts, relations, hierarchy, pruning."""
        # (Keep implementation from previous version)
        if min_nodes is None: min_nodes = self.config.get('consolidation', {}).get('min_nodes', 5)
        consolidation_turn_count = self.config.get('consolidation', {}).get('turn_count', 10)
        concept_sim_threshold = self.config.get('consolidation', {}).get('concept_similarity_threshold', 0.3)
        logger.info(f"--- Starting Consolidation (Process ~{consolidation_turn_count} Turns) ---")
        node_uuids_to_process = self._select_nodes_for_consolidation(consolidation_turn_count)
        if len(node_uuids_to_process) < min_nodes: logger.info(f"Not enough turns ({len(node_uuids_to_process)} < {min_nodes}). Skip."); return
        logger.info(f"Selected {len(node_uuids_to_process)} nodes for consolidation: {node_uuids_to_process}")
        context_text = ""; nodes_data = []
        for uuid in reversed(node_uuids_to_process):
            if uuid in self.graph: nodes_data.append(self.graph.nodes[uuid])
        for node_data in nodes_data: context_text += f"{node_data.get('speaker', '?')}: {node_data.get('text', '')}\n"
        context_text = context_text.strip()
        if not context_text: logger.warning("Empty context. Skip consolidation."); return
        # --- 1. Summarization ---
        summary_prompt = f"<start_of_turn>user\nSummarize key points concisely in third person:\n--- START ---\n{context_text}\n--- END ---\nConcise Summary:<end_of_turn>\n<start_of_turn>model\n"
        logger.info("Requesting summary..."); summary_text = self._call_kobold_api(summary_prompt, 150, 0.5)
        summary_node_uuid = None; summary_created = False
        if summary_text and len(summary_text) > 10:
            logger.info(f"Generated Summary: '{summary_text[:100]}...'")
            summary_ts = nodes_data[-1].get('timestamp') if nodes_data else datetime.now(timezone.utc).isoformat()
            summary_node_uuid = self.add_memory_node(summary_text, "System", 'summary', summary_ts, 0.7)
            if summary_node_uuid:
                 summary_created = True; logger.info(f"Added summary node {summary_node_uuid[:8]}. Adding edges...")
                 current_time = time.time()
                 for orig_uuid in node_uuids_to_process:
                     if orig_uuid in self.graph: self.graph.add_edge(summary_node_uuid, orig_uuid, type='SUMMARY_OF', base_strength=0.9, last_traversed_ts=current_time)
            else: logger.error("Failed to add summary node.")
        else: logger.warning(f"No valid summary ('{summary_text}').")
        # --- 2. Concept Extraction & Deduplication ---
        concept_prompt = f"<start_of_turn>user\nList key concepts, topics, or named entities (max 5-7). Output ONLY a comma-separated list:\n--- START ---\n{context_text}\n--- END ---\nComma-separated list:<end_of_turn>\n<start_of_turn>model\n"
        logger.info("Requesting concepts..."); concepts_text = self._call_kobold_api(concept_prompt, 100, 0.3)
        extracted_concepts = [c.strip() for c in concepts_text.split(',') if c.strip() and len(c.strip()) > 1] if concepts_text else []
        concept_node_map = {}
        if extracted_concepts:
            logger.info(f"Extracted Concepts: {extracted_concepts}")
            logger.info(f"Adding/linking concept nodes...")
            current_time = time.time()
            for concept in extracted_concepts:
                if len(concept) > 80: logger.warning(f"Skip long concept: '{concept[:50]}...'"); continue
                existing_concept_uuid = None
                similar_concepts = self._search_similar_nodes(concept, k=1, node_type_filter='concept')
                if similar_concepts and similar_concepts[0][1] <= concept_sim_threshold:
                    existing_concept_uuid = similar_concepts[0][0]
                    if existing_concept_uuid in self.graph: logger.info(f"Found existing similar concept '{self.graph.nodes[existing_concept_uuid].get('text','')}' ({existing_concept_uuid[:8]}) for '{concept}'. Linking.")
                    else: logger.warning(f"Similar concept node {existing_concept_uuid} found but missing from graph. Creating new."); existing_concept_uuid = None
                else: logger.debug(f"No sufficiently similar existing concept found for '{concept}'. Creating new."); existing_concept_uuid = None
                if existing_concept_uuid is None:
                    new_concept_uuid = self.add_memory_node(concept, "System", 'concept', base_strength=0.8)
                    if new_concept_uuid: concept_node_map[concept] = new_concept_uuid; logger.debug(f"Added new concept node {new_concept_uuid[:8]} for '{concept}'.")
                    else: logger.warning(f"Failed add concept node for '{concept}'."); continue
                else: concept_node_map[concept] = existing_concept_uuid
                current_concept_uuid = concept_node_map.get(concept)
                if current_concept_uuid: # Link summary and original turns to concept (new or existing)
                    if summary_node_uuid and summary_node_uuid in self.graph and not self.graph.has_edge(summary_node_uuid, current_concept_uuid):
                        try: self.graph.add_edge(summary_node_uuid, current_concept_uuid, type='MENTIONS_CONCEPT', base_strength=0.7, last_traversed_ts=current_time); logger.debug(f"Edge Summary->Concept {current_concept_uuid[:8]}")
                        except Exception as e: logger.error(f"Error adding summary->concept edge: {e}")
                    for orig_uuid in node_uuids_to_process:
                        if orig_uuid in self.graph and not self.graph.has_edge(orig_uuid, current_concept_uuid):
                            try: self.graph.add_edge(orig_uuid, current_concept_uuid, type='MENTIONS_CONCEPT', base_strength=0.5, last_traversed_ts=current_time); logger.debug(f"Edge Turn {orig_uuid[:8]}->Concept {current_concept_uuid[:8]}")
                            except Exception as e: logger.error(f"Error adding turn->concept edge: {e}")
        else: logger.warning("LLM returned no concepts.")
        # --- 3. Relationship Extraction ---
        # (Keep implementation from previous version)
        if len(concept_node_map) >= 2:
            concept_list_str = "\n".join([f"- {c}" for c in concept_node_map.keys()])
            relation_prompt = f"""<start_of_turn>user
Given concepts:
{concept_list_str}
List related pairs using 'CONCEPT 1 -> CONCEPT 2' format, one per line. If none, output "NONE".
Related pairs:<end_of_turn>
<start_of_turn>model
"""
            logger.info("Requesting concept relationships..."); relations_text = self._call_kobold_api(relation_prompt, 100, 0.4)
            if relations_text and relations_text.strip().upper() != "NONE":
                logger.info(f"Found potential relationships:\n{relations_text}")
                current_time = time.time(); lines = relations_text.strip().split('\n')
                for line in lines:
                    if '->' in line:
                        parts = line.split('->');
                        if len(parts) == 2:
                            c1_txt, c2_txt = parts[0].strip().lstrip('- '), parts[1].strip()
                            uuid1, uuid2 = concept_node_map.get(c1_txt), concept_node_map.get(c2_txt)
                            if uuid1 and uuid2 and uuid1 in self.graph and uuid2 in self.graph:
                                try:
                                     if not self.graph.has_edge(uuid1, uuid2): self.graph.add_edge(uuid1, uuid2, type='ASSOCIATIVE', base_strength=0.6, last_traversed_ts=current_time); logger.info(f"Added ASSOC edge: {uuid1[:8]} -> {uuid2[:8]}")
                                     else: logger.debug(f"Assoc edge {uuid1[:8]}->{uuid2[:8]} exists.")
                                except Exception as e: logger.error(f"Error adding assoc edge: {e}")
                            else: logger.warning(f"Could not find nodes for relation: '{c1_txt}' -> '{c2_txt}'")
            else: logger.info("LLM reported no direct relationships.")
        # --- 4. Hierarchy Extraction ---
        # (Keep implementation from previous version)
        if concept_node_map:
            concept_list_str = "\n".join([f"- {c}" for c in concept_node_map.keys()])
            hierarchy_prompt = f"""<start_of_turn>user
Consider the following concepts:
{concept_list_str}
Are there any clear hierarchical relationships (e.g., 'Concept A' is a type of 'Concept B')?
If yes, list them one per line using the format: 'CHILD_CONCEPT is_a PARENT_CONCEPT'.
If no clear hierarchy exists among these specific concepts, output "NONE".
Hierarchical relationships:<end_of_turn>
<start_of_turn>model
"""
            logger.info("Requesting concept hierarchy..."); hierarchy_text = self._call_kobold_api(hierarchy_prompt, 100, 0.4)
            if hierarchy_text and hierarchy_text.strip().upper() != "NONE":
                logger.info(f"Found potential hierarchies:\n{hierarchy_text}")
                current_time = time.time(); lines = hierarchy_text.strip().split('\n')
                for line in lines:
                    match = re.search(r"(.+)\s+(is_a|is a type of|is part of)\s+(.+)", line, re.IGNORECASE)
                    if match:
                        child_text, parent_text = match.group(1).strip().lstrip("- "), match.group(3).strip()
                        child_uuid, parent_uuid = concept_node_map.get(child_text), concept_node_map.get(parent_text)
                        if not parent_uuid and len(parent_text) < 80:
                            logger.info(f"Identified new parent concept: '{parent_text}'. Checking existing...")
                            similar_parents = self._search_similar_nodes(parent_text, k=1, node_type_filter='concept')
                            if similar_parents and similar_parents[0][1] <= concept_sim_threshold: parent_uuid = similar_parents[0][0]; logger.info(f"Found existing node for parent: {parent_uuid[:8]}"); concept_node_map[parent_text] = parent_uuid
                            else: logger.info(f"Adding new node for parent '{parent_text}'"); parent_uuid = self.add_memory_node(parent_text, "System", 'concept', base_strength=0.85);
                            if parent_uuid and parent_text not in concept_node_map: concept_node_map[parent_text] = parent_uuid
                        if child_uuid and parent_uuid and child_uuid in self.graph and parent_uuid in self.graph:
                            try:
                                if not self.graph.has_edge(parent_uuid, child_uuid): self.graph.add_edge(parent_uuid, child_uuid, type='HIERARCHICAL', base_strength=0.85, last_traversed_ts=current_time); logger.info(f"Added HIERARCHICAL edge: {parent_uuid[:8]} -> {child_uuid[:8]} ('{child_text}' is_a '{parent_text}')")
                                else: logger.debug(f"Hierarchical edge {parent_uuid[:8]}->{child_uuid[:8]} exists.")
                            except Exception as e: logger.error(f"Error adding hierarchical edge: {e}")
                        else: logger.warning(f"Could not find nodes for hierarchy: '{child_text}' is_a '{parent_text}'")
            else: logger.info("LLM reported no direct hierarchical relationships.")
        # --- 5. Pruning (Refined) ---
        # (Keep implementation from previous version)
        if summary_created and summary_node_uuid:
            logger.info(f"Pruning original turn nodes covered by summary {summary_node_uuid[:8]}...")
            pruned_count, failed_prune_count, skipped_prune_count = 0, 0, 0
            nodes_in_batch = set(node_uuids_to_process)
            for original_uuid in node_uuids_to_process:
                if original_uuid in self.graph:
                    is_safe_to_prune = True; significant_neighbors = 0
                    for neighbor in nx.all_neighbors(self.graph, original_uuid):
                        if neighbor == summary_node_uuid: continue
                        if neighbor in nodes_in_batch:
                            edge_data_fwd = self.graph.get_edge_data(original_uuid, neighbor)
                            edge_data_bwd = self.graph.get_edge_data(neighbor, original_uuid)
                            if (edge_data_fwd and edge_data_fwd.get('type') == 'TEMPORAL') or \
                               (edge_data_bwd and edge_data_bwd.get('type') == 'TEMPORAL'): continue
                        significant_neighbors += 1; break
                    if significant_neighbors > 0: is_safe_to_prune = False; skipped_prune_count += 1; logger.debug(f" Skipping prune {original_uuid[:8]}: {significant_neighbors} other structural neighbor(s).")
                    if is_safe_to_prune:
                         logger.debug(f" Attempting prune: {original_uuid[:8]}")
                         if self.delete_memory_entry(original_uuid): pruned_count += 1
                         else: failed_prune_count += 1
                else: logger.debug(f"Node {original_uuid[:8]} already gone before prune.")
            logger.info(f"Pruning done. Deleted: {pruned_count}, Skipped: {skipped_prune_count}, Failed: {failed_prune_count}.")
        else: logger.info("Skipping pruning: no summary node created.")

        logger.info("--- Consolidation Step Finished ---")
        self._save_memory() # Save after consolidation & pruning

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
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    try: os.remove(file_path); logger.info(f"Deleted: {file_path}")
                    except OSError as e: logger.error(f"Error deleting {file_path}: {e}")
                else: logger.debug(f"Not found, skip delete: {file_path}")
            logger.info("--- MEMORY RESET COMPLETE ---"); return True
        except Exception as e:
            logger.error(f"Error during memory reset: {e}", exc_info=True)
            self.graph = nx.DiGraph(); self.embeddings = {}; self.faiss_id_to_uuid = {}; self.uuid_to_faiss_id = {}; self.last_added_node_uuid = None; self.index = faiss.IndexFlatL2(self.embedding_dim); logger.warning("Reset failed, re-initialized empty state.")
            return False

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
        return file_manager.read_calendar_events(self.config, self.personality, target_date)

    def _call_kobold_multimodal_api(self, messages: list, max_tokens: int = 512, temperature: float = 0.7,
                                    top_p: float = 0.9) -> str:
        """Sends prompt to KoboldCpp OpenAI-compatible API, handles multimodal messages."""
        logger.debug(f"Calling Kobold Chat Completions API ({self.kobold_chat_api_url})")
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


# --- Example Usage Block ---
if __name__ == "__main__":
    # (Keep implementation from previous version)
    logger.info("Running basic test...")
    client = GraphMemoryClient()
    print("\n--- Initial State ---"); print(f"Nodes: {client.graph.number_of_nodes()}, Edges: {client.graph.number_of_edges()}, Embeds: {len(client.embeddings)}, FAISS: {client.index.ntotal if client.index else 'N/A'}"); print(f"Last node: {client.last_added_node_uuid}"); print("-" * 20)
    uuid1, uuid2, uuid3 = None, None, None; min_consolidation_nodes = client.config.get('consolidation', {}).get('min_nodes', 5)
    if client.graph.number_of_nodes() < 4: print("Adding nodes..."); ts1 = (datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat(); uuid1=client.add_memory_node("The quick brown fox", "User", timestamp=ts1); ts2 = (datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat(); uuid2=client.add_memory_node("jumped over the lazy dog", "AI", timestamp=ts2); uuid3=client.add_memory_node("Both are animals.", "User"); print(f" Added: {uuid1[:8]}, {uuid2[:8]}, {uuid3[:8]}"); client._save_memory()
    else: uuids = sorted(list(client.graph.nodes()), key=lambda u: client.graph.nodes[u].get('timestamp','')); uuid1=uuids[-3] if len(uuids)>=3 else None; uuid2=uuids[-2] if len(uuids)>=2 else None; uuid3=uuids[-1] if len(uuids)>=1 else None; print(f"Using existing nodes: {uuid1}, {uuid2}, {uuid3}")
    print("\n--- Testing Retrieval ---")
    if client.graph.number_of_nodes() > 0: query = "animal sounds"; print(f"Query: '{query}'"); initial_nodes = client._search_similar_nodes(query); initial_uuids = [uid for uid,score in initial_nodes]; print(f" Initial nodes: {initial_uuids}"); chain = client.retrieve_memory_chain(initial_uuids); print(f" Retrieved chain ({len(chain)}):"); [print(f"  UUID: {node['uuid'][:8]}, Act: {node['final_activation']:.3f}, TS: {node.get('timestamp')}, Text: '{node['text'][:40]}...'") for node in chain[:10]]
    else: print("Graph empty.")
    print("\n--- Testing Consolidation ---")
    if client.graph.number_of_nodes() >= min_consolidation_nodes: print("Running consolidation..."); client.run_consolidation(); print("--- State After Consolidation ---"); print(f"Nodes: {client.graph.number_of_nodes()}, Edges: {client.graph.number_of_edges()}"); client._save_memory()
    else: print(f"Skipping consolidation (requires >= {min_consolidation_nodes} nodes).")
    logger.info("Basic test finished.")