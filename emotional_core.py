import logging
import json
import re
import os
import typing
from typing import Tuple, Dict, Any, List, Optional

# --- Dependency Imports ---
# Transformer for basic emotion classification
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch # PyTorch is usually required by transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("transformers library not found. Basic emotion classification will be disabled. Run `pip install transformers torch`")
    pipeline = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False

# VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    logging.warning("vaderSentiment library not found. VADER sentiment analysis will be disabled. Run `pip install vaderSentiment`")
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False

# Type hinting for GraphMemoryClient (avoid circular import)
if typing.TYPE_CHECKING:
    from persistent_backend_graph import GraphMemoryClient

logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_EMOTIONAL_CORE_CONFIG = {
    "enabled": True,
    "emotion_model_name": "SamLowe/roberta-base-go_emotions", # Example model
    "analysis_prompt_file": "emotional_analysis_prompt.txt",
    "llm_task_name": "emotional_analysis", # Task name in main config's llm_models
    "mood_valence_factor": 0.3, # How much EmotionalCore valence influences final mood
    "mood_arousal_factor": 0.2, # How much EmotionalCore arousal influences final mood
    "store_insights_enabled": False, # Whether to write insights back to KG
    "basic_emotion_threshold": 0.3, # Min probability to consider a basic emotion present
    "vader_sentiment_threshold": 0.05, # Threshold for VADER compound score
    "needs_fears_prefs_definitions": {
        "needs": {
            "Competence": "Desire to feel capable, effective, and skilled.",
            "Esteem": "Desire for self-respect, recognition, and appreciation from others.",
            "Belonging": "Desire for connection, acceptance, and positive relationships.",
            "Autonomy": "Desire for self-direction, choice, and control over one's actions.",
            "Safety": "Desire for security, stability, predictability, and freedom from threat."
        },
        "fears": {
            "Failure": "Fear of not meeting expectations or achieving goals.",
            "Rejection": "Fear of being excluded, disapproved of, or abandoned.",
            "LossOfControl": "Fear of being powerless or unable to influence events.",
            "Threat": "Fear of harm, danger, or negative consequences."
        },
        "preferences": { # Updated structure
            "Clarity": {"type": "positive", "description": "Preference for clear, unambiguous communication."},
            "Politeness": {"type": "positive", "description": "Preference for respectful and courteous interaction."},
            "Ambiguity": {"type": "negative", "description": "Dislike of vague or unclear communication."},
            "Rudeness": {"type": "negative", "description": "Dislike of disrespectful or impolite interaction."}
        }
    }
}

class EmotionalCore:
    """
    Manages the analysis and simulation of the AI's emotional state,
    including basic emotions, needs, fears, and preferences.
    """
    def __init__(self, client: 'GraphMemoryClient', config: dict):
        """
        Initializes the EmotionalCore.

        Args:
            client: An instance of GraphMemoryClient for accessing KG and LLM calls.
            config: The main application configuration dictionary.
        """
        self.client = client
        self.main_config = config
        self.config = self._load_config()
        self.is_enabled = self.config.get("enabled", False)

        self.emotion_classifier = None
        self.sentiment_analyzer = None
        # Load definitions from the merged config
        needs_fears_prefs_defs = self.config.get("needs_fears_prefs_definitions", {})
        self.needs_defs = needs_fears_prefs_defs.get("needs", {})
        self.fears_defs = needs_fears_prefs_defs.get("fears", {})
        self.prefs_defs = needs_fears_prefs_defs.get("preferences", {})


        # Internal state tracking (example structure - adjust as needed)
        self.current_analysis_results = {
            "basic_emotions": {}, # {"emotion_label": probability}
            "sentiment": {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0},
            "triggered_needs": {}, # {"need_name": {"confidence": float, "rationale": str}}
            "triggered_fears": {},
            "triggered_preferences": {}, # {"pref_name": {"type": "positive/negative", "confidence": float, "rationale": str}}
        }
        self.derived_tendency = "neutral" # e.g., "defensive", "curious", "supportive"
        self.derived_mood_hints = {"valence": 0.0, "arousal": 0.0} # Hints to augment main mood

        if self.is_enabled:
            logger.info("Initializing EmotionalCore...")
            self._load_emotion_model()
            self._load_vader()
            logger.info("EmotionalCore initialized.")
        else:
            logger.info("EmotionalCore is disabled by configuration.")

    def _load_config(self) -> dict:
        """Loads the emotional_core specific configuration."""
        core_config = self.main_config.get('emotional_core', {})
        # Merge with defaults to ensure all keys exist
        merged_config = DEFAULT_EMOTIONAL_CORE_CONFIG.copy()
        # Deep merge for needs_fears_prefs_definitions if it exists in core_config
        if 'needs_fears_prefs_definitions' in core_config:
            merged_config['needs_fears_prefs_definitions'] = {
                **DEFAULT_EMOTIONAL_CORE_CONFIG.get('needs_fears_prefs_definitions', {}),
                **core_config.get('needs_fears_prefs_definitions', {})
            }
            # Further deep merge for needs, fears, preferences individually
            for key in ['needs', 'fears', 'preferences']:
                if key in core_config.get('needs_fears_prefs_definitions', {}):
                     merged_config['needs_fears_prefs_definitions'][key] = {
                         **DEFAULT_EMOTIONAL_CORE_CONFIG.get('needs_fears_prefs_definitions', {}).get(key, {}),
                         **core_config.get('needs_fears_prefs_definitions', {}).get(key, {})
                     }

        merged_config.update({k: v for k, v in core_config.items() if k != 'needs_fears_prefs_definitions'})

        logger.debug(f"EmotionalCore configuration loaded: {merged_config}")
        return merged_config

    def _load_emotion_model(self):
        """Loads the transformer model for basic emotion classification."""
        if not TRANSFORMERS_AVAILABLE or not self.is_enabled:
            logger.info("Skipping emotion model load (transformers unavailable or core disabled).")
            return

        model_name = self.config.get("emotion_model_name")
        if not model_name:
            logger.warning("No emotion_model_name specified in config. Cannot load emotion classifier.")
            return

        try:
            logger.info(f"Loading emotion classification pipeline: {model_name}...")
            # Determine device (use GPU if available and configured)
            # TODO: Add device selection logic based on config/availability
            device = 0 if torch.cuda.is_available() else -1 # Basic check: use GPU 0 if available
            logger.info(f"Using device {device} for emotion model.")

            # Load pipeline (handles model and tokenizer)
            # Using 'text-classification' pipeline is simpler
            self.emotion_classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name, # Often same as model
                device=device,
                top_k=None # Return probabilities for all labels
            )
            logger.info(f"Emotion classification pipeline loaded successfully for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load emotion classification pipeline '{model_name}': {e}", exc_info=True)
            self.emotion_classifier = None

    def _load_vader(self):
        """Initializes the VADER sentiment analyzer."""
        if not VADER_AVAILABLE or not self.is_enabled:
            logger.info("Skipping VADER load (library unavailable or core disabled).")
            return
        try:
            logger.info("Initializing VADER sentiment analyzer...")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}", exc_info=True)
            self.sentiment_analyzer = None

    def analyze_input(self, user_input: str, history_context: str, kg_context: str):
        """
        Performs the full emotional analysis of the user input and context.

        Args:
            user_input: The latest input text from the user.
            history_context: Recent conversation history formatted as a string.
            kg_context: Relevant context retrieved from the knowledge graph.
        """
        if not self.is_enabled:
            return # Do nothing if disabled

        logger.info(f"EmotionalCore analyzing input: '{user_input[:50]}...'")
        self.current_analysis_results = { # Reset results
            "basic_emotions": {}, "sentiment": {"compound": 0.0},
            "triggered_needs": {}, "triggered_fears": {}, "triggered_preferences": {}
        }

        # 1. Basic Emotion Classification
        self.current_analysis_results["basic_emotions"] = self._classify_basic_emotions(user_input)

        # 2. Sentiment Analysis (VADER)
        self.current_analysis_results["sentiment"] = self._analyze_sentiment_vader(user_input)

        # 3. LLM Interpretation for Needs, Fears, Preferences
        llm_interpretation = self._interpret_needs_fears_prefs(user_input, history_context, kg_context)
        if llm_interpretation:
            self.current_analysis_results["triggered_needs"] = llm_interpretation.get("needs", {})
            self.current_analysis_results["triggered_fears"] = llm_interpretation.get("fears", {})
            self.current_analysis_results["triggered_preferences"] = llm_interpretation.get("preferences", {})

        logger.debug(f"Emotional analysis complete. Results: {self.current_analysis_results}")

    def _classify_basic_emotions(self, text: str) -> Dict[str, float]:
        """Classifies basic emotions using the loaded transformer model."""
        if not self.emotion_classifier or not text:
            return {}

        try:
            results = self.emotion_classifier(text)
            # Results might be nested [[{'label': '...', 'score': ...}]]
            if isinstance(results, list) and results and isinstance(results[0], list):
                processed_results = {item['label']: item['score'] for item in results[0]}
                logger.debug(f"Basic emotion classification results: {processed_results}")
                return processed_results
            else:
                 logger.warning(f"Unexpected format from emotion classifier: {results}")
                 return {}
        except Exception as e:
            logger.error(f"Error during basic emotion classification: {e}", exc_info=True)
            return {}

    def _analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyzes sentiment using VADER."""
        if not self.sentiment_analyzer or not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0} # Return neutral default

        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            logger.debug(f"VADER sentiment scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error during VADER sentiment analysis: {e}", exc_info=True)
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}

    def _interpret_needs_fears_prefs(self, user_input: str, history_context: str, kg_context: str) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM call to interpret which needs, fears, or preferences
        are relevant to the current interaction.
        """
        prompt_file = self.config.get("analysis_prompt_file")
        llm_task = self.config.get("llm_task_name")

        if not prompt_file or not llm_task:
            logger.error("Missing analysis_prompt_file or llm_task_name in EmotionalCore config.")
            return None

        prompt_template = self.client._load_prompt(prompt_file) # Use client's loader
        if not prompt_template:
            logger.error(f"Failed to load emotional analysis prompt: {prompt_file}")
            return None

        # Format definitions for the prompt
        needs_str = "\n".join([f"- {name}: {desc}" for name, desc in self.needs_defs.items()])
        fears_str = "\n".join([f"- {name}: {desc}" for name, desc in self.fears_defs.items()])
        # Correctly format preferences based on the updated structure
        prefs_str_parts = []
        for name, pref_data in self.prefs_defs.items():
            if isinstance(pref_data, dict):
                pref_type = pref_data.get('type', 'N/A')
                pref_desc = pref_data.get('description', '')
                prefs_str_parts.append(f"- {name} ({pref_type}): {pref_desc}")
            else: # Fallback for old string format (though default is now dict)
                prefs_str_parts.append(f"- {name}: {pref_data}")
        prefs_str = "\n".join(prefs_str_parts)


        try:
            full_prompt = prompt_template.format(
                user_input=user_input,
                history_context=history_context,
                kg_context=kg_context,
                needs_definitions=needs_str,
                fears_definitions=fears_str,
                preferences_definitions=prefs_str
            )
        except KeyError as e:
            logger.error(f"Missing placeholder in emotional analysis prompt template '{prompt_file}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error formatting emotional analysis prompt: {e}", exc_info=True)
            return None

        logger.debug(f"Sending emotional analysis LLM prompt (Task: {llm_task}):\n{full_prompt[:500]}...")
        llm_response = self.client._call_configured_llm(llm_task, prompt=full_prompt)

        if not llm_response or llm_response.startswith("Error:"):
            logger.error(f"Emotional analysis LLM call failed: {llm_response}")
            return None

        # Parse the expected JSON output
        try:
            logger.debug(f"Raw emotional analysis LLM response:  ```{llm_response}```")
            # Extract JSON (assuming it's enclosed in ```json ... ``` or just {})
            match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*?\}', llm_response, re.DOTALL)
            if match:
                json_str = match.group(1) or match.group(0) # Get content from group 1 if exists, else group 0
                parsed_data = json.loads(json_str)
                logger.info(f"Parsed emotional interpretation from LLM: {parsed_data}")
                # Basic validation (can be expanded)
                if isinstance(parsed_data, dict) and all(k in parsed_data for k in ["needs", "fears", "preferences"]):
                    return parsed_data
                else:
                    logger.warning("Parsed emotional interpretation JSON has incorrect structure.")
                    return None
            else:
                logger.warning("Could not extract JSON from emotional analysis LLM response.")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from emotional analysis LLM response: {e}. Raw: '{llm_response}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing emotional analysis LLM response: {e}", exc_info=True)
            return None

    def aggregate_and_combine(self) -> Tuple[str, Dict[str, float]]:
        """
        Aggregates the analysis results and determines a dominant emotional
        tendency and mood hints (valence/arousal adjustments).

        Returns:
            Tuple[str, Dict[str, float]]: (derived_tendency, derived_mood_hints)
        """
        if not self.is_enabled:
            return "neutral", {"valence": 0.0, "arousal": 0.0}

        # --- Simple Aggregation Logic (Example - Needs Refinement) ---
        valence_hint = 0.0
        arousal_hint = 0.0
        tendency = "neutral"

        # 1. Sentiment Influence
        sentiment_compound = self.current_analysis_results["sentiment"].get("compound", 0.0)
        valence_hint += sentiment_compound * 0.5 # VADER compound score directly influences valence hint

        # 2. Basic Emotion Influence
        basic_emotions = self.current_analysis_results["basic_emotions"]
        threshold = self.config.get("basic_emotion_threshold", 0.3)
        dominant_emotion = None
        max_prob = 0.0

        # Simple valence/arousal mapping for basic emotions (adjust weights as needed)
        emotion_va_map = {
            'joy': (0.8, 0.4), 'love': (0.7, 0.5), 'optimism': (0.6, 0.3),
            'sadness': (-0.7, -0.3), 'anger': (-0.6, 0.7), 'fear': (-0.5, 0.8),
            'pessimism': (-0.5, -0.2), 'disgust': (-0.6, 0.5),
            'surprise': (0.2, 0.6), # Surprise can be positive or negative arousal
            'neutral': (0.0, 0.0),
            # Add mappings for other labels from the model if needed
            'admiration': (0.6, 0.3), 'amusement': (0.5, 0.4), 'approval': (0.4, 0.2),
            'caring': (0.5, 0.2), 'confusion': (-0.2, 0.4), 'curiosity': (0.3, 0.5),
            'desire': (0.4, 0.6), 'disappointment': (-0.4, -0.1), 'disapproval': (-0.3, 0.3),
            'embarrassment': (-0.3, 0.4), 'excitement': (0.6, 0.7), 'gratitude': (0.7, 0.3),
            'grief': (-0.8, -0.4), 'nervousness': (-0.2, 0.6), 'pride': (0.7, 0.4),
            'realization': (0.1, 0.3), 'relief': (0.5, -0.2), 'remorse': (-0.4, -0.1),
        }

        for emotion, prob in basic_emotions.items():
            if prob >= threshold:
                if prob > max_prob:
                    max_prob = prob
                    dominant_emotion = emotion.lower() # Use lowercase for map lookup

                # Add weighted influence to mood hints based on map
                va = emotion_va_map.get(emotion.lower(), (0.0, 0.0))
                valence_hint += va[0] * prob * 0.4 # Weight influence by probability
                arousal_hint += va[1] * prob * 0.3

        # 3. Needs/Fears/Preferences Influence (Simplified)
        # Example: Unmet need -> negative valence, slight arousal increase
        # Example: Triggered fear -> negative valence, high arousal increase
        # Example: Violated preference -> slight negative valence
        for need, data in self.current_analysis_results["triggered_needs"].items():
            if data.get("confidence", 0.0) > 0.5: # Example threshold
                valence_hint -= 0.1 # Unmet need is slightly negative
                arousal_hint += 0.05

        for fear, data in self.current_analysis_results["triggered_fears"].items():
            if data.get("confidence", 0.0) > 0.5:
                valence_hint -= 0.2 # Fear is more negative
                arousal_hint += 0.15

        for pref, data in self.current_analysis_results["triggered_preferences"].items():
             if data.get("confidence", 0.0) > 0.5:
                 pref_type = data.get("type", "positive") # Assume positive if missing
                 if pref_type == "negative": # Violated preference
                     valence_hint -= 0.05

        # 4. Determine Dominant Tendency (Very Basic Example)
        if dominant_emotion == 'anger' or dominant_emotion == 'disgust':
            tendency = "annoyed"
        elif dominant_emotion == 'fear' or self.current_analysis_results["triggered_fears"]:
            tendency = "cautious"
        elif dominant_emotion == 'sadness':
            tendency = "subdued"
        elif dominant_emotion == 'joy' or dominant_emotion == 'love':
            tendency = "positive"
        elif self.current_analysis_results["triggered_needs"].get("Competence", {}).get("confidence", 0) > 0.6:
             tendency = "focused" # Example based on need
        # Add more complex logic based on combinations

        # Clamp mood hints
        self.derived_mood_hints["valence"] = max(-1.0, min(1.0, valence_hint))
        self.derived_mood_hints["arousal"] = max(-1.0, min(1.0, arousal_hint)) # Allow negative arousal hint for calming effect? Or clamp 0-1? Let's clamp 0-1 for now.
        self.derived_mood_hints["arousal"] = max(0.0, min(1.0, arousal_hint))

        self.derived_tendency = tendency

        logger.info(f"Emotional aggregation complete: Tendency='{self.derived_tendency}', MoodHints={self.derived_mood_hints}")
        return self.derived_tendency, self.derived_mood_hints

    def craft_prompt_instructions(self) -> str:
        """
        Generates system prompt instructions based on the derived emotional state.
        """
        if not self.is_enabled:
            return ""

        # --- Example Instruction Crafting ---
        instructions = []
        tendency = self.derived_tendency
        valence = self.derived_mood_hints["valence"]
        arousal = self.derived_mood_hints["arousal"]

        # Base instruction
        instructions.append(f"[System Note: Your current internal emotional tendency is '{tendency}'. Shape your response tone accordingly.]")

        # Add more specific instructions based on state
        if tendency == "cautious" or valence < -0.3:
            instructions.append("[System Note: Respond cautiously. Avoid making strong commitments or definitive statements if unsure. Prioritize safety and clarity.]")
        elif tendency == "annoyed" or valence < -0.2 and arousal > 0.5:
            instructions.append("[System Note: Respond concisely and perhaps slightly formally. Avoid overly enthusiastic or effusive language.]")
        elif tendency == "positive" or valence > 0.5:
            instructions.append("[System Note: Respond warmly and positively. Use encouraging language.]")

        # Add instructions based on triggered needs/fears (if available and strong)
        if self.current_analysis_results["triggered_fears"].get("Rejection", {}).get("confidence", 0) > 0.7:
             instructions.append("[System Note: User might be sensitive to rejection. Respond reassuringly and inclusively.]")
        if self.current_analysis_results["triggered_needs"].get("Competence", {}).get("confidence", 0) > 0.7:
             instructions.append("[System Note: User values competence. Provide clear, accurate, and helpful information. Acknowledge their skills if appropriate.]")

        # Combine instructions
        instruction_str = "\n".join(instructions)
        logger.info(f"Crafted emotional prompt instructions:\n{instruction_str}")
        return instruction_str

    def update_memory_with_emotional_insight(self):
        """
        (Optional) Analyzes the current emotional state and potentially writes
        significant findings back to the knowledge graph.
        """
        if not self.is_enabled or not self.config.get("store_insights_enabled", False):
            return

        logger.info("Checking for significant emotional insights to store...")
        # --- Example Logic ---
        # - If a strong fear was triggered consistently with a topic -> add relationship?
        # - If a strong positive reaction (preference) occurred -> add preference node/relationship?
        # - If a significant mood shift happened -> add event node?

        # This requires defining what constitutes a "significant insight" and
        # implementing the corresponding KG update methods in the client.
        # For now, this is a placeholder.
        pass

