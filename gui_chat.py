# gui_chat.py
import sys
import os
import base64
from datetime import datetime, timezone, timedelta
import re
import logging
import yaml
import mimetypes  # <<< Add mimetypes

# Removed incorrect import: from pip._internal.utils import urls

# Import zoneinfo safely
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    print("WARNING: zoneinfo module not found. Timestamps will use UTC. Consider `pip install tzdata`.",
          file=sys.stderr)
    ZoneInfo = None  # type: ignore
    ZoneInfoNotFoundError = Exception  # Placeholder

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser, QLineEdit,  # Added QTextBrowser
    QPushButton, QScrollArea, QLabel, QHBoxLayout, QFrame,
    QSizePolicy, QSpacerItem, QMessageBox, QInputDialog,QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, pyqtSlot, QMimeData, QUrl, QBuffer, QByteArray, \
    QIODevice
from PyQt6.QtGui import QFont, QAction, QActionGroup, QDragEnterEvent, QDropEvent, \
    QDragMoveEvent, QPixmap, QImage, QKeyEvent, QKeySequence, QDesktopServices  # Added QDesktopServices

from persistent_backend_graph import GraphMemoryClient, logger as backend_logger, logger, strip_emojis # <<< Import strip_emojis
import file_manager

# --- Logger Setup ---
gui_logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - GUI - %(levelname)s - %(message)s')
else:
    gui_logger.setLevel(logging.DEBUG)

# --- Constants ---
DEFAULT_CONFIG_PATH = "config.yaml"
FALLBACK_MODIFICATION_KEYWORDS = ["forget", "delete", "remove", "edit", "change", "correct", "update"]


# --- Helper Function ---
def get_available_personalities(config_path=DEFAULT_CONFIG_PATH):
    # (Implementation remains the same as previous version)
    personalities = []
    default_personality = None  # Track default separately
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        base_path = config.get('base_memory_path', 'memory_sets')
        default_personality = config.get('default_personality')  # Don't assume 'default'

        # Add explicitly listed personalities first (maintaining order if specified)
        config_personalities = config.get('available_personalities', [])
        if isinstance(config_personalities, list):
            for p in config_personalities:
                if p not in personalities:
                    personalities.append(p)

        # Discover personalities from directories
        if base_path and os.path.isdir(base_path):
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    if item not in personalities:  # Add only if not already listed
                        personalities.append(item)

        # Ensure the default from config is present if specified, add if missing
        if default_personality and default_personality not in personalities:
            personalities.insert(0, default_personality)  # Add to beginning if missing

    except Exception as e:
        gui_logger.error(f"Error discovering personalities: {e}. Using minimal fallback.")
        if default_personality and default_personality not in personalities:
            personalities.append(default_personality)
        if not personalities:
            personalities = ['default']  # Absolute fallback

    gui_logger.info(f"Discovered personalities: {personalities}")
    return personalities


# --- Worker Thread Signals ---
class WorkerSignals(QObject):
    # Added backend_ready signal
    backend_ready = pyqtSignal(bool, str)  # success_flag, personality_name
    # Signature changed: Replaced action_result_info object with needs_planning flag (bool)
    response_ready = pyqtSignal(str, str, list, str, bool)
    modification_response_ready = pyqtSignal(str, str, str, str) # Keep for memory mods AND workspace results
    memory_reset_complete = pyqtSignal()
    consolidation_complete = pyqtSignal(str)
    clarification_needed = pyqtSignal(str, list)
    confirmation_needed = pyqtSignal(str, dict)  # NEW: action_type, details_dict
    feedback_provided = pyqtSignal(str, str)  # NEW: node_uuid, feedback_type ('up'/'down')
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    initial_history_ready = pyqtSignal(list) # NEW: Signal for initial history


# --- Worker Thread ---
class Worker(QThread):
    # --- Add class-level type hint for signals ---
    signals: WorkerSignals

    def __init__(self, personality_name, config_path=DEFAULT_CONFIG_PATH):
        super().__init__()
        self.signals = WorkerSignals() # Instantiation remains the same
        self.client = None
        self.current_conversation = []
        self.is_running = True
        self.input_queue = []  # List to hold tasks: (task_type, data)
        self.mod_keywords = []
        self.consolidation_trigger_count = 0  # Interactions before auto-consolidation
        self.forgetting_trigger_count = 0  # Interactions before auto-forgetting
        self.interaction_count = 0  # Tracks user/AI turns since last maintenance/consolidation trigger
        self.personality = personality_name
        self.config_path = config_path
        self.pending_clarification = None  # Store pending action details {'original_action': str, 'args': dict, 'missing_args': list}
        self.pending_confirmation = None  # NEW: Store pending confirmation details {'action': str, 'args': dict}

        # Load config for keywords and trigger counts
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            loaded_keywords = config.get('modification_keywords', FALLBACK_MODIFICATION_KEYWORDS)
            self.mod_keywords = [str(kw).lower() for kw in loaded_keywords] if isinstance(loaded_keywords,
                                                                                          list) else FALLBACK_MODIFICATION_KEYWORDS
            gui_logger.info(f"Worker({self.personality}) loaded keywords: {self.mod_keywords}")

            # Load forgetting trigger count (used for memory maintenance)
            forgetting_cfg = config.get('forgetting', {})
            self.forgetting_trigger_count = int(forgetting_cfg.get('trigger_interaction_count', 0))
            gui_logger.info(f"Worker({self.personality}) Forgetting trigger count: {self.forgetting_trigger_count}")

            # Load consolidation trigger count
            consolidation_cfg = config.get('consolidation', {})
            self.consolidation_trigger_count = int(
                consolidation_cfg.get('trigger_interaction_count', 0))  # Default 0 (disabled)
            gui_logger.info(
                f"Worker({self.personality}) Consolidation trigger count: {self.consolidation_trigger_count}")

        except Exception as e:
            gui_logger.error(f"Error loading config for worker: {e}. Using fallbacks.", exc_info=True)
            self.mod_keywords = FALLBACK_MODIFICATION_KEYWORDS
            self.consolidation_trigger_count = 0
            self.forgetting_trigger_count = 0

    def run(self):
        """Initializes backend and processes tasks from the queue."""
        initialized_successfully = False
        try:
            gui_logger.info(f"Worker run: Initializing backend for '{self.personality}'...")
            # Ensure GraphMemoryClient exists and is imported
            self.client = GraphMemoryClient(config_path=self.config_path, personality_name=self.personality)
            initialized_successfully = True
            self.signals.log_message.emit(f"Backend for '{self.personality}' initialized.")
            # --- Get and emit initial history ---
            try:
                initial_history = self.client.get_initial_history()
                if initial_history:
                    self.current_conversation = initial_history # Pre-populate conversation history
                    self.signals.initial_history_ready.emit(initial_history)
                    gui_logger.info(f"Worker emitted {len(initial_history)} initial history turns.")
                else:
                    self.current_conversation = [] # Start fresh if no history
            except Exception as hist_e:
                gui_logger.error(f"Error getting initial history from backend: {hist_e}", exc_info=True)
                self.current_conversation = [] # Start fresh on error
            # --- End initial history handling ---

            # --- Check for and emit pending re-greeting ---
            try:
                pending_greeting = self.client.get_pending_re_greeting()
                if pending_greeting:
                    gui_logger.info(f"Worker found pending re-greeting: '{pending_greeting[:50]}...'")
                    greeting_timestamp = datetime.now(timezone.utc).isoformat()
                    # Add greeting to conversation history (no UUID needed here as it wasn't added to graph yet)
                    greeting_turn = {"speaker": "AI", "text": pending_greeting, "timestamp": greeting_timestamp, "uuid": None}
                    self.current_conversation.append(greeting_turn)
                    # Emit response_ready to display the greeting in the GUI
                    # User input is empty, no memories, no AI node UUID yet, no planning needed
                    self.signals.response_ready.emit("", pending_greeting, [], None, False)
                else:
                    gui_logger.debug("No pending re-greeting found.")
            except Exception as greet_e:
                gui_logger.error(f"Error checking/emitting pending re-greeting: {greet_e}", exc_info=True)
            # --- End re-greeting check ---

            # --- Signal backend is ready AFTER history and potential greeting are handled ---
            self.signals.backend_ready.emit(True, self.personality)
        except Exception as e:
            error_msg = f"FATAL: Failed initialize backend for '{self.personality}': {e}"
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg, exc_info=True)
            self.signals.backend_ready.emit(False, self.personality)
            self.is_running = False
            return

        # --- Main Processing Loop ---
        while self.is_running:
            if self.input_queue:
                task_type, data = self.input_queue.pop(0)
                try:
                    gui_logger.debug(f"Worker processing task: {task_type}")
                    if task_type == 'chat':
                        self.handle_chat_task(data)
                    elif task_type == 'modify':
                        self.handle_modify_task(data)
                    elif task_type == 'reset':
                        self.handle_reset_task()
                    elif task_type == 'consolidate':
                        self.handle_consolidation_task()
                    elif task_type == 'execute_action':
                        self.handle_execute_action_task(data)
                    elif task_type == 'execute_action_confirmed':
                        self.handle_confirmed_action_task(data)
                    elif task_type == 'saliency_update':
                        self.handle_saliency_update_task(data)
                    elif task_type == 'feedback':
                        self.handle_feedback_task(data)
                    # --- Add handler for memory maintenance ---
                    elif task_type == 'memory_maintenance':
                        self.handle_memory_maintenance_task()
                    # --- NEW: Add handler for separate planning task ---
                    elif task_type == 'plan_and_execute_workspace':
                        self.handle_plan_and_execute_task(data)
                    else:
                        gui_logger.warning(f"Unknown task type received in worker queue: {task_type}")

                except Exception as e:
                    # Catch errors within specific task handlers if possible,
                    # this is a fallback for unexpected errors in the loop itself.
                    error_msg = f"Unexpected error handling task '{task_type}': {e}"
                    self.signals.error.emit(error_msg)
                    backend_logger.error(error_msg, exc_info=True)
            else:
                # Prevent busy-waiting
                self.msleep(100)  # Sleep for 100 milliseconds if queue is empty

        # --- Cleanup when loop ends ---
        self.save_memory_on_stop()
        self.signals.log_message.emit(f"Worker processing stopped for '{self.personality}'.")

    def handle_chat_task(self, data: dict):
        """Handles chat tasks, potentially with attachments, and triggers maintenance."""
        user_input_text = data.get('text', '')
        attachment = data.get('attachment')

        user_timestamp = datetime.now(timezone.utc).isoformat()
        history_text = user_input_text
        if attachment and attachment.get('type') == 'image':
            placeholder = f" [Image: {attachment.get('filename', 'Attached')}]"
            separator = " " if history_text else ""
            history_text += separator + placeholder

        # --- Add user turn to history (WITH UUID if available later) ---
        # We need the UUID *after* adding to graph, so add placeholder for now
        user_turn_data = {"speaker": "User", "text": history_text, "timestamp": user_timestamp, "uuid": None}
        self.current_conversation.append(user_turn_data)
        self.signals.log_message.emit(f"Processing chat: {strip_emojis(history_text[:30])}...") # Strip emojis

        ai_response_text = "Error: Could not get response."
        memory_chain_data = []
        ai_node_uuid = None  # Initialize AI node UUID
        interaction_successful = False  # Flag to track if LLM call succeeded

        try:
            # process_interaction now returns (response, memories, ai_node_uuid, needs_planning_flag)
            interaction_result = self.client.process_interaction(
                user_input=user_input_text,
                conversation_history=self.current_conversation, # Pass only history up to user turn
                attachment_data=attachment
            )

            # --- Defensive Check ---
            if interaction_result is None or not isinstance(interaction_result, tuple) or len(interaction_result) != 4:
                error_msg = f"FATAL: process_interaction returned invalid data: {interaction_result} (Type: {type(interaction_result)})"
                backend_logger.error(error_msg)
                self.signals.error.emit(error_msg)
                # Emit default error response
                self.signals.response_ready.emit(history_text, "Error: Backend interaction failed unexpectedly.", [], None, False)
                interaction_successful = False
                # Skip further processing in the try block for this interaction
                raise Exception("Backend interaction returned invalid data.") # Raise exception to trigger the except block below

            # --- Unpack if valid ---
            ai_response_text, memory_chain_data, ai_node_uuid, needs_planning = interaction_result
            gui_logger.debug(f"Successfully unpacked interaction result. Needs Planning: {needs_planning}")

            # Add AI response to worker's history
            ai_timestamp = datetime.now(timezone.utc).isoformat()
            # --- Add AI response to history (WITH UUID if available later) ---
            ai_turn_data = {"speaker": "AI", "text": ai_response_text, "timestamp": ai_timestamp, "uuid": None}
            self.current_conversation.append(ai_turn_data)

            # --- Get UUIDs from backend (if returned) and update history ---
            # This requires process_interaction to return the UUIDs
            # For now, assume they are NOT returned and priming uses last N turns implicitly
            # If backend is modified later, update here:
            # user_uuid_from_backend = ...
            # ai_uuid_from_backend = ...
            # if user_uuid_from_backend: user_turn_data['uuid'] = user_uuid_from_backend
            # if ai_uuid_from_backend: ai_turn_data['uuid'] = ai_uuid_from_backend # No longer needed here
            if ai_node_uuid:
                ai_turn_data['uuid'] = ai_node_uuid  # Store UUID if returned
                gui_logger.debug(f"Received AI node UUID from backend: {ai_node_uuid}")

            # Emit result signal (including AI node UUID and needs_planning flag)
            self.signals.response_ready.emit(history_text, ai_response_text, memory_chain_data, ai_node_uuid, needs_planning)

            # If planning is needed, queue the separate task
            if needs_planning:
                gui_logger.info("Flag indicates workspace planning needed. Queuing separate task.")
                # Pass necessary context to the planning task
                # For now, just pass the original user input text.
                # Could also pass relevant memory UUIDs if needed later.
                planning_context = {'user_input': user_input_text, 'history': self.current_conversation[-5:]} # Pass recent history too
                self.input_queue.append(('plan_and_execute_workspace', planning_context))

            interaction_successful = True  # Mark as successful

        except Exception as e:
            error_msg = f"Error during chat processing: {e}"
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg, exc_info=True)
            error_timestamp = datetime.now(timezone.utc).isoformat()
            self.current_conversation.append(
                {"speaker": "Error", "text": f"Failed to process interaction: {e}", "timestamp": error_timestamp})
            # Emit error response, providing defaults for missing arguments
            self.signals.response_ready.emit(history_text, f"Error generating response: {e}", [], None, False) # Pass None for UUID, False for needs_planning
            interaction_successful = False  # Mark as failed

        # --- Trigger Maintenance Tasks AFTER interaction attempt ---
        if interaction_successful:
            self.interaction_count += 1
            gui_logger.debug(f"Interaction count for '{self.personality}': {self.interaction_count}")

            maintenance_task_queued = False  # Flag to track if any task was queued

            # Check Forgetting Trigger
            # Ensure trigger count is positive (enabled)
            if self.forgetting_trigger_count > 0 and self.interaction_count >= self.forgetting_trigger_count:
                gui_logger.info(
                    f"Forgetting trigger count ({self.forgetting_trigger_count}) reached. Queuing memory maintenance task.")
                self.input_queue.append(('memory_maintenance', None))  # Add maintenance task
                maintenance_task_queued = True

            # --- Check Consolidation Trigger ---
            # Ensure trigger count is positive (enabled)
            # Check even if forgetting triggered, in case counts are the same or consolidation is lower
            if self.consolidation_trigger_count > 0 and self.interaction_count >= self.consolidation_trigger_count:
                gui_logger.info(
                    f"Consolidation trigger count ({self.consolidation_trigger_count}) reached. Queuing consolidation task.")
                self.input_queue.append(('consolidate', True))  # Pass True for automatic trigger
                maintenance_task_queued = True

            # Reset counter if EITHER task was queued to prevent immediate re-triggering
            if maintenance_task_queued:
                gui_logger.debug("Resetting interaction counter after queuing maintenance task(s).")
                self.interaction_count = 0

    def handle_memory_maintenance_task(self):
        """Handles the task for running the nuanced forgetting process."""
        self.signals.log_message.emit("[Auto] Running memory maintenance (forgetting)...")
        backend_logger.info(f"[Auto] Worker running memory maintenance for '{self.personality}'.")
        if self.client:
            try:
                # Call the backend method responsible for forgetting/archiving
                self.client.run_memory_maintenance()
                # Optionally emit a signal if specific feedback is needed?
                # self.signals.maintenance_complete.emit("Memory maintenance finished.")
                self.signals.log_message.emit("[Auto] Memory maintenance finished.")  # Simple log message for now
            except Exception as e:
                error_msg = f"Error during automatic memory maintenance: {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
        else:
            error_msg = "Backend client not available for memory maintenance."
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg)

    def handle_feedback_task(self, data: dict):
        """Handles the task to apply user feedback to a node via the backend."""
        uuid = data.get('uuid')
        feedback_type = data.get('type')
        if not uuid or not feedback_type:
            gui_logger.error(f"Invalid data for feedback task: {data}")
            return

        gui_logger.info(f"Worker handling feedback: UUID={uuid}, Type={feedback_type}")
        if self.client:
            try:
                # Call the backend method
                self.client.apply_feedback(uuid, feedback_type)
                # Optionally emit a signal back to GUI? For now, just log.
                self.signals.log_message.emit(f"Feedback processed for {uuid[:8]}.")
            except Exception as e:
                error_msg = f"Error during feedback processing for {uuid}: {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
        else:
            error_msg = "Backend client not available for feedback processing."
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg)

    # --- NEW Task Handler for Planning/Execution ---
    # --- NEW Task Handler for Planning/Execution ---
    def handle_plan_and_execute_task(self, context_data: dict):
        """Handles the separate task for planning and executing workspace actions."""
        user_input = context_data.get('user_input', '')
        history_context = context_data.get('history', []) # Get history context
        gui_logger.info(f"Worker handling plan_and_execute task for input: '{strip_emojis(user_input[:50])}...'") # Strip emojis
        self.signals.log_message.emit("Planning workspace actions...") # Update status

        if self.client:
            try:
                # Call the new backend method
                workspace_results = self.client.plan_and_execute(user_input, history_context)

                # Emit results using modification_response_ready signal for each result, unless silent & successful
                if workspace_results:
                    gui_logger.info(f"Received {len(workspace_results)} results from plan_and_execute.")
                    actions_reported = 0
                    for result_tuple in workspace_results:
                        # Expecting 4-tuple: (success, message, action_suffix, silent_and_successful)
                        if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
                            success, message, action_suffix, silent_and_successful = result_tuple

                            if silent_and_successful:
                                gui_logger.info(f"Skipping GUI notification for silent successful action: {action_suffix}")
                                continue # Skip emitting signal for this action

                            # If not silent or if failed, emit the signal
                            actions_reported += 1
                            action_name = action_suffix.split('_')[0] if '_' in action_suffix else action_suffix
                            placeholder_input = f"Workspace Action: {action_name}"
                            placeholder_target = "" # Target info not directly available here
                            self.signals.modification_response_ready.emit(placeholder_input, strip_emojis(message), action_suffix, placeholder_target) # Strip emojis from message
                        else:
                            gui_logger.error(f"Invalid result format (expected 4-tuple) from plan_and_execute: {result_tuple}")
                            self.signals.error.emit(f"Received invalid workspace result format: {result_tuple}")

                    if actions_reported > 0:
                        self.signals.log_message.emit(f"Workspace actions finished ({actions_reported} reported).")
                    else:
                        # This case means either no actions were planned, or all planned actions were silent and successful.
                        gui_logger.info("No workspace action results to report to GUI (empty plan or all silent successes).")
                        self.signals.log_message.emit("Workspace actions finished silently.") # Provide some feedback
                else:
                    gui_logger.info("plan_and_execute returned no results (no plan or empty plan).")
                    self.signals.log_message.emit("No workspace actions were needed.")

            except Exception as e:
                error_msg = f"Error during workspace plan/execute task: {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
        else:
            error_msg = "Backend client not available for workspace planning."
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg)


    def handle_modify_task(self, user_input):
        # (Implementation remains the same - no interaction counter increment)
        user_timestamp = datetime.now(timezone.utc).isoformat()
        # self.current_conversation.append({"speaker": "User", "text": user_input, "timestamp": user_timestamp}) # Maybe don't add modification command itself to history? Or add differently?
        self.signals.log_message.emit(f"Processing modification: {strip_emojis(user_input[:30])}...") # Strip emojis
        final_confirmation_msg = "Could not understand modification request."
        final_action_type = "error";
        final_target_info = "";
        ok = False;
        msg = final_confirmation_msg
        try:
            # --- Call backend analysis ---
            action_data = self.client.analyze_memory_modification_request(user_input)
            if isinstance(action_data, dict) and 'action' in action_data:
                action = action_data.get('action');
                final_action_type = action
                target_uuid = action_data.get('target_uuid');
                target_desc = action_data.get('target')
                new_text = action_data.get('new_text');
                topic = action_data.get('topic')

                # Determine target info string for logging/signal
                if action in ["delete", "edit"]:
                    final_target_info = target_uuid or target_desc or "?"
                elif action == "forget":
                    final_target_info = topic or "?"

                # --- Execute backend modification ---
                if action == "delete":
                    if target_uuid:
                        ok = self.client.delete_memory_entry(target_uuid)
                        msg = f"Deleted entry: {target_uuid[:8]}..." if ok else f"Failed to delete entry: {target_uuid[:8]}..."
                    else:
                        msg = f"Cannot delete: Please specify the exact UUID of the memory entry."; final_action_type = "delete_clarify"
                elif action == "edit":
                    if target_uuid and new_text is not None:  # Allow empty string edit? Yes.
                        new_uuid = self.client.edit_memory_entry(target_uuid, new_text)
                        ok = bool(new_uuid)
                        msg = f"Edited entry {target_uuid[:8]}. New ID: {new_uuid[:8]}" if ok else f"Failed to edit entry: {target_uuid[:8]}..."
                        final_target_info = new_uuid[:8] if ok else target_uuid[:8]  # Show new or old UUID
                    elif not target_uuid:
                        msg = f"Cannot edit: Please specify the exact UUID of the memory entry."; final_action_type = "edit_clarify"
                    else:
                        msg = "Cannot edit: Please provide the new text after the UUID."; final_action_type = "edit_clarify"
                elif action == "forget":
                    if topic:
                        ok, msg = self.client.forget_topic(topic)  # forget_topic returns (bool, message)
                    else:
                        msg = "Cannot forget: Please specify the topic to forget."; final_action_type = "forget_clarify"
                elif action == "none":
                    ok, msg = True, "No memory modification action taken."
                elif action == "error":
                    ok, msg = False, f"Analysis Error: {action_data.get('reason', 'Unknown')}"
                else:
                    ok, msg = False, f"Unknown action type '{action}' received from analysis."

                final_confirmation_msg = msg
                # Refine suffix logic
                if action not in ["none", "error", "unknown"] and "clarify" not in action:
                    final_action_type = f"{action}_{'success' if ok else 'fail'}"

            else:
                final_confirmation_msg = "Failed to analyze modification request structure."
                final_action_type = "analysis_fail"

        except Exception as e:
            error_msg = f"Error during modification processing: {e}"
            final_confirmation_msg = f"An internal error occurred during modification: {e}"
            final_action_type = "processing_exception"
            backend_logger.error(error_msg, exc_info=True)

        # Append system response to conversation history
        confirmation_timestamp = datetime.now(timezone.utc).isoformat()
        self.current_conversation.append(
            {"speaker": "System", "text": final_confirmation_msg, "timestamp": confirmation_timestamp})
        # Emit signal to GUI
        self.signals.modification_response_ready.emit(strip_emojis(user_input), strip_emojis(final_confirmation_msg), final_action_type,
                                                      str(final_target_info)) # Strip emojis

    def handle_reset_task(self):
        # (Implementation remains the same - no interaction counter increment)
        self.signals.log_message.emit("Resetting memory...")
        backend_logger.info("Worker received reset request.")
        if self.client:
            try:
                reset_ok = self.client.reset_memory();
            except Exception as e:
                error_msg = f"Error calling backend reset_memory: {e}"; self.signals.error.emit(
                    error_msg); backend_logger.error(error_msg, exc_info=True); return
            if reset_ok:
                self.current_conversation.clear() # Clear worker's history too
                self.interaction_count = 0  # Also reset interaction count on manual memory reset
                backend_logger.info("Memory reset successful.")
                self.signals.memory_reset_complete.emit()
            else:
                self.signals.error.emit("Backend failed to reset memory.")
        else:
            self.signals.error.emit("Backend client not available for reset.")

    def handle_consolidation_task(self, triggered_automatically=False):
        """Handles the task for running memory consolidation."""
        prefix = "[Auto] " if triggered_automatically else "[Manual] "
        self.signals.log_message.emit(f"{prefix}Running memory consolidation...")
        backend_logger.info(f"{prefix}Worker received consolidation request for '{self.personality}'.")
        if self.client:
            try:
                # --- Ensure the correct method name is called ---
                self.client.run_consolidation()  # <<< This call
                # --- End Correction ---
            except AttributeError as e:
                # Catch the specific error if the method is still missing
                error_msg = f"Error during consolidation: Method 'run_consolidation' not found on client. {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
                return
            except Exception as e:
                # Catch other errors during execution
                error_msg = f"Error during consolidation: {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
                return  # Stop if error occurs

            # This logic correctly handles automatic vs manual triggers for logging/signals
            if triggered_automatically:
                # Counter reset happens in handle_chat_task *before* automatic triggering logic queueing the task
                gui_logger.info("Automatic consolidation finished.")  # Log completion
            self.signals.consolidation_complete.emit(f"{prefix}Consolidation finished.")  # Signal GUI
        else:
            self.signals.error.emit("Backend client not available for consolidation.")

    def handle_execute_action_task(self, action_data):
        # (Implementation remains the same - no interaction counter increment)
        action = action_data.get('action', 'unknown');
        self.signals.log_message.emit(f"Executing action: {action}...")
        if self.client:
            success = False;
            message = f"Failed to execute action '{action}'.";
            action_suffix = f"{action}_fail"
            try:
                # Backend now returns (bool, message, suffix) or dict for confirmation
                backend_response = self.client.execute_action(action_data)

                # --- Check if confirmation is needed (backend returns dict) ---
                if isinstance(backend_response, dict) and backend_response.get("action") == "confirm_overwrite":
                    gui_logger.info(f"Overwrite confirmation needed for action: {backend_response}")
                    # Store the details needed to perform the action later
                    self.pending_confirmation = {
                        'action': 'create_file',  # The action to perform if confirmed
                        'args': backend_response.get('args', {})
                    }
                    # Emit confirmation signal instead of modification response
                    self.signals.confirmation_needed.emit("confirm_overwrite", backend_response.get("args", {}))
                    return  # Stop processing here, wait for GUI confirmation
                # --- Otherwise, assume (success, message, suffix) tuple ---
                elif isinstance(backend_response, tuple) and len(backend_response) == 3:
                    success, message, action_suffix = backend_response
                else:
                    # Handle unexpected response format
                    logger.error(f"Unexpected response format from execute_action: {backend_response}")
                    success = False
                    message = "Received unexpected response from backend during action execution."
                    action_suffix = f"{action}_error"

            except Exception as e:
                error_msg = f"Error calling backend execute_action for '{action}': {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
                success = False
                message = f"Internal error executing action: {e}"
                action_suffix = f"{action}_exception"

            # Add result message to conversation history
            result_timestamp = datetime.now(timezone.utc).isoformat()
            self.current_conversation.append({"speaker": "System", "text": message, "timestamp": result_timestamp})

            # Emit signal to GUI (use the suffix returned by execute_action)
            # Need placeholder for user input and target info for this signal
            user_input_placeholder = f"Action request: {action}"
            target_info_placeholder = str(action_data.get('args', {}))[:100]  # Use args as target info
            self.signals.modification_response_ready.emit(user_input_placeholder, message, action_suffix,
                                                          target_info_placeholder)
        else:
            error_msg = "Backend client not available for action execution."
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg)

    def handle_confirmed_action_task(self, confirmed_action_data):
        """Executes an action that the user has explicitly confirmed (e.g., overwrite)."""
        action = confirmed_action_data.get('action')
        args = confirmed_action_data.get('args', {})
        gui_logger.info(f"Executing CONFIRMED action: {action} with args: {args}")

        success = False
        message = f"Failed to execute confirmed action '{action}'."
        action_suffix = f"{action}_fail"  # Start with fail suffix

        try:
            if action == "create_file":
                filename, content = args.get("filename"), args.get("content")
                if filename and content is not None:
                    # Call file manager directly, bypassing the check in execute_action
                    success, message = file_manager.create_or_overwrite_file(
                        self.client.config, self.client.personality, filename, str(content)
                    )
                    action_suffix = f"{action}_{'success' if success else 'fail'}"
                else:
                    message = "Error: Missing filename or content for confirmed create_file."
                    success = False
            # Add other confirmed actions here if needed later
            else:
                message = f"Error: Unknown confirmed action type '{action}'."
                success = False
                action_suffix = "unknown_action_fail"

        except Exception as e:
            gui_logger.error(f"Exception during execution of CONFIRMED action '{action}': {e}", exc_info=True)
            message = f"An internal error occurred while trying to perform confirmed action '{action}'."
            success = False
            action_suffix = f"{action}_exception"

        # Add result message to conversation history
        result_timestamp = datetime.now(timezone.utc).isoformat()
        self.current_conversation.append({"speaker": "System", "text": message, "timestamp": result_timestamp})

        # Emit signal to GUI
        user_input_placeholder = f"Confirmed Action: {action}"
        target_info_placeholder = str(args)[:100]
        self.signals.modification_response_ready.emit(user_input_placeholder, message, action_suffix,
                                                      target_info_placeholder)

        # Clear pending state
        self.pending_confirmation = None

    def handle_saliency_update_task(self, data: dict):
        """Handles the task to update node saliency via the backend."""
        uuid = data.get('uuid')
        direction = data.get('direction')
        if not uuid or not direction:
            gui_logger.error(f"Invalid data for saliency update task: {data}")
            return

        gui_logger.info(f"Worker handling saliency update: UUID={uuid}, Direction={direction}")
        if self.client:
            try:
                # Call the backend method (no return value expected currently)
                self.client.update_node_saliency(uuid, direction)
                # Optionally emit a signal back to GUI? For now, just log.
                self.signals.log_message.emit(f"Saliency update processed for {uuid[:8]}.")
            except Exception as e:
                error_msg = f"Error during saliency update for {uuid}: {e}"
                self.signals.error.emit(error_msg)
                backend_logger.error(error_msg, exc_info=True)
        else:
            error_msg = "Backend client not available for saliency update."
            self.signals.error.emit(error_msg)
            backend_logger.error(error_msg)

    def add_input(self, text: str, attachment: dict | None = None):
        """Analyzes input/attachment and adds appropriate task to the queue."""
        if not self.client:
            self.signals.error.emit("Backend client not ready.")
            return

        gui_logger.info(
            f"Worker received input: Text='{text[:50]}...', Attachment Type='{attachment.get('type') if attachment else None}'")

        # --- Check for Pending Clarification ---
        if self.pending_clarification:
            gui_logger.info(
                f"Handling input as clarification response for action: {self.pending_clarification.get('original_action')}")
            # Assume the new text provides the missing arguments.
            # A more robust approach might try to parse the new text specifically
            # for the missing args, but let's start simple.
            # We'll merge the new text into a likely missing argument.
            # This is heuristic - assumes the user provides *only* the missing info.

            original_action = self.pending_clarification.get('original_action')
            current_args = self.pending_clarification.get('args', {})
            missing_args = self.pending_clarification.get('missing_args', [])

            # Simple strategy: Assume the text fills the *first* missing argument.
            # This might need refinement based on typical user responses.
            if missing_args:
                first_missing = missing_args[0]
                current_args[first_missing] = text  # Assign the user's input text
                gui_logger.debug(f"Attempting to fill missing arg '{first_missing}' with text: '{text[:50]}...'")

                # Construct the action data to execute
                action_data_to_execute = {
                    "action": original_action,
                    "args": current_args
                }

                # Clear pending state *before* queuing execution
                self.pending_clarification = None

                # Add user's clarification text to history
                self.current_conversation.append(
                    {"speaker": "User", "text": text, "timestamp": datetime.now(timezone.utc).isoformat()})
                # Queue the execution task
                gui_logger.info(f"Queuing action '{original_action}' for execution with clarified args: {current_args}")
                self.input_queue.append(('execute_action', action_data_to_execute))
                return  # Stop further processing of this input

            else:
                # Should not happen if pending_clarification is set correctly, but handle defensively.
                gui_logger.warning(
                    "Clarification was pending, but no missing args listed. Clearing state and proceeding normally.")
                self.pending_clarification = None
                # Fall through to normal processing...

        # --- If no clarification pending, proceed as normal ---

        # --- Prioritize Image Attachment ---
        if attachment and attachment.get('type') == 'image':
            gui_logger.info("Image attachment found. Queuing chat task directly.")
            task_data = {'text': text, 'attachment': attachment}
            self.input_queue.append(('chat', task_data))
            return

        # --- If no image, proceed with standard analysis ---
        gui_logger.info(f"Analyzing text input for actions/commands: '{text[:50]}...'")
        try:
            # Analyze for File/Calendar actions first
            analysis_result = self.client.analyze_action_request(text)
            action = analysis_result.get("action")

            if action == "error":
                self.signals.error.emit(f"Action Analysis Error: {analysis_result.get('reason', '?')}")
                # Maybe proceed with chat? Or just stop? For now, stop.
                return
            elif action == "clarify":
                # If clarification is needed, add user text to history and emit signal
                self.current_conversation.append(
                    {"speaker": "User", "text": text, "timestamp": datetime.now(timezone.utc).isoformat()})
                # --- Store pending clarification state ---
                self.pending_clarification = {
                    "original_action": analysis_result.get("original_action", "?"),
                    "args": analysis_result.get("args", {}),  # Store args already extracted by LLM
                    "missing_args": analysis_result.get("missing_args", [])
                }
                gui_logger.info(f"Stored pending clarification: {self.pending_clarification}")
                self.signals.clarification_needed.emit(
                    self.pending_clarification["original_action"],
                    self.pending_clarification["missing_args"]
                )
                # Don't queue chat task, wait for user clarification
                return
            elif action != "none":
                # If a specific file/calendar action is identified, queue it
                # Clear any potentially stale clarification state if a new, complete action is detected
                if self.pending_clarification:
                    gui_logger.debug("Clearing stale pending clarification due to new complete action.")
                    self.pending_clarification = None
                gui_logger.info(f"Action '{action}' detected. Queuing execution task.")
                # Add user text to history before queuing action
                self.current_conversation.append(
                    {"speaker": "User", "text": text, "timestamp": datetime.now(timezone.utc).isoformat()})
                self.input_queue.append(('execute_action', analysis_result))
                return  # Don't queue chat task

        except Exception as e:
            # Log error during analysis but potentially fall through to check modification/chat
            gui_logger.error(f"Error during action request analysis: {e}", exc_info=True)
            self.signals.error.emit(f"Failed to analyze for file/calendar actions: {e}")
            # Fall through to check memory modification (currently disabled)

        # --- If not a file/calendar action, analyze for memory modification ---
        # --- DISABLED Memory Modification Analysis ---
        # try:
        #     mod_analysis_result = self.client.analyze_memory_modification_request(text)
        #     mod_action = mod_analysis_result.get("action")
        #
        #     if mod_action == "error":
        #         self.signals.error.emit(f"Memory Modification Analysis Error: {mod_analysis_result.get('reason', '?')}")
        #         # Proceed with chat? Or stop? Let's proceed with chat for now.
        #         mod_action = "none" # Treat error as no mod action
        #
        #     if mod_action != "none":
        #         # If a specific memory modification action is identified, queue it
        #         # Clear any potentially stale clarification state
        #         if self.pending_clarification:
        #             gui_logger.debug("Clearing stale pending clarification due to modification request.")
        #             self.pending_clarification = None
        #         gui_logger.info(f"Memory modification action '{mod_action}' detected by LLM. Queuing modify task.")
        #         # Don't add user input to history here, let handle_modify_task decide
        #         # Pass the *original text* to the modify handler, as it expects the raw command
        #         self.input_queue.append(('modify', text))
        #         return # Don't queue chat task
        #
        # except Exception as e:
        #      # Log error during analysis but fall through to chat
        #      gui_logger.error(f"Error during memory modification analysis: {e}", exc_info=True)
        #      self.signals.error.emit(f"Failed to analyze for memory modification actions: {e}")
        # --- END DISABLED Memory Modification Analysis ---

        # --- If not action or modification (both analyses returned "none" or errored), treat as regular chat ---
        gui_logger.info(
            "No specific file/calendar or memory modification action detected by analysis. Queuing chat task.")
        # Clear any potentially stale clarification state
        if self.pending_clarification:
            gui_logger.debug("Clearing stale pending clarification due to regular chat input.")
            self.pending_clarification = None
        task_data = {'text': text, 'attachment': None}  # Ensure chat task always gets dict
        self.input_queue.append(('chat', task_data))

    def request_memory_reset(self):
        # (Implementation remains the same)
        gui_logger.info("GUI requested memory reset.");
        self.input_queue.append(('reset', None))

    def request_consolidation(self):
        # (Implementation remains the same)
        gui_logger.info("GUI requested manual consolidation.");
        self.input_queue.append(('consolidate', None))

    def request_saliency_update(self, uuid: str, direction: str):
        """Adds a saliency update task to the queue."""
        gui_logger.info(f"GUI requested saliency update for {uuid} ({direction}).")
        self.input_queue.append(('saliency_update', {'uuid': uuid, 'direction': direction}))

    def stop(self):
        # (Implementation remains the same)
        self.is_running = False;
        gui_logger.info("Stop requested for worker.")

    def save_memory_on_stop(self):
        # (Implementation remains the same)
        if self.client:
            backend_logger.info("Triggering final save from worker...")
            try:
                self.client._save_memory(); backend_logger.info("Final save complete.")
            except Exception as e:
                backend_logger.error(f"Error during final save: {e}", exc_info=True)
        else:
            backend_logger.warning("Worker stopping, but no backend client to save.")


# --- Collapsible Memory Widget ---
# (Implementation remains the same as previous version)
class CollapsibleMemoryWidget(QWidget):
    # NEW Signal
    saliency_feedback_requested = pyqtSignal(str, str)  # uuid, direction ('increase'/'decrease')

    def __init__(self, memories, parent=None):
        super().__init__(parent)
        self.memories = memories or []
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("MemoryToggle")
        self.content_area = QTextBrowser()  # Changed from QTextEdit
        self.content_area.setReadOnly(True)
        self.content_area.setVisible(False)
        self.content_area.setObjectName("MemoryContent")
        # Enable link clicking (anchorClicked signal handles this)
        self.content_area.anchorClicked.connect(self.handle_link_click)  # Connect signal

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.toggled.connect(self.toggle_content)
        self.update_button_text()
        self.populate_content()

    def update_button_text(self):
        prefix = "[-] Hide" if self.toggle_button.isChecked() else "[+] Show"; count = len(
            self.memories); self.toggle_button.setText(f"{prefix} Retrieved Memories ({count})")

    def toggle_content(self, checked):
        self.content_area.setVisible(checked); self.update_button_text(); self.adjustSize(); QTimer.singleShot(0,
                                                                                                               self._update_parent_layout)

    def _update_parent_layout(self):
        if self.parentWidget() and self.parentWidget().layout(): self.parentWidget().layout().activate()

    def populate_content(self):
        if not self.memories: self.content_area.setHtml("<small><i>No relevant memories retrieved.</i></small>"); return
        # Sort memories by final_activation score (descending), then timestamp as secondary key
        html_content = "";
        sorted_memories = sorted(self.memories, key=lambda x: (x.get('final_activation', 0.0), x.get('timestamp', '')),
                                 reverse=True)
        for mem in sorted_memories:
            ts_str = mem.get('timestamp', 'N/A');
            node_type = mem.get('node_type', '?type');
            act_score = mem.get('final_activation', -1.0);
            speaker = str(mem.get('speaker', '?')).replace('<', '&lt;').replace('>', '&gt;');
            text = str(mem.get('text', '')).replace('<', '&lt;').replace('>', '&gt;');
            text_multiline = text.replace('\n', '<br/>');
            uuid_str = mem.get('uuid', '')[:8];
            score_str = f"(Act: {act_score:.3f})" if act_score >= 0 else "";
            time_desc = ts_str[:10]
            try:
                dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00')); time_diff = datetime.now(
                    timezone.utc) - dt_obj;
            except (ValueError, TypeError):
                dt_obj = None; time_diff = None
            if time_diff:
                if time_diff < timedelta(minutes=1):
                    time_desc = f"{int(time_diff.total_seconds())}s ago"
                elif time_diff < timedelta(hours=1):
                    time_desc = f"{int(time_diff.total_seconds() / 60)}m ago"
                elif time_diff < timedelta(days=1):
                    time_desc = f"{int(time_diff.total_seconds() / 3600)}h ago"
                elif dt_obj:
                    time_desc = dt_obj.strftime('%y-%m-%d')

            # --- Add Saliency Links ---
            saliency_links = ""
            if uuid_str:  # Only add links if we have a full UUID
                increase_link = f'<a href="saliency://increase/{mem.get("uuid", "")}" style="color: #8FBC8F; text-decoration: none;">[+S]</a>'  # Use full UUID
                decrease_link = f'<a href="saliency://decrease/{mem.get("uuid", "")}" style="color: #F08080; text-decoration: none;">[-S]</a>'  # Use full UUID
                saliency_links = f"&nbsp;{increase_link}&nbsp;{decrease_link}"

            html_content += (
                f"<div style='margin-bottom: 5px; border-left: 2px solid #555; padding-left: 5px;'>"
                f"<small><i><b>{speaker}</b> [{node_type}] ({time_desc}) {score_str} [{uuid_str}]{saliency_links}</i></small>"  # Add links here
                f"<br/>{text_multiline}</div>"
            )
        self.content_area.setHtml(html_content)

    # --- NEW Link Handler ---
    @pyqtSlot(QUrl)
    def handle_link_click(self, url: QUrl):
        """Handles clicks on custom saliency links."""
        if url.scheme() == "saliency":
            action = url.host()  # 'increase' or 'decrease'
            uuid = url.path().strip('/')  # Get the UUID part
            if action in ['increase', 'decrease'] and uuid:
                gui_logger.info(f"Saliency feedback link clicked: Action={action}, UUID={uuid}")
                self.saliency_feedback_requested.emit(uuid, action)
            else:
                gui_logger.warning(f"Invalid saliency link format: {url.toString()}")
        else:
            # Could potentially handle other schemes or pass to default handler here
            gui_logger.debug(f"Ignoring non-saliency link click: {url.toString()}")


# (Before class ChatWindow(QMainWindow): in gui_chat.py)

# (Before class ChatWindow(QMainWindow): in gui_chat.py)

class PasteLineEdit(QLineEdit):
    """A QLineEdit subclass that accepts pasted/dropped images/files via explicit reference."""

    # Add chat_window_ref argument to __init__
    def __init__(self, chat_window_ref, parent=None):
        super().__init__(parent)
        # Store the reference to the main ChatWindow instance
        self.chat_window = chat_window_ref
        gui_logger.info("PasteLineEdit Initialized")
        self.setAcceptDrops(True)

    # --- Drag/Drop Handling ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls() or mime_data.hasImage():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls() or mime_data.hasImage():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        mime_data = event.mimeData();
        gui_logger.debug("Drop event detected.")
        if mime_data.hasUrls():
            urls = mime_data.urls();
            file_path = urls[0].toLocalFile() if urls else None;
            gui_logger.debug(f"Dropped URL path: {file_path}")
            if file_path and os.path.isfile(file_path):
                # Use the stored chat_window reference to handle the file path
                if hasattr(self.chat_window, 'handle_attach_file_path'):
                    gui_logger.debug("Calling chat_window.handle_attach_file_path for drop...")
                    self.chat_window.handle_attach_file_path(file_path)
                    event.acceptProposedAction()
                    return
                else:
                    gui_logger.error("Drop Event: ChatWindow reference missing handle_attach_file_path method!")
            else:
                gui_logger.debug("Dropped URL path is not a valid file.")
        elif mime_data.hasImage():
            # Handle dropped image data directly (similar to paste)
            if self.handle_pasted_image_data(mime_data):
                event.acceptProposedAction()
                return
            else:
                gui_logger.warning("Dropped image data could not be handled.")
        else:
            gui_logger.debug("Drop does not contain URLs or Image data.")

        super().dropEvent(event)  # Pass to default if not handled

    # --- Handle Paste Directly in Key Press ---
    def keyPressEvent(self, event: QKeyEvent):
        """Intercepts key presses, handles Ctrl+V for images/files directly."""
        paste_handled = False
        if event.matches(QKeySequence.StandardKey.Paste):
            clipboard = QApplication.clipboard()
            mime_data = clipboard.mimeData()
            available_formats = mime_data.formats() if mime_data else []
            gui_logger.debug(f"--- Ctrl+V Detected! MIME formats: {available_formats} ---")
            # (Keep internal logging...)

            # 1. Handle Image Data
            if mime_data.hasImage():
                paste_handled = self.handle_pasted_image_data(mime_data)
            # 2. Handle File Path URLs
            elif mime_data.hasUrls() and not paste_handled:
                paste_handled = self.handle_pasted_urls(mime_data)

            if paste_handled:
                gui_logger.debug("Paste event handled by custom logic.")
                event.accept();
                return  # Consume event

        # If not handled, pass to default
        # gui_logger.debug("Passing key event to default handler.")
        super().keyPressEvent(event)

    # --- Helper Methods for Paste Handling ---
    def handle_pasted_image_data(self, mime_data: QMimeData) -> bool:
        """Processes raw image data from clipboard."""
        gui_logger.debug("Attempting to handle pasted image data...")
        # Use the stored chat_window reference
        if not hasattr(self.chat_window, 'handle_attach_payload'):
            gui_logger.error("Paste Error: ChatWindow reference missing handle_attach_payload method!")
            return False

        image_data = mime_data.imageData()
        if isinstance(image_data, QImage) and not image_data.isNull():
            byte_array = QByteArray();
            buffer = QBuffer(byte_array)
            if buffer.open(QIODevice.OpenModeFlag.WriteOnly):
                saved_ok = image_data.save(buffer, "PNG");
                buffer.close()
                if saved_ok:
                    try:
                        image_bytes = byte_array.data();
                        base64_encoded_data = base64.b64encode(image_bytes);
                        base64_string = base64_encoded_data.decode('utf-8');
                        image_data_url = f"data:image/png;base64,{base64_string}"
                        payload = {"type": "image", "filename": "pasted_image.png", "data_url": image_data_url,
                                   "base64_string": base64_string}
                        gui_logger.debug("Calling chat_window.handle_attach_payload (from pasted image data).")
                        self.chat_window.handle_attach_payload(payload)  # Call using reference
                        return True  # Success
                    except Exception as e:
                        gui_logger.error(f"Error encoding/handling pasted image data: {e}", exc_info=True)
                else:
                    gui_logger.error("Failed to save QImage to buffer.")
            else:
                gui_logger.error("Failed to open QBuffer for writing image data.")
        else:
            gui_logger.warning("mimeData had image, but imageData() was null/invalid.")
        return False  # Failed

    def handle_pasted_urls(self, mime_data: QMimeData) -> bool:
        """Processes URL list from clipboard, looking for image file paths."""
        gui_logger.debug("Attempting to handle pasted URLs...")
        # Use the stored chat_window reference
        if not hasattr(self.chat_window, 'handle_attach_file_path'):
            gui_logger.error("Paste Error: ChatWindow reference missing handle_attach_file_path method!")
            return False

        urls = mime_data.urls()
        if urls:
            file_path = urls[0].toLocalFile()
            gui_logger.debug(f"Checking URL path: {file_path}")
            if file_path and os.path.isfile(file_path):
                # Let the main window handler decide if it's an image or generic file
                gui_logger.debug("Calling chat_window.handle_attach_file_path (from pasted URL).")
                self.chat_window.handle_attach_file_path(file_path)  # Call using reference
                return True  # Success # Corrected indentation
            else:
                gui_logger.debug("Pasted URL path is not a valid file.")
        else:
            gui_logger.debug("mimeData hasUrls() true, but URL list empty?")
        return False  # Failed


# --- Main Chat Window ---
class ChatWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Persistent Memory Chat")  # Initial title
        self.setGeometry(100, 100, 850, 750)

        self.config_path = DEFAULT_CONFIG_PATH
        self._load_style_config()  # Load style config first
        # self.current_personality = self._load_default_personality() # Don't load default
        self.current_personality = None  # Start with no personality selected

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self._create_menu_bar()
        self._setup_ui_elements()  # Setup UI elements first
        self._setup_status_bar()  # Setup status bar including indicator
        self.apply_dark_theme()  # Apply theme AFTER UI elements exist

        self.worker = None  # Worker thread instance
        self.attached_file_path = None  # Store path of file to attach
        self.is_processing = False  # Flag to prevent concurrent processing
        self.awaiting_clarification = False  # Flag for clarification state
        self.pending_confirmation = None  # NEW: Store details for pending confirmation {'action': str, 'args': dict}
        self.user_scrolled_up = False # Flag to track if user manually scrolled up

        # --- Initial State ---
        self.set_input_enabled(False)  # Start with input disabled
        self.update_status_light(False)  # Start with red light
        self.statusBar().showMessage("Please select a personality from the menu to begin.")
        # Add initial message to chat display
        QTimer.singleShot(100, lambda: self.display_message("System",
                                                            "Welcome! Please select a personality from the 'Personality' menu to load the AI."))

    def _load_style_config(self):
        """Loads GUI style parameters from the config file."""
        gui_logger.info(f"Loading GUI style config from: {self.config_path}")
        # Set defaults (including new thumbnail width)
        self.bubble_max_width = "75%";
        self.bubble_side_margin = 50;
        self.bubble_edge_margin = 5;
        self.bubble_min_width = 100;
        self.bubble_border_radius = 18;
        self.bubble_padding_tb = 6;
        self.bubble_padding_lr = 10;
        self.bubble_ts_color_user = "#E0E0E0";
        self.bubble_ts_color_ai = "#B0B0B0";
        self.input_border_radius = 15;
        self.input_padding = 8;
        self.button_border_radius = 15;
        self.button_padding_v = 8;
        self.button_padding_h = 16
        self.thumbnail_max_width = 200  # Default thumbnail width

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config and 'gui_style' in config:
                style_config = config['gui_style'];
                bubbles = style_config.get('bubbles', {});
                input_field = style_config.get('input_field', {});
                buttons = style_config.get('buttons', {})
                self.bubble_max_width = f"{bubbles.get('max_width_percent', 75)}%";
                self.bubble_side_margin = bubbles.get('side_margin_px', 50);
                self.bubble_edge_margin = bubbles.get('edge_margin_px', 5);
                self.bubble_min_width = bubbles.get('min_width_px', 100);
                self.bubble_border_radius = bubbles.get('border_radius_px', 18);
                self.bubble_padding_tb = bubbles.get('internal_padding_top_bottom_px', 6);
                self.bubble_padding_lr = bubbles.get('internal_padding_left_right_px', 10);
                self.bubble_ts_color_user = bubbles.get('timestamp_color_user', "#E0E0E0");
                self.bubble_ts_color_ai = bubbles.get('timestamp_color_ai', "#B0B0B0")
                self.thumbnail_max_width = bubbles.get('thumbnail_max_width_px', 200)  # Load thumbnail width

                self.input_border_radius = input_field.get('border_radius_px', 15);
                self.input_padding = input_field.get('padding_px', 8)
                self.button_border_radius = buttons.get('border_radius_px', 15);
                self.button_padding_v = buttons.get('padding_vertical_px', 8);
                self.button_padding_h = buttons.get('padding_horizontal_px', 16)
                gui_logger.info("GUI style config loaded successfully.")
            else:
                gui_logger.warning("GUI style section not found in config. Using defaults.")
        except FileNotFoundError:
            gui_logger.error(f"Config file not found at {self.config_path}. Using default styles.")
        except (yaml.YAMLError, KeyError, TypeError) as e:
            gui_logger.error(f"Error loading GUI style config: {e}. Using defaults.")

    def _load_default_personality(self):
        # (Implementation remains the same - not called in __init__ anymore)
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f); return config.get('default_personality')  # Return None if not set
        except Exception as e:
            gui_logger.error(f"Could not load default personality setting: {e}."); return None

    def _create_menu_bar(self):
        # (Implementation remains the same)
        self.personality_menu = self.menuBar().addMenu("&Personality")
        self._rebuild_personality_menu()

    def _rebuild_personality_menu(self):
        # (Implementation remains the same - marks current personality)
        self.personality_menu.clear();
        switch_action_group = QActionGroup(self);
        switch_action_group.setExclusive(True)
        available = get_available_personalities(self.config_path)
        found_current = False
        for name in available:
            action = QAction(name, self, checkable=True)
            action.triggered.connect(lambda checked, p=name: self.switch_personality(p))
            if name == self.current_personality: action.setChecked(True); found_current = True
            self.personality_menu.addAction(action);
            switch_action_group.addAction(action)
        # If current_personality is None or not found (e.g., deleted), uncheck all
        if not found_current:
            if switch_action_group.checkedAction():
                switch_action_group.checkedAction().setChecked(False)
        self.personality_menu.addSeparator();
        create_action = QAction("&New Personality...", self);
        create_action.triggered.connect(self.create_new_personality);
        self.personality_menu.addAction(create_action)

    def _setup_ui_elements(self):
        # (Implementation mostly same, uses PasteLineEdit)
        self.scroll_area = QScrollArea();
        self.scroll_area.setObjectName("ScrollArea");
        self.scroll_area.setWidgetResizable(True);
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_widget = QWidget();
        self.scroll_widget.setObjectName("ScrollWidget");
        self.chat_layout = QVBoxLayout(self.scroll_widget);
        self.chat_layout.setContentsMargins(10, 10, 10, 10);
        self.chat_layout.setSpacing(8);
        self.chat_layout.addStretch();
        self.scroll_area.setWidget(self.scroll_widget)
        self.input_frame = QFrame();
        self.input_frame.setObjectName("InputFrame");
        self.input_layout = QHBoxLayout(self.input_frame);
        self.input_layout.setContentsMargins(5, 5, 5, 5);
        self.input_layout.setSpacing(5)
        self.input_field = PasteLineEdit(self, self.input_frame)  # Pass self as chat_window_ref, input_frame as parent
        self.input_field.setPlaceholderText("Select a personality from the menu to start...")
        self.input_field.returnPressed.connect(self.send_message)
        self.attach_button = QPushButton("+");
        self.attach_button.setObjectName("AttachButton");
        self.attach_button.setToolTip("Attach Image File");
        self.attach_button.clicked.connect(self.handle_attach_file)
        self.send_button = QPushButton("Send");
        self.send_button.setObjectName("SendButton");
        self.send_button.clicked.connect(self.send_message)
        self.consolidate_button = QPushButton("Consolidate");
        self.consolidate_button.setObjectName("ConsolidateButton");
        self.consolidate_button.clicked.connect(self.request_consolidation);
        self.consolidate_button.setToolTip("Run consolidation for loaded personality")
        self.reset_button = QPushButton("Reset Mem");
        self.reset_button.setObjectName("ResetButton");
        self.reset_button.clicked.connect(self.confirm_reset_memory);
        self.reset_button.setToolTip("Reset memory for loaded personality")
        self.input_layout.addWidget(self.input_field, 1);
        self.input_layout.addWidget(self.attach_button);
        self.input_layout.addWidget(self.send_button);
        self.input_layout.addWidget(self.consolidate_button);
        self.input_layout.addWidget(self.reset_button)
        self.layout.addWidget(self.scroll_area, 1);
        self.layout.addWidget(self.input_frame)

        # --- Connect scrollbar signals ---
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.rangeChanged.connect(self._scroll_to_bottom_on_range_change)
        scrollbar.valueChanged.connect(self._handle_scroll_value_changed) # Connect valueChanged

    def _setup_status_bar(self):
        """Creates the status bar and adds the status light widget."""
        # Status Light Indicator (Label with a circle character)
        self.status_light = QLabel("●")  # Use a unicode circle
        self.status_light.setObjectName("StatusLight")  # For styling
        # Add as a permanent widget to the right side
        self.statusBar().addPermanentWidget(self.status_light)
        # Initial message set in __init__

    def apply_dark_theme(self):
        """Applies a dark theme stylesheet using values from config."""
        # Use f-string for stylesheet, double {{ }} for literal CSS braces
        self.setStyleSheet(f"""
            /* ... (Keep existing styles for QMainWindow, QLineEdit, QPushButton, etc.) ... */
            QMainWindow, QWidget {{ background-color: #1E1E1E; color: #E0E0E0; }}
            QTextEdit, QLineEdit {{ background-color: #2D2D2D; color: #E0E0E0; border: 1px solid #444; border-radius: {self.input_border_radius}px; padding: {self.input_padding}px; font-size: 10pt; }}
            QLineEdit {{ background-color: #2D2D2D; }}
            QPushButton {{ background-color: #007AFF; color: white; border: none; padding: {self.button_padding_v}px {self.button_padding_h}px; border-radius: {self.button_border_radius}px; font-size: 10pt; font-weight: bold; }}
            QPushButton:hover {{ background-color: #005ECB; }}
            QPushButton:disabled {{ background-color: #333; color: #666; }}

            #AttachButton {{ background-color: #2D2D2D; color: white; font-weight: bold; font-size: 14pt; min-width: {max(20, self.input_field.fontMetrics().height() + 4)}px; max-width: {max(20, self.input_field.fontMetrics().height() + 4)}px; padding: 1px; border-radius: {self.button_border_radius // 2}px; padding-top: 1px; padding-bottom: 1px; padding-left: 1px; padding-right: 1px; }}
            #AttachButton:hover {{ background-color: #444; }}
            #AttachButton:disabled {{ background-color: #222; color: #555; }}

            QScrollArea {{ border: none; background-color: #1E1E1E; }}
            #ScrollWidget {{ background-color: #1E1E1E; }}
            QScrollBar:vertical {{ background: #2D2D2D; width: 10px; margin: 0px; border-radius: 5px; }}
            QScrollBar::handle:vertical {{ background: #555; min-height: 20px; border-radius: 5px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ background: none; height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

            /* Bubble Frames */
            #UserBubbleFrame {{ background-color: #007AFF; color: white; border-radius: {self.bubble_border_radius}px; margin-left: {self.bubble_side_margin}px; margin-right: {self.bubble_edge_margin}px; margin-top: 4px; margin-bottom: 4px; min-width: {self.bubble_min_width}px; max-width: {self.bubble_max_width}; padding: {self.bubble_padding_tb}px {self.bubble_padding_lr}px; }}
            #AIBubbleFrame, #SystemConfirmationBubbleFrame, #AIConfirmationBubbleFrame, #ErrorBubbleFrame {{ background-color: #3A3A3C; color: #E0E0E0; border-radius: {self.bubble_border_radius}px; margin-right: {self.bubble_side_margin}px; margin-left: {self.bubble_edge_margin}px; margin-top: 4px; margin-bottom: 4px; min-width: {self.bubble_min_width}px; max-width: {self.bubble_max_width}; padding: {self.bubble_padding_tb}px {self.bubble_padding_lr}px; }}
            #ErrorBubbleFrame {{ background-color: #7D1E1E; color: #FFCCCC; }}

            /* Labels inside Bubbles */
            #UserBubbleFrame QLabel, #AIBubbleFrame QLabel, #SystemConfirmationBubbleFrame QLabel, #AIConfirmationBubbleFrame QLabel, #ErrorBubbleFrame QLabel {{ background-color: transparent; border: none; padding: 0px; }}
            #UserBubbleFrame QLabel {{ color: white; }}
            #AIBubbleFrame QLabel, #SystemConfirmationBubbleFrame QLabel, #AIConfirmationBubbleFrame QLabel {{ color: #E0E0E0; }}
            #ErrorBubbleFrame QLabel {{ color: #FFCCCC; }}

            /* Timestamp Labels */
            #UserTimestampLabel {{ color: {self.bubble_ts_color_user}; font-size: 8pt; background-color: transparent; padding-top: 3px; }}
            #AITimestampLabel, #SystemTimestampLabel, #ErrorTimestampLabel {{ color: {self.bubble_ts_color_ai}; font-size: 8pt; background-color: transparent; padding-top: 3px; }}
            #ErrorBubbleFrame #ErrorTimestampLabel {{ color: #FFCCCC; }}

            /* Collapsible Memory */
            #MemoryToggle {{ background-color: transparent; color: #AAAAAA; border: none; text-align: left; padding: 2px 5px; font-size: 9pt; font-weight: normal; margin: 0px {self.bubble_edge_margin + 5}px; }}
            #MemoryToggle:hover {{ color: #E0E0E0; }}
            #MemoryContent {{ background-color: #252526; color: #CCCCCC; border: 1px solid #444; border-radius: 8px; padding: 5px; margin: 0px {self.bubble_edge_margin + 5}px 5px {self.bubble_edge_margin + 5}px; font-size: 9pt; }}

            /* Action Indicators */
            #MemoryActionIndicator, #MemoryDeleteIndicator, #MemoryEditIndicator, #MemoryForgetIndicator, #MemoryNeutralIndicator {{ color: #AAAAAA; font-size: 8pt; font-style: italic; padding-left: {self.bubble_edge_margin + 10}px; padding-top: 0px; padding-bottom: 5px; border: none; background-color: transparent; }}
            #MemoryDeleteIndicator {{ color: #FF8C8C; }}
            #MemoryEditIndicator {{ color: #AAAAAA; }}
            #MemoryNeutralIndicator {{ color: #90CAF9; }}

            /* --- NEW: Task Bubble Styles --- */
            #TaskBubbleFrame {{
                background-color: #2C2C2E; /* Slightly different background */
                color: #B0B0B0; /* Lighter grey text */
                border: 1px solid #444;
                border-radius: 8px; /* Less rounded than chat bubbles */
                margin-left: {self.bubble_edge_margin + 10}px; /* Indent slightly */
                margin-right: {self.bubble_edge_margin + 10}px;
                margin-top: 6px;
                margin-bottom: 6px;
                padding: 8px 12px; /* More padding */
            }}
            #TaskBubbleFrame QLabel {{ /* General labels inside */
                background-color: transparent;
                border: none;
                padding: 0px;
                color: #B0B0B0;
            }}
            #TaskLabel {{ /* The main message label */
                font-style: italic;
            }}
            #TaskFilenameLink a {{ /* Style for the filename link */
                color: #77AADD; /* Link color */
                text-decoration: none;
                font-weight: bold;
            }}
            #TaskFilenameLink a:hover {{
                text-decoration: underline;
            }}
            /* --- End Task Bubble Styles --- */

            /* Input Frame */
            #InputFrame {{ background-color: #1E1E1E; border-top: 1px solid #333; padding: 5px; }}
            /* Menu Bar */
            QMenuBar {{ background-color: #2D2D2D; color: #E0E0E0; }}
            QMenuBar::item {{ background-color: #2D2D2D; color: #E0E0E0; padding: 4px 10px; }}
            QMenuBar::item:selected {{ background-color: #007AFF; color: white; }}
            QMenu {{ background-color: #2D2D2D; color: #E0E0E0; border: 1px solid #444; }}
            QMenu::item:selected {{ background-color: #007AFF; color: white; }}
            /* Status Bar */
            QStatusBar {{ color: #AAAAAA; padding-left: 5px; }} /* Add padding */

            /* --- Status Light Styles --- */
            #StatusLight {{
                font-size: 14pt; /* Adjust size of circle */
                font-weight: bold;
                min-width: 20px; /* Ensure space for the circle */
                text-align: center;
                padding-right: 5px; /* Padding on the right */
            }}
            #StatusLight[status="ready"] {{
                color: #4CAF50; /* Green */
            }}
            #StatusLight[status="not_ready"] {{
                color: #F44336; /* Red */
            }}
            #StatusLight[status="loading"] {{
                color: #FFC107; /* Yellow/Orange for loading */
            }}
           #StatusLight[status="processing"] {{
               color: #FFA500; /* Orange for processing */
           }}

       """)
        # Re-apply object names
        if hasattr(self, 'status_light'): self.status_light.setObjectName("StatusLight")
        # ... (re-apply other object names) ...
        if hasattr(self, 'attach_button'): self.attach_button.setObjectName("AttachButton")
        if hasattr(self, 'send_button'): self.send_button.setObjectName("SendButton")
        if hasattr(self, 'reset_button'): self.reset_button.setObjectName("ResetButton")
        if hasattr(self, 'consolidate_button'): self.consolidate_button.setObjectName("ConsolidateButton")
        if hasattr(self, 'scroll_area'): self.scroll_area.setObjectName("ScrollArea")
        if hasattr(self, 'scroll_widget'): self.scroll_widget.setObjectName("ScrollWidget")
        if hasattr(self, 'input_frame'): self.input_frame.setObjectName("InputFrame")

    def set_input_enabled(self, enabled: bool):
        """Enable or disable input field and buttons."""
        if hasattr(self, 'input_field'):
            self.input_field.setEnabled(enabled)
            # Update placeholder text based on state
            if enabled:
                self.input_field.setPlaceholderText("Type message, paste image, or /command...")
            else:
                self.input_field.setPlaceholderText("Select a personality from the menu to start...")

        if hasattr(self, 'send_button'): self.send_button.setEnabled(enabled)
        if hasattr(self, 'attach_button'): self.attach_button.setEnabled(enabled)
        # Only enable Reset/Consolidate if a model is loaded (enabled=True)
        if hasattr(self, 'reset_button'): self.reset_button.setEnabled(enabled)
        if hasattr(self, 'consolidate_button'): self.consolidate_button.setEnabled(enabled)

        if enabled and hasattr(self, 'input_field'):
            self.input_field.setFocus()

    def update_status_light(self, status: str):
        """Updates the status light color via dynamic property."""
        # status should be "ready", "not_ready", "loading", or "processing"
        if hasattr(self, 'status_light'):
            valid_statuses = ["ready", "not_ready", "loading", "processing"]
            if status not in valid_statuses:
                gui_logger.warning(
                    f"Invalid status '{status}' passed to update_status_light. Defaulting to 'not_ready'.")
                status = "not_ready"
            self.status_light.setProperty("status", status)
            # Re-polish to apply the style based on the new property
            self.style().unpolish(self.status_light)
            self.style().polish(self.status_light)

    def switch_personality(self, name):
        """Handles switching to a different personality."""
        if name == self.current_personality and self.worker and self.worker.isRunning():
            gui_logger.info(f"Already using: {name}")
            return
        # Allow switching even if processing, but warn/handle carefully?
        # For now, prevent switching if busy to avoid complex state issues.
        if self.is_processing:
            QMessageBox.warning(self, "Busy", "Cannot switch personality while processing.")
            self._rebuild_personality_menu()  # Fix menu check state
            return

        gui_logger.info(f"Switching personality to: {name}")
        self.statusBar().showMessage(f"Loading '{name}'...")
        self.set_input_enabled(False)  # Disable input during switch
        self.update_status_light("loading")  # Yellow light while loading
        self.clear_chat_display()
        self.display_message("System", f"Loading personality: {name}...")
        self.clear_attachment()  # Clear any pending attachments
        self.user_scrolled_up = False # Reset scroll flag on personality switch

        # Stop existing worker gracefully
        if self.worker and self.worker.isRunning():
            gui_logger.info("Stopping previous worker...")
            self.worker.stop()
            if not self.worker.wait(5000):
                gui_logger.warning("Previous worker thread did not stop gracefully.")
            else:
                gui_logger.info("Previous worker stopped.")

        # Disconnect signals from the old worker instance
        if self.worker:
            # Simplified disconnection - disconnect all slots from this object
            try:
                self.worker.signals.backend_ready.disconnect(self.on_backend_ready)
            except TypeError:
                pass
            try:
                self.worker.signals.response_ready.disconnect(self.display_response)
            except TypeError:
                pass
            try:
                self.worker.signals.modification_response_ready.disconnect(self.display_modification_confirmation)
            except TypeError:
                pass
            try:
                self.worker.signals.memory_reset_complete.disconnect(self.on_memory_reset_complete)
            except TypeError:
                pass
            try:
                self.worker.signals.consolidation_complete.disconnect(self.on_consolidation_complete)
            except TypeError:
                pass
            try:
                self.worker.signals.clarification_needed.disconnect(self.handle_clarification_request)
            except TypeError:
                pass
            try:
                self.worker.signals.error.disconnect(self.display_error)
            except TypeError:
                pass
            try:
                self.worker.signals.log_message.disconnect()
            except TypeError:
                pass
            self.worker = None  # Clear reference

        # Update personality state
        self.current_personality = name
        self.setWindowTitle(f"Persistent Memory Chat [{name}]")
        self._rebuild_personality_menu()  # Update menu check marks

        # Start new worker (which will signal back when ready/failed)
        self.start_worker_for_personality(name)

    def create_new_personality(self):
        # (Implementation remains the same as previous version)
        if self.is_processing: QMessageBox.warning(self, "Busy", "Cannot create while processing."); return
        name, ok = QInputDialog.getText(self, "New Personality", "Enter name:");
        if ok and name:
            sanitized_name = re.sub(r'[^\w\-]+', '_', name).strip('_.- ');
            if not sanitized_name: QMessageBox.warning(self, "Invalid Name", "Invalid name after sanitation."); return
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                base_path = config.get('base_memory_path', 'memory_sets');
                new_path = os.path.join(base_path, sanitized_name)
                if os.path.exists(new_path): QMessageBox.warning(self, "Exists",
                                                                 f"Personality '{sanitized_name}' already exists."); return
                os.makedirs(new_path, exist_ok=True);
                gui_logger.info(f"Created dir: {new_path}")
                self._rebuild_personality_menu()
                for action in self.personality_menu.actions():
                    if action.text() == sanitized_name: action.trigger(); break
            except Exception as e:
                gui_logger.error(f"Error creating personality '{sanitized_name}': {e}",
                                 exc_info=True); QMessageBox.critical(self, "Error", f"Could not create: {e}")

    def start_worker_for_personality(self, name):
        """Starts a new worker thread for the given personality name."""
        # Note: Signal disconnection should happen in switch_personality before calling this
        gui_logger.info(f"Starting worker for: {name}")
        self.worker = Worker(personality_name=name, config_path=self.config_path)

        # --- Connect signals ---
        # Connect the crucial backend_ready signal
        self.worker.signals.backend_ready.connect(self.on_backend_ready)
        # Connect other signals
        self.worker.signals.response_ready.connect(self.display_response)
        self.worker.signals.modification_response_ready.connect(self.display_modification_confirmation)
        self.worker.signals.memory_reset_complete.connect(self.on_memory_reset_complete)
        self.worker.signals.consolidation_complete.connect(self.on_consolidation_complete)
        self.worker.signals.clarification_needed.connect(self.handle_clarification_request)
        self.worker.signals.confirmation_needed.connect(self.handle_confirmation_request)  # NEW Connection
        self.worker.signals.error.connect(self.display_error)
        self.worker.signals.log_message.connect(lambda msg: self.statusBar().showMessage(msg, 4000))
        self.worker.signals.initial_history_ready.connect(self.display_initial_history) # NEW Connection
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.start()  # Start the thread execution (calls run())
        # DO NOT enable input here - wait for backend_ready signal

    @pyqtSlot(bool, str)  # Slot for backend_ready signal
    def on_backend_ready(self, success: bool, personality_name: str):
        """Handles the signal indicating backend initialization status."""
        gui_logger.info(
            f"Received backend_ready signal: Success={success}, Personality='{personality_name}' (Current: '{self.current_personality}')")

        # Only update UI if the signal is for the currently selected personality
        if personality_name == self.current_personality:
            if success:
                gui_logger.info(f"Backend ready for '{personality_name}'. Enabling input.")
                self.update_status_light("ready")  # Green light
                self.set_input_enabled(True)  # Enable input fields and buttons
                self.is_processing = False  # No longer processing the load
                self.statusBar().showMessage(f"'{personality_name}' loaded. Ready.", 5000)
            else:
                gui_logger.error(f"Backend failed to initialize for '{personality_name}'.")
                self.update_status_light("not_ready")  # Red light
                self.set_input_enabled(False)  # Keep input disabled
                self.is_processing = False  # Loading failed
                self.statusBar().showMessage(f"Error loading '{personality_name}'. Select another personality.",
                                             0)  # Persistent message
                # Optionally display a more prominent error message
                # self.display_error(f"Failed to load backend for {personality_name}. Check logs.")
        else:
            gui_logger.warning(
                f"Received backend_ready signal for '{personality_name}', but current personality is '{self.current_personality}'. Ignoring.")

    @pyqtSlot()
    def on_worker_finished(self):
        """Slot called when worker thread finishes."""
        # This might be called during personality switching OR if the worker crashes
        gui_logger.info(f"Worker thread finished signal received (Current Personality: '{self.current_personality}').")
        # If the currently selected personality's worker finished unexpectedly, update UI
        # We might need a flag to distinguish between intentional stop (switching) and crash
        # For now, assume if worker stops, UI should reflect inactive state unless switching is in progress
        # This logic might need refinement depending on desired behavior on worker crash.
        # If no worker is running for the current personality, set to not ready.
        # This logic might need refinement if the worker finishes *during* a switch.
        if self.current_personality and (not self.worker or not self.worker.isRunning()):
            # Check if we are *expecting* it to be stopped (e.g., during switch) might be complex.
            # Safest is perhaps to check if a personality is selected.
            if self.current_personality:
                gui_logger.info(
                    f"Worker stopped unexpectedly for {self.current_personality}. Setting status to not ready.")
                # self.update_status_light("not_ready")
                # self.set_input_enabled(False)
                # self.statusBar().showMessage(f"Worker for '{self.current_personality}' stopped.", 0)

    def clear_chat_display(self):
        # (Implementation remains the same)
        while self.chat_layout.count() > 0:
            item = self.chat_layout.takeAt(self.chat_layout.count() - 1);
            if item is None: continue
            widget = item.widget();
            layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif layout is not None:
                while layout.count() > 0:
                    child_item = layout.takeAt(0);
                    if child_item and child_item.widget(): child_item.widget().deleteLater()
            elif isinstance(item, QSpacerItem):
                pass
        self.chat_layout.addStretch()

    @pyqtSlot(list)
    def display_initial_history(self, history_turns: list):
        """Displays the initial history turns from the previous session."""
        gui_logger.info(f"Received {len(history_turns)} initial history turns to display.")
        if not history_turns:
            return

        # Add a visual separator
        separator_label = QLabel("--- Previous Conversation ---")
        separator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator_label.setStyleSheet("color: #888; font-style: italic; margin-top: 10px; margin-bottom: 5px;")
        self._add_widget_to_chat_layout(separator_label)

        # Display each turn
        for turn in history_turns:
            speaker = turn.get("speaker", "?")
            text = turn.get("text", "")
            uuid = turn.get("uuid") # Get UUID if available
            # Call display_message, passing UUID only if it's an AI turn
            ai_node_uuid = uuid if speaker == "AI" else None
            self.display_message(speaker, text, ai_node_uuid=ai_node_uuid)

        # Scroll to bottom after adding initial history
        QTimer.singleShot(100, self._scroll_to_bottom)

    def show_startup_error(self, message):
        # (Implementation remains the same)
        QMessageBox.critical(None, "Startup Error", message)

    def send_message(self):
        """Sends user input and potentially attachment payload to worker."""
        user_input_text = self.input_field.text().strip()
        attachment_to_send = None
        attachment_info_for_display = None  # Store info needed for display (type, filename)

        current_attachment = getattr(self, 'attachment_payload', None)
        if current_attachment:
            file_type = current_attachment.get('type')
            file_name = current_attachment.get('filename')
            gui_logger.debug(f"Attachment payload detected: Type={file_type}, File={file_name}")
            attachment_to_send = current_attachment
            # Include base64 in display info ONLY if it's an image
            attachment_info_for_display = {"type": file_type, "filename": file_name}
            if file_type == 'image':
                attachment_info_for_display['base64_string'] = current_attachment.get('base64_string')

            # Remove placeholder from text input
            placeholder_pattern = r'\s*\[(Image|File) Attached:\s*.*?\s*\]\s*'
            user_input_text = re.sub(placeholder_pattern, '', user_input_text).strip()
            gui_logger.debug(f"User text after removing placeholder: '{user_input_text}'")

            # Clear payload after preparing it for sending
            self.clear_attachment(clear_status=False)
        else:
            gui_logger.debug("No attachment payload detected.")

        if user_input_text or attachment_to_send:  # Proceed if text OR attachment exists
            # --- Clear clarification state when user sends input ---
            if self.awaiting_clarification:
                gui_logger.info("User provided input while clarification was pending. Clearing flag.")
                self.awaiting_clarification = False
                # Placeholder text will be reset by _finalize_display or next clarification

            if not self.is_processing:
                self.user_scrolled_up = False # Reset scroll flag when user sends message
                # (Command handling logic remains the same)
                if not attachment_to_send:
                    if user_input_text.lower() == "/consolidate":
                        self.request_consolidation()
                        self.input_field.clear()
                        return
                    if user_input_text.lower() == "/reset":
                        self.confirm_reset_memory()
                        self.input_field.clear()
                        return
                    if user_input_text.lower() == "/clear":
                        self.clear_attachment()
                        return

                self.is_processing = True
                self.set_input_enabled(False)
                self.update_status_light("processing")  # Orange light
                self.statusBar().showMessage("Processing...", 0)  # Persistent message

                # --- Display user message (potentially with attachment info) ---
                display_text = user_input_text  # Text part
                self.display_message("User", display_text, attachment_info=attachment_info_for_display)

                # --- Send data to worker ---
                gui_logger.debug(
                    f"Sending to worker: Text='{user_input_text[:50]}...', "
                    f"Attachment Type='{attachment_to_send.get('type') if attachment_to_send else None}'"
                )
                self.worker.add_input(text=user_input_text, attachment=attachment_to_send)

                self.input_field.clear()
            else:
                gui_logger.warning("Processing already in progress.")
                self.statusBar().showMessage("Please wait...", 2000)
        else:
            gui_logger.info("Empty input and no attachment.")
            self.clear_attachment(clear_status=False)

    def confirm_reset_memory(self):
        # (Implementation remains the same)
        if self.is_processing: self.statusBar().showMessage("Cannot reset while processing.", 3000); return
        if not self.current_personality: QMessageBox.warning(self, "No Personality",
                                                             "Please select a personality first."); return
        reply = QMessageBox.warning(self, 'Confirm Reset',
                                    f"Reset ALL memory for '{self.current_personality}'?\nCannot be undone.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Yes:
            gui_logger.info(
                f"User confirmed memory reset for {self.current_personality}."); self.statusBar().showMessage(
                "Resetting memory..."); self.set_input_enabled(
                False); self.is_processing = True; self.worker.request_memory_reset()
        else:
            gui_logger.info("User cancelled memory reset."); self.statusBar().showMessage("Memory reset cancelled.",
                                                                                          3000)

    def request_consolidation(self):
        # (Implementation remains the same)
        if self.is_processing: self.statusBar().showMessage("Cannot consolidate while processing.", 3000); return
        if not self.current_personality: QMessageBox.warning(self, "No Personality",
                                                             "Please select a personality first."); return
        gui_logger.info(f"User requested manual consolidation for {self.current_personality}.");
        self.statusBar().showMessage("Requesting consolidation...");
        self.set_input_enabled(False);
        self.is_processing = True;
        self.worker.request_consolidation()

    @pyqtSlot(str)
    def on_consolidation_complete(self, status_message):
        # (Implementation remains the same - re-enables input via _finalize_display)
        gui_logger.info(f"Consolidation complete signal received: {status_message}")
        self.display_message("System", status_message, object_name_suffix="ConfirmationMessage")
        self._finalize_display(status_msg="Consolidation Finished.", status_duration=5000)

    @pyqtSlot()
    def on_memory_reset_complete(self):
        # (Implementation remains the same - re-enables input via _finalize_display)
        gui_logger.info("Memory reset complete signal received by GUI.")
        self.clear_chat_display()
        self.display_message("System", "Memory has been reset.", object_name_suffix="ConfirmationMessage")
        self.user_scrolled_up = False # Reset scroll flag after reset
        self._finalize_display(status_msg="Memory Reset Complete.", status_duration=5000)

    @pyqtSlot(str, dict)
    def handle_confirmation_request(self, action_type: str, details: dict):
        """Handles the signal that user confirmation is needed for an action."""
        gui_logger.info(f"Confirmation needed: Type={action_type}, Details={details}")
        self.pending_confirmation = {'action': action_type, 'args': details}  # Store details

        confirm_msg = "Confirmation required."
        title = "Confirm Action"
        if action_type == "confirm_overwrite":
            filename = details.get("filename", "?")
            title = "Confirm Overwrite"
            confirm_msg = f"File '{filename}' already exists.\n\nDo you want to overwrite it?"

        reply = QMessageBox.question(self, title, confirm_msg,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)  # Default to No

        if reply == QMessageBox.StandardButton.Yes:
            gui_logger.info("User confirmed action.")
            if self.worker and self.pending_confirmation:
                # Queue the confirmed action task
                # We need the original action type ('create_file') and args
                confirmed_action_data = {
                    'action': 'create_file',  # The action to execute
                    'args': self.pending_confirmation['args']
                }
                self.worker.input_queue.append(('execute_action_confirmed', confirmed_action_data))
                self.statusBar().showMessage("Executing confirmed action...")
                self.is_processing = True  # Mark as processing again
                self.set_input_enabled(False)
            else:
                gui_logger.error("Cannot execute confirmed action: Worker or pending data missing.")
                self.display_error("Could not execute confirmed action.")
                self._finalize_display()  # Reset UI state
        else:
            gui_logger.info("User cancelled action.")
            self.display_message("System", f"Action cancelled by user.", object_name_suffix="ConfirmationMessage")
            self._finalize_display(status_msg="Action cancelled.", status_duration=3000)

        # Clear pending state regardless of choice
        self.pending_confirmation = None

    @pyqtSlot(str, list)
    def handle_clarification_request(self, original_action, missing_args):
        """Handles the signal that the backend needs more info for an action."""
        gui_logger.info(f"Clarification needed for action '{original_action}', missing: {missing_args}")
        self.awaiting_clarification = True  # Set the flag

        missing_str = ", ".join([f"'{arg}'" for arg in missing_args])
        clarification_msg = (
            f"To perform the '{original_action}' action, I still need: **{missing_str}**.\n\n"
            f"Please provide the missing information."
        )
        # Display the request message
        self.display_message("AI", clarification_msg, object_name_suffix="ConfirmationMessage")

        # Update UI state persistently
        status_msg = f"Waiting for: {missing_str}"
        self.statusBar().showMessage(status_msg, 0)  # Persistent message
        if hasattr(self, 'input_field'):
            self.input_field.setPlaceholderText(f"Enter the missing info for '{original_action}'...")
        # Ensure input is enabled (it should be if we got here, but double-check)
        self.set_input_enabled(True)
        self.update_status_light("loading")  # Use yellow light to indicate waiting state
        self.is_processing = False  # No longer processing the initial request

    @pyqtSlot(str, str, list)
    def display_response(self, user_input, ai_response, memories):
        # (Implementation remains the same - calls display_message, re-enables via _finalize_display)
        gui_logger.debug(f"GUI received AI response: {ai_response[:50]}...")
        self.display_message("AI", ai_response)
        if memories:
            gui_logger.debug(f"Creating CollapsibleMemoryWidget with {len(memories)} memories.")
            try:
                memory_widget = CollapsibleMemoryWidget(memories, self.scroll_widget)
                # --- Connect the new signal ---
                memory_widget.saliency_feedback_requested.connect(self.handle_saliency_feedback)
                # --- Add widget to layout ---
                mem_container_layout = QHBoxLayout()
                mem_container_layout.setContentsMargins(self.bubble_edge_margin + 5, 0, self.bubble_side_margin, 0)
                mem_container_layout.addWidget(memory_widget)
                self._add_widget_to_chat_layout(mem_container_layout)
            except Exception as e:
                gui_logger.error(f"Failed to create/connect memory widget: {e}", exc_info=True)
                self.display_error(f"Failed display memories: {e}")
        else:
            gui_logger.debug("No memories received for this interaction.")

    # --- Slot signature updated to accept 5th argument (needs_planning flag) ---
    @pyqtSlot(str, str, list, str, bool)
    def display_response(self, user_input, ai_response, memories, ai_node_uuid, needs_planning):
        """Displays AI response and memories. Workspace results handled separately."""
        # This slot ONLY displays the AI's conversational response and memories.
        # The needs_planning flag is received but not directly used for display here.
        # Workspace results will arrive via modification_response_ready signal later if planning was needed.
        gui_logger.debug(f"GUI received AI response: '{strip_emojis(ai_response[:50])}...' (UUID: {ai_node_uuid})") # Strip emojis
        # --- Pass ai_node_uuid to display_message for the AI bubble ---
        self.display_message("AI", ai_response, ai_node_uuid=ai_node_uuid)
        if memories:
            gui_logger.debug(f"Creating CollapsibleMemoryWidget with {len(memories)} memories.")
            try:
                memory_widget = CollapsibleMemoryWidget(memories, self.scroll_widget)
                # --- Connect the new signal ---
                memory_widget.saliency_feedback_requested.connect(self.handle_saliency_feedback)
                # --- Add widget to layout ---
                mem_container_layout = QHBoxLayout()
                mem_container_layout.setContentsMargins(self.bubble_edge_margin + 5, 0, self.bubble_side_margin, 0)
                mem_container_layout.addWidget(memory_widget)
                self._add_widget_to_chat_layout(mem_container_layout)
            except Exception as e:
                gui_logger.error(f"Failed to create/connect memory widget: {e}", exc_info=True)
                self.display_error(f"Failed display memories: {e}")
        else:
            gui_logger.debug("No memories received for this interaction.")
        self._finalize_display()

    @pyqtSlot(str, str, str, str)
    def display_modification_confirmation(self, user_input, confirmation_message, action_type, target_info):
        """Displays confirmation/result messages for actions, memory mods, etc., using a styled task bubble."""
        gui_logger.debug(
            f"GUI received confirmation/result. Action Type: {action_type}, Target: {target_info}, Msg: {confirmation_message}")

        # --- Determine Action Category and Status ---
        parts = action_type.split('_');
        type_prefix = parts[0];
        success_suffix = parts[-1] if len(parts) > 1 else "unknown"
        is_success = success_suffix == "success"
        is_fail = success_suffix in ["fail", "exception"]
        is_file_action = type_prefix in ["create", "append", "read", "delete", "list"]  # Include list
        is_calendar_action = type_prefix in ["add", "read"] and "calendar" in action_type  # Be more specific
        is_memory_action = type_prefix in ["delete", "edit", "forget"]

        # --- Create Task Bubble Frame ---
        task_frame = QFrame()
        task_frame.setObjectName("TaskBubbleFrame")
        task_layout = QVBoxLayout(task_frame)
        task_layout.setContentsMargins(10, 8, 10, 8)  # Slightly different margins
        task_layout.setSpacing(4)

        # --- Add Main Message Label ---
        # Use confirmation_message directly, but maybe shorten very long file reads?
        display_message = confirmation_message
        if action_type == "read_file_success" and len(confirmation_message) > 300:
            display_message = confirmation_message[:300] + "\n... (file content truncated)"

        message_label = QLabel(display_message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # Allow selecting text
        message_label.setObjectName("TaskLabel")
        task_layout.addWidget(message_label)

        # --- Add Clickable File Link (if applicable and successful) ---
        filename_to_link = None
        # *** MODIFIED: Don't link for delete_file_success ***
        if is_success and is_file_action and action_type != "list_files_success" and action_type != "delete_file_success":
            # Extract filename from target_info (which is likely args dict string repr)
            # Example target_info: "{'filename': 'notes.txt', 'content': '...'}"; "{'filename': 'report.txt'}"
            filename_match = re.search(r"'filename':\s*'([^']+)'", target_info) # Look for 'filename': '...'
            if filename_match:
                filename_to_link = filename_match.group(1)
                logger.debug(f"Extracted filename '{filename_to_link}' from target_info using regex.")
            # Fallback: Try extracting from the success message itself (less reliable)
            elif action_type in ["create_file_success", "append_file_success", "read_file_success"]:
                # Try extracting from the success message itself (less reliable)
                msg_match = re.search(r"'(.*?)'", confirmation_message)
                if msg_match: filename_to_link = msg_match.group(1)

        if filename_to_link:
            # Construct the full path using file_manager helper
            # Use file_manager helper directly, avoid relying on self.config if worker isn't ready
            workspace_path = file_manager.get_workspace_path(
                self.worker.client.config if self.worker and self.worker.client else {}, self.current_personality)
            if workspace_path:
                full_file_path = os.path.join(workspace_path, filename_to_link)
                if os.path.exists(full_file_path):
                    # Create file URL
                    file_url = QUrl.fromLocalFile(full_file_path).toString()
                    link_label = QLabel(f"<a href='{file_url}'>Open '{filename_to_link}'</a>")
                    link_label.setOpenExternalLinks(False)  # We handle the click
                    link_label.linkActivated.connect(self.open_local_link)
                    link_label.setObjectName("TaskFilenameLink")
                    link_label.setToolTip(f"Click to open file: {full_file_path}")
                    task_layout.addWidget(link_label)
                else:
                    gui_logger.warning(f"File path for link does not exist: {full_file_path}")
            else:
                gui_logger.warning("Could not get workspace path to create file link.")

        # --- Add Task Bubble to Chat Layout ---
        # Use a simple row layout to allow alignment (optional, could add directly)
        row_layout = QHBoxLayout();
        row_layout.setSpacing(0);
        row_layout.setContentsMargins(0, 0, 0, 0)
        # Add some spacing before the task frame
        row_layout.addSpacerItem(
            QSpacerItem(self.bubble_edge_margin + 10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        row_layout.addWidget(task_frame)
        row_layout.addStretch(1)  # Push to left
        self._add_widget_to_chat_layout(row_layout)

        # --- Finalize UI State ---
        # Determine status message based on success/failure
        final_status = "Action Completed." if is_success else "Action Failed."
        if action_type == "none":
            final_status = "No action taken."
        elif "clarify" in action_type:
            final_status = "Waiting for clarification..."  # Should be handled by finalize_display logic

        self._finalize_display(status_msg=final_status, status_duration=4000)

    @pyqtSlot(str)
    def open_local_link(self, link_str: str):
        """Opens a local file link using QDesktopServices."""
        gui_logger.info(f"Attempting to open local link: {link_str}")
        url = QUrl(link_str)
        if url.isLocalFile():
            opened = QDesktopServices.openUrl(url)
            if not opened:
                gui_logger.error(f"Failed to open local file link: {link_str}")
                QMessageBox.warning(self, "Open Failed", f"Could not open the file:\n{url.toLocalFile()}")
        else:
            gui_logger.warning(f"Link is not a local file: {link_str}")

    @pyqtSlot(str, str)
    def handle_saliency_feedback(self, uuid: str, direction: str):
        """Handles the signal from memory widget and requests worker update."""
        gui_logger.info(f"GUI received saliency feedback: UUID={uuid}, Direction={direction}")
        if self.worker and self.worker.isRunning():
            self.worker.request_saliency_update(uuid, direction)
            # Optionally provide brief status bar feedback
            self.statusBar().showMessage(f"Updating saliency for {uuid[:8]} ({direction})...", 2000)
        else:
            gui_logger.error("Cannot handle saliency feedback: Worker not running.")
            self.display_error("Cannot update saliency: Backend worker not active.")

    def handle_feedback_click(self, node_uuid: str, feedback_type: str):
        """Handles clicks on feedback buttons and signals the worker."""
        gui_logger.info(f"Feedback button clicked: UUID={node_uuid}, Type={feedback_type}")
        if self.worker and self.worker.isRunning():
            # Find the specific label that was clicked to potentially disable it
            # This requires storing references or finding the widget, which is complex here.
            # For now, just send the signal and provide status bar feedback.
            self.worker.input_queue.append(('feedback', {'uuid': node_uuid, 'type': feedback_type}))
            self.statusBar().showMessage(f"Feedback ({feedback_type}) registered for message.", 3000)
            # TODO: Visually disable/change the clicked button?
        else:
            gui_logger.error("Cannot handle feedback click: Worker not running.")
            self.display_error("Cannot register feedback: Backend worker not active.")

    @pyqtSlot(str)
    def display_error(self, error_message):
        # (Implementation remains the same - calls display_message, re-enables via _finalize_display)
        gui_logger.error(f"GUI received error: {error_message}")
        self.display_message("Error", f"System Error: {error_message}")
        self._finalize_display(status_msg="Error occurred.", status_duration=5000)

    def _add_widget_to_chat_layout(self, widget_or_layout):
        # (Implementation remains the same)
        stretch_item = None
        if self.chat_layout.count() > 0:
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if isinstance(last_item, QSpacerItem): stretch_item = self.chat_layout.takeAt(self.chat_layout.count() - 1)
        insert_index = self.chat_layout.count()
        if isinstance(widget_or_layout, QWidget):
            self.chat_layout.insertWidget(insert_index, widget_or_layout)
        elif isinstance(widget_or_layout, (QHBoxLayout, QVBoxLayout)):
            self.chat_layout.insertLayout(insert_index, widget_or_layout)
        else:
            gui_logger.warning(f"Attempted to add unsupported item type to chat layout: {type(widget_or_layout)}")
        if stretch_item:
            self.chat_layout.addItem(stretch_item)
        else:
            self.chat_layout.addStretch()
        # QTimer.singleShot(50, self._scroll_to_bottom) # Removed: Scrolling handled by rangeChanged signal

    def _finalize_display(self, status_msg="Ready.", status_duration=3000):
        """ Common actions after displaying content or finishing a task. """
        # Always mark processing as finished when this is called
        self.is_processing = False

        # --- Determine UI State based on personality, processing, and clarification ---
        should_be_enabled = bool(self.current_personality) and not self.is_processing
        final_status_msg = status_msg
        final_status_duration = status_duration
        final_placeholder = "Type message, paste image, or /command..."
        final_light_status = "not_ready"  # Default to red if no personality

        if self.current_personality:
            if self.is_processing:  # Should generally not happen if called correctly, but handle defensively
                final_status_msg = "Processing..."
                final_status_duration = 0  # Persistent
                final_placeholder = "Processing..."
                final_light_status = "processing"  # Orange while processing
                should_be_enabled = False
            elif self.awaiting_clarification:
                # Find the missing args from the worker's pending state (if possible)
                # This is a bit indirect, ideally the signal would carry this info again
                missing_args_str = "missing info"
                if self.worker and self.worker.pending_clarification:
                    missing = self.worker.pending_clarification.get('missing_args', [])
                    if missing: missing_args_str = ", ".join([f"'{arg}'" for arg in missing])

                final_status_msg = f"Waiting for: {missing_args_str}"
                final_status_duration = 0  # Persistent
                final_placeholder = f"Enter the missing info..."
                final_light_status = "loading"  # Yellow while waiting for clarification
                should_be_enabled = True  # Input should be enabled to provide clarification
            else:
                # Normal ready state
                final_status_msg = status_msg  # Use the message passed in (e.g., "Ready.", "Consolidation complete.")
                final_status_duration = status_duration
                final_placeholder = "Type message, paste image, or /command..."
                final_light_status = "ready"  # Green light
                should_be_enabled = True
        else:
            # No personality loaded state
            final_status_msg = "Please select a personality from the menu."
            final_status_duration = 0  # Persistent
            final_placeholder = "Select a personality from the menu to start..."
            final_light_status = "not_ready"  # Red light
            should_be_enabled = False

        # --- Apply Final UI State ---
        self.statusBar().showMessage(final_status_msg, final_status_duration)
        self.set_input_enabled(should_be_enabled)  # Handles placeholder text too
        if hasattr(self, 'input_field'):  # Update placeholder specifically if needed
            self.input_field.setPlaceholderText(final_placeholder)
        self.update_status_light(final_light_status)

    def _scroll_to_bottom(self):
        """Scrolls to the bottom only if the user hasn't manually scrolled up."""
        if not self.user_scrolled_up:
            try:
                scrollbar = self.scroll_area.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            except Exception as e:
                gui_logger.debug(f"Minor error during scroll to bottom: {e}")
        # else:
            # gui_logger.debug("Skipping auto-scroll, user has scrolled up.")

    @pyqtSlot(int)
    def _handle_scroll_value_changed(self, value: int):
        """Updates the user_scrolled_up flag based on scrollbar position."""
        scrollbar = self.scroll_area.verticalScrollBar()
        # Consider scrolled up if not at the maximum value
        is_at_bottom = (value == scrollbar.maximum())
        if not is_at_bottom and not self.user_scrolled_up:
            gui_logger.debug("User scrolled up.")
            self.user_scrolled_up = True
        elif is_at_bottom and self.user_scrolled_up:
            gui_logger.debug("User scrolled back to bottom.")
            self.user_scrolled_up = False

    @pyqtSlot(int, int)
    def _scroll_to_bottom_on_range_change(self, min_val, max_val):
        """Scrolls to bottom when the scroll range changes, respecting user scroll."""
        # This method is called when content is added, potentially changing the max value.
        # We should scroll down if the user hasn't scrolled up.
        self._scroll_to_bottom() # _scroll_to_bottom now contains the check

    def handle_attach_file(self):
        """Opens a file dialog allowing any file type and calls the path handler."""
        if self.is_processing: self.statusBar().showMessage("Cannot attach file while processing.", 3000); return
        # Allow selecting any file type
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Attach", "", "All Files (*)")
        if file_path:
            self.handle_attach_file_path(file_path)
        else:
            gui_logger.info("File selection cancelled via dialog.")

    def handle_attach_file_path(self, file_path: str):
        """Handles attaching a file via path, differentiating images and other files."""
        if not file_path or not isinstance(file_path, str): gui_logger.warning("Invalid file path."); return
        file_name = os.path.basename(file_path)
        if not os.path.isfile(file_path): gui_logger.error(f"Path not valid file: {file_path}"); QMessageBox.warning(
            self, "Invalid File Path", "Path not valid."); return

        # --- Check if it's an image ---
        file_extension = os.path.splitext(file_name)[1].lower()
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        is_image = file_extension in image_extensions

        if is_image:
            gui_logger.debug(f"Processing image file: {file_name}")
            try:
                # --- Read image data ---
                with open(file_path, "rb") as image_file:
                    binary_data = image_file.read()
                base64_encoded_data = base64.b64encode(binary_data)
                base64_string = base64_encoded_data.decode("utf-8")
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                image_data_url = f"data:{mime_type};base64,{base64_string}"

                # --- Create payload ---
                payload = {
                    "type": "image",
                    "filename": file_name,  # Use original filename
                    "data_url": image_data_url,
                    "base64_string": base64_string,
                }
                # --- Call common payload handler ---
                self.handle_attach_payload(payload)

            except Exception as e:
                gui_logger.error(f"Error processing image file path {file_path}: {e}", exc_info=True)
                self.display_error(f"Error attaching image file: {e}")
                self.clear_attachment()
        else:
            # --- Handle Generic File ---
            gui_logger.info(f"Attaching generic file: {file_name}")
            # Create a simpler payload for generic files
            payload = {
                "type": "file",
                "filename": file_name,
                "path": file_path,  # Store the path for potential future use (backend needs changes to use this)
            }
            # Call common payload handler
            self.handle_attach_payload(payload)


    def handle_attach_payload(self, payload: dict):
        """Stores attachment payload and updates UI."""
        if not payload or "type" not in payload or "filename" not in payload:
            gui_logger.warning("Invalid payload received by handle_attach_payload.")
            return

        try:
            file_name = payload.get("filename", "attached_file")
            file_type = payload.get("type")  # 'image' or 'file'

            # --- Store payload ---
            self.clear_attachment(clear_status=False)  # Clear previous first
            self.attachment_payload = payload
            gui_logger.info(f"Attachment payload stored: Type={file_type}, Name={file_name}")

            # --- Update UI ---
            current_text = self.input_field.text()
            # Clean any previous placeholder before adding new one
            current_text = re.sub(r'(^|\s)\[(Image|File) Attached:\s*.*?\s*\](\s|$)', r'\1\3', current_text).strip()

            # Use appropriate placeholder text
            placeholder_prefix = "Image" if file_type == "image" else "File"
            placeholder_text = f" [{placeholder_prefix} Attached: {file_name}] "

            separator = " " if current_text and not current_text.endswith(" ") else ""
            self.input_field.setText(current_text + separator + placeholder_text)
            self.input_field.setFocus()
            self.statusBar().showMessage(f"Ready to send with {placeholder_prefix.lower()}: {file_name}", 5000)

        except Exception as e:
            gui_logger.error(f"Error handling attachment payload for {payload.get('filename', '?')}: {e}", exc_info=True)
            # Just log and clear, don't call display_error which might trigger finalize_display prematurely
            self.clear_attachment()

        # --- Removed duplicate handle_attach_payload method ---


    def clear_attachment(self, clear_status=True):
        """Helper to clear the attachment payload and UI placeholder."""
        cleared = False
        # Check and clear the payload now
        if hasattr(self, 'attachment_payload') and self.attachment_payload:
            gui_logger.info(f"Clearing attachment: {self.attachment_payload.get('filename', 'unknown')}")
            self.attachment_payload = None
            cleared = True

        # Still clear placeholder text
        current_text = self.input_field.text();
        cleaned_text = re.sub(r'(^|\s)\[(Image|File) Attached:\s*.*?\s*\](\s|$)', r'\1\3', current_text).strip()
        if cleaned_text != current_text: self.input_field.setText(cleaned_text); cleared = True
        if cleared and clear_status: self.statusBar().showMessage("Attachment cleared.", 3000)



        # --- Signature updated to accept attachment_info and ai_node_uuid ---


    def display_message(self, speaker, text, attachment_info: dict | None = None, object_name_suffix="Message",
                        ai_node_uuid: str | None = None):
        """Adds a message bubble with optional text, image thumbnail or file placeholder, and timestamp."""

        # --- Initialize variables ---
        attachment_label = None
        file_type = None # Initialize file_type
        file_name = None # Initialize file_name

        # --- Create Attachment Label (Image or File Placeholder) ---
        if attachment_info:
            file_type = attachment_info.get("type") # Assign if info exists
            file_name = attachment_info.get("filename", "attached_file") # Assign if info exists

            if file_type == "image":
                # Get base64 string directly from the passed attachment_info
                image_base64 = attachment_info.get('base64_string')

                if image_base64:
                    try:
                        # Decode and create pixmap
                        image_bytes = base64.b64decode(image_base64)
                        pixmap = QPixmap()
                        loaded = pixmap.loadFromData(image_bytes)

                        if loaded and not pixmap.isNull():
                            # Scale pixmap to thumbnail size (using configured max width)
                            scaled_pixmap = pixmap.scaledToWidth(
                                self.thumbnail_max_width,
                                Qt.TransformationMode.SmoothTransformation
                            )
                            attachment_label = QLabel()
                            attachment_label.setPixmap(scaled_pixmap)
                            attachment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center image within label
                            attachment_label.setStyleSheet(
                                "background-color: transparent; border: 1px solid #555; margin-bottom: 4px;")  # Add border/margin
                        else:
                            gui_logger.warning("Failed to load QPixmap from base64 data.")
                            attachment_label = QLabel("[Image Error]")  # Fallback text
                            attachment_label.setStyleSheet("color: #FF8C8C;")
                    except Exception as e:
                        gui_logger.error(f"Error processing image base64 for display: {e}", exc_info=True)
                        attachment_label = QLabel("[Image Load Error]")
                        attachment_label.setStyleSheet("color: #FF8C8C;")
                else:
                    gui_logger.warning("Image attachment info provided, but base64 data missing for display.")
                    attachment_label = QLabel(f"[Image: {file_name}]")  # Show filename as fallback
                    attachment_label.setStyleSheet("color: #AAAAAA; font-style: italic;")


        elif file_type == "file":

            # Display placeholder for generic files

            attachment_label = QLabel(f"[File Attached: {file_name}]")

            attachment_label.setStyleSheet(

                "color: #AAAAAA; font-style: italic; background-color: transparent; border: 1px dashed #555; padding: 4px; margin-bottom: 4px;"

            )

        else:

            gui_logger.warning(f"Unknown attachment type in attachment_info: {file_type}")

        # --- Create Main Message Label (if text provided) ---
        message_label = None
        if text:  # Only create label if there is text
            message_label = QLabel()
            message_label.setWordWrap(True)
            message_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            message_label.setOpenExternalLinks(True)
            # (HTML processing remains the same...)
            escaped_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            processed_text = re.sub(r"(?<![*\n])\*\*(?![*\n])(.+?)(?<![*\n])\*\*(?![*\n])", r"<b>\1</b>", escaped_text)
            lines = processed_text.split("\n")
            html_lines = []
            for line in lines:
                is_list_item = re.match(r"^\s*[\*\-\+]\s+", line)
                if is_list_item:
                    line_content = re.sub(r"^\s*[\*\-\+]\s+", "", line)
                    html_lines.append(f"<div style='margin-left: 15px; padding-left: 5px;'>&bull;&nbsp;{line_content}</div>")
                else:
                    html_lines.append(line)
            processed_text = "<br>".join(html_lines)
            processed_text = re.sub(r"(<br>\s*){2,}", "<br>", processed_text)
            message_label.setText(processed_text)
            # Set object name for potential styling (needed for text color from main bubble style)
            message_label.setObjectName(f"{speaker}MessageLabel")

        # --- Create Timestamp Label (Generated Locally) ---
        timestamp_label = None
        # (Timestamp generation/conversion logic remains the same...)
        time_str_display = ""
        try:
            dt_obj_utc = datetime.now(timezone.utc)
            if ZoneInfo:
                try:
                    german_tz = ZoneInfo("Europe/Berlin")
                    local_dt = dt_obj_utc.astimezone(german_tz)
                    time_str_display = local_dt.strftime("%H:%M")
                except ZoneInfoNotFoundError:
                    gui_logger.warning("TZ 'Europe/Berlin' not found.")
                    time_str_display = dt_obj_utc.strftime("%H:%M UTC")
                except Exception as tz_err:
                    gui_logger.warning(f"TZ conversion error: {tz_err}")
                    time_str_display = dt_obj_utc.strftime("%H:%M UTC")
            else:
                time_str_display = dt_obj_utc.strftime("%H:%M UTC")
            if time_str_display:
                timestamp_label = QLabel(time_str_display)
                ts_object_name = "TimestampLabel"
                if speaker == "User":
                    ts_object_name = "UserTimestampLabel"
                elif speaker == "AI":
                    ts_object_name = "AITimestampLabel"
                elif speaker == "System":
                    ts_object_name = "SystemTimestampLabel"
                elif speaker == "Error":
                    ts_object_name = "ErrorTimestampLabel"
                timestamp_label.setObjectName(ts_object_name)
                timestamp_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        except Exception as e:
            gui_logger.warning(f"Timestamp error: {e}")
            timestamp_label = None

        # --- Create Feedback Buttons (only for AI messages with UUID) ---
        feedback_layout = None
        if speaker == "AI" and ai_node_uuid:
            feedback_layout = QHBoxLayout()
            feedback_layout.setSpacing(5)
            feedback_layout.addStretch()  # Push buttons to the right

            thumb_up = QLabel("👍")
            thumb_up.setToolTip("Good response")
            thumb_up.setCursor(Qt.CursorShape.PointingHandCursor)
            thumb_up.mousePressEvent = lambda event, u=ai_node_uuid, t="up": self.handle_feedback_click(u, t)

            thumb_down = QLabel("👎")
            thumb_down.setToolTip("Bad response")
            thumb_down.setCursor(Qt.CursorShape.PointingHandCursor)
            thumb_down.mousePressEvent = lambda event, u=ai_node_uuid, t="down": self.handle_feedback_click(u, t)

            feedback_layout.addWidget(thumb_up)
            feedback_layout.addWidget(thumb_down)
            # Add timestamp label to the feedback layout as well
            if timestamp_label:
                feedback_layout.addWidget(timestamp_label)
            else:  # Add spacer if no timestamp
                feedback_layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # --- Assemble Bubble Content ---
        if not attachment_label and not message_label:
            return  # Don't display anything if both attachment and text are missing/failed

        bubble_frame = QFrame()  # Corrected indentation
        bubble_layout = QVBoxLayout(bubble_frame)  # Corrected indentation
        bubble_layout.setContentsMargins(self.bubble_padding_lr, self.bubble_padding_tb, self.bubble_padding_lr,
                                         self.bubble_padding_tb)  # Corrected indentation
        bubble_layout.setSpacing(2)  # Small spacing between elements

        if attachment_label:
            bubble_layout.addWidget(attachment_label)  # Add attachment placeholder/image first
        if message_label:
            bubble_layout.addWidget(message_label)  # Add text second

        # --- Add Feedback Layout (or just timestamp if no feedback) ---
        if feedback_layout:  # Corrected indentation
            bubble_layout.addLayout(feedback_layout)  # Add feedback buttons + timestamp
        elif timestamp_label:  # Corrected indentation
            # Add timestamp directly if no feedback buttons needed
            bubble_layout.addWidget(timestamp_label, alignment=Qt.AlignmentFlag.AlignRight)  # Corrected indentation

        # --- Determine Frame Object Name ---
        frame_name = "AIBubbleFrame";  # Corrected indentation
        if speaker == "User":
            frame_name = "UserBubbleFrame"  # Corrected indentation
        elif "Confirmation" in object_name_suffix:
            frame_name = "AIConfirmationBubbleFrame"  # Corrected indentation
        elif speaker == "AI": frame_name = "AIBubbleFrame"
        elif speaker == "System": frame_name = "AIBubbleFrame"
        elif speaker == "Error": frame_name = "ErrorBubbleFrame"
        bubble_frame.setObjectName(frame_name)  # Corrected indentation

        # --- Row Layout (remains the same) ---
        row_layout = QHBoxLayout();
        row_layout.setSpacing(0);
        row_layout.setContentsMargins(0, 0, 0, 0)  # Corrected indentation
        if speaker == "User":
            row_layout.addStretch(1); row_layout.addWidget(bubble_frame, stretch=1,
                                                           alignment=Qt.AlignmentFlag.AlignRight)  # Corrected indentation
        else:
            row_layout.addWidget(bubble_frame, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft); row_layout.addStretch(
                1)  # Corrected indentation

        self._add_widget_to_chat_layout(row_layout)  # Corrected indentation


        def closeEvent(self, event):
            # (Implementation remains the same)
            gui_logger.info("Close event triggered. Stopping worker thread...")
            if self.worker and self.worker.isRunning(): self.worker.stop();
            if not self.worker or not self.worker.wait(5000):
                gui_logger.warning("Worker thread did not stop gracefully on close.")
            else:
                gui_logger.info("Worker thread stopped on close.")
            event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    # (Keep DPI setting removed from previous fix)
    try:
        app = QApplication(sys.argv)
        font = QFont();
        font.setPointSize(10);
        app.setFont(font)
        window = ChatWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        gui_logger.critical(f"Failed to start application: {e}", exc_info=True)
        try:
            if 'app' not in locals(): app = QApplication(sys.argv)
            QMessageBox.critical(None, "Application Startup Error", f"Failed to start application:\n{e}")
        except Exception as msg_e:
            print(f"CRITICAL STARTUP ERROR: {e}\nMsgBox Error: {msg_e}", file=sys.stderr)
        sys.exit(1)
