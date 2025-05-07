# file_manager.py
import os
import json
import logging
import re
from datetime import datetime, timezone # Keep existing imports

logger = logging.getLogger(__name__)
INVALID_FILENAME_CHARS_PATTERN = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
# DEFAULT_WORKSPACE and DEFAULT_CALENDAR_FILE are now less critical as paths come from config

def _is_filename_safe(filename: str) -> bool: # Keep this helper
    # ... (implementation remains the same) ...
    if not filename or filename in ['.', '..'] or not filename.strip():
        logger.warning(f"Unsafe filename detected (empty, '.', or '..'): '{filename}'")
        return False
    if INVALID_FILENAME_CHARS_PATTERN.search(filename):
        logger.warning(f"Unsafe filename detected (invalid characters): '{filename}'")
        return False
    return True

def get_workspace_path(config: dict, personality_name: str) -> str | None:
    """
    Gets the absolute workspace path for a specific personality, ensuring it exists.
    THIS FUNCTION IS NOW CENTRAL FOR ALL FILE OPERATIONS.
    """
    if not personality_name:
        logger.error("Cannot get workspace path: Personality name is missing.")
        return None
    base_memory_path = config.get('base_memory_path')
    if not base_memory_path:
        logger.error("Config missing 'base_memory_path'. Cannot determine workspace.")
        return None

    # Workspace directory name comes from config, relative to personality data_dir
    # The personality_name is already part of the data_dir in GraphMemoryClient.
    # Here, we need to construct data_dir/workspace_dir_name
    data_dir = os.path.join(base_memory_path, personality_name)
    workspace_dir_name = config.get('workspace_dir', 'Workspace') # Fallback if missing

    workspace_path = os.path.join(data_dir, workspace_dir_name)
    abs_workspace_path = os.path.abspath(workspace_path)
    logger.debug(f"Attempting to ensure workspace directory exists: {abs_workspace_path}")

    try:
        os.makedirs(abs_workspace_path, exist_ok=True)
        if not os.access(abs_workspace_path, os.W_OK):
            logger.error(f"Write permissions check failed for workspace directory: {abs_workspace_path}")
            return None
        return abs_workspace_path
    except OSError as e:
        logger.error(f"OS error creating/accessing workspace directory '{abs_workspace_path}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error ensuring workspace directory '{abs_workspace_path}': {e}", exc_info=True)
        return None

# --- NEW: Helper to check if a file is protected ---
def _is_file_protected(filename: str, config: dict) -> bool:
    protected_files = config.get('protected_workspace_files', [])
    protected_prefixes = config.get('protected_workspace_prefixes', [])
    base_filename = os.path.basename(filename) # Compare against basename

    if base_filename in protected_files:
        logger.warning(f"Operation blocked: File '{base_filename}' is a protected file.")
        return True
    for prefix in protected_prefixes:
        if base_filename.startswith(prefix):
            logger.warning(f"Operation blocked: File '{base_filename}' starts with a protected prefix '{prefix}'.")
            return True
    return False

# --- Modify existing file operation functions ---

def create_or_overwrite_file(config: dict, personality: str, filename: str, content: str) -> tuple[bool, str]:
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    base_filename = os.path.basename(filename)
    if not _is_filename_safe(base_filename):
        err_msg = f"Invalid or unsafe filename: '{filename}'"
        logger.error(err_msg)
        return False, err_msg
    safe_filename = base_filename

    # --- ADDED PROTECTION CHECK (for overwrite scenarios) ---
    file_path = os.path.join(workspace_path, safe_filename)
    if os.path.exists(file_path): # Only check protection if overwriting
        if _is_file_protected(safe_filename, config):
            return False, f"Cannot overwrite protected file: '{safe_filename}'."
    # --- END PROTECTION CHECK ---

    logger.info(f"Attempting write/overwrite: {file_path}")
    # ... (rest of the function: content type check, file write logic - remains the same)
    if not isinstance(content, str):
        logger.warning(f"Content for file '{safe_filename}' is not a string (type: {type(content)}). Converting to string.")
        content = str(content)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        msg = f"File '{safe_filename}' created/overwritten successfully."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except Exception as e:
        err_msg = f"Error writing file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg


def append_to_file(config: dict, personality: str, filename: str, content_to_append: str) -> tuple[bool, str]:
    workspace_path = get_workspace_path(config, personality) # Ensure this uses the new get_workspace_path
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."
    # ... (filename validation, content type check, and file append logic remain the same) ...
    # Note: Appending to a protected file is generally less risky than overwriting/deleting,
    # but you could add a check here too if desired:
    # if _is_file_protected(safe_filename, config):
    #     return False, f"Cannot append to protected file: '{safe_filename}'."
    base_filename = os.path.basename(filename)
    if not _is_filename_safe(base_filename):
        err_msg = f"Invalid or unsafe filename for append: '{filename}'"
        logger.error(err_msg)
        return False, err_msg
    safe_filename = base_filename

    file_path = os.path.join(workspace_path, safe_filename)
    logger.info(f"Attempting append to: {file_path}")
    if not isinstance(content_to_append, str):
        logger.warning(f"Content for appending to '{safe_filename}' is not a string. Converting.")
        content_to_append = str(content_to_append)
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content_to_append + "\n")
        msg = f"Successfully appended to file '{safe_filename}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except Exception as e:
        err_msg = f"Error appending to file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg

def add_calendar_event(config: dict, personality: str, event_date: str, event_time: str, description: str) -> tuple[bool, str]:
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    # Calendar filename now comes from config, used by GraphMemoryClient
    calendar_filename = config.get('calendar_file', 'calendar.jsonl') # Fallback
    safe_calendar_filename = os.path.basename(calendar_filename)

    # --- CALENDAR PROTECTION ---
    # The calendar is typically a protected file, appending is okay.
    # `_is_file_protected` checks against `protected_workspace_files` from config.
    # No explicit check here as `append_to_file` or direct write would be used by a wrapper.
    # If you call this directly, ensure the calendar file isn't overwritten if it's protected.
    # For `add_calendar_event`, it *appends*, which is usually fine for protected files.
    # However, if `calendar.jsonl` was in `protected_files` AND someone tried to *overwrite* it, that would be blocked.

    file_path = os.path.join(workspace_path, safe_calendar_filename)
    # ... (rest of the function remains the same) ...
    if not all(isinstance(arg, str) for arg in [event_date, event_time, description]):
        err_msg = "Invalid argument type for add_calendar_event. All args must be strings."
        return False, err_msg
    if not event_date.strip() or not event_time.strip() or not description.strip():
        err_msg = "Missing required information for calendar event."
        return False, err_msg
    event_record = {
        "timestamp_added": datetime.now(timezone.utc).isoformat(),
        "event_date": event_date.strip(), "event_time": event_time.strip(), "description": description.strip()
    }
    logger.info(f"Attempting to add calendar event to {file_path}: {event_record}")
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(event_record, f); f.write('\n')
        msg = f"Event '{description[:30]}...' added to calendar '{safe_calendar_filename}'."
        return True, msg
    except Exception as e:
        err_msg = f"Error adding calendar event to '{safe_calendar_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg


def list_files(config: dict, personality: str) -> tuple[list[str] | None, str]:
    workspace_path = get_workspace_path(config, personality) # Ensure this uses the new get_workspace_path
    # ... (rest of the function remains the same) ...
    if not workspace_path: return None, f"Could not access workspace for '{personality}'."
    logger.info(f"Listing files in workspace: {workspace_path}")
    try:
        files = [f for f in os.listdir(workspace_path) if os.path.isfile(os.path.join(workspace_path, f))]
        msg = f"Found {len(files)} file(s) in workspace."
        return files, msg
    except Exception as e:
        err_msg = f"Error listing files in '{workspace_path}': {e}"
        logger.error(err_msg, exc_info=True)
        return None, err_msg

def read_file(config: dict, personality: str, filename: str) -> tuple[str | None, str]:
    workspace_path = get_workspace_path(config, personality) # Ensure this uses the new get_workspace_path
    # ... (filename validation, file read logic remains the same) ...
    if not workspace_path: return None, f"Could not access workspace for '{personality}'."
    base_filename = os.path.basename(filename)
    if not _is_filename_safe(base_filename): return None, f"Invalid filename: '{filename}'"
    safe_filename = base_filename
    file_path = os.path.join(workspace_path, safe_filename)
    if not os.path.isfile(file_path): return None, f"File not found: '{safe_filename}'"
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        return content, f"Successfully read '{safe_filename}'."
    except Exception as e:
        err_msg = f"Error reading file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return None, err_msg


def delete_file(config: dict, personality: str, filename: str) -> tuple[bool, str]:
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    base_filename = os.path.basename(filename)
    if not _is_filename_safe(base_filename):
        err_msg = f"Invalid or unsafe filename for deletion: '{filename}'"
        logger.error(err_msg)
        return False, err_msg
    safe_filename = base_filename

    # --- ADDED PROTECTION CHECK ---
    if _is_file_protected(safe_filename, config):
        return False, f"Cannot delete protected file: '{safe_filename}'."
    # --- END PROTECTION CHECK ---

    file_path = os.path.join(workspace_path, safe_filename)
    logger.warning(f"Attempting to DELETE file: {file_path}")
    # ... (rest of file existence check and deletion logic remains the same) ...
    if not os.path.isfile(file_path):
        return False, f"Cannot delete, file not found: '{safe_filename}'."
    try:
        os.remove(file_path)
        msg = f"Successfully deleted file '{safe_filename}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except Exception as e:
        err_msg = f"Error deleting file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg


def read_calendar_events(config: dict, personality: str, target_date: str | None = None) -> tuple[list[dict], str]:
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return [], f"Could not access workspace for personality '{personality}'."

    calendar_filename = config.get('calendar_file', 'calendar.jsonl') # Fallback
    safe_calendar_filename = os.path.basename(calendar_filename)
    file_path = os.path.join(workspace_path, safe_calendar_filename)
    # ... (rest of the function remains the same) ...
    events = []
    if not os.path.exists(file_path):
        return events, f"Calendar file '{safe_calendar_filename}' not found."
    logger.info(f"Reading calendar events from {file_path} (Filter: {target_date or 'All'})")
    lines_failed = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict): lines_failed += 1; continue
                        if target_date:
                            if event.get("event_date") == target_date: events.append(event)
                        else: events.append(event)
                    except json.JSONDecodeError: lines_failed += 1; continue
        msg = f"Found {len(events)} event(s)" + (f" for '{target_date}'" if target_date else "") + "."
        if lines_failed > 0: msg += f" Skipped {lines_failed} invalid line(s)."
        return events, msg
    except Exception as e:
        err_msg = f"Error reading calendar file '{safe_calendar_filename}': {e}"
        return [], err_msg


# --- NEW: archive_file function ---
def archive_file(config: dict, personality: str, filename: str) -> tuple[bool, str]:
    """Moves a file from the workspace to the personality's archive directory."""
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    base_filename = os.path.basename(filename)
    if not _is_filename_safe(base_filename):
        return False, f"Invalid or unsafe filename for archive: '{filename}'"
    safe_filename = base_filename

    # --- Cannot archive protected files ---
    if _is_file_protected(safe_filename, config):
        return False, f"Cannot archive protected file: '{safe_filename}'."

    source_file_path = os.path.join(workspace_path, safe_filename)
    if not os.path.isfile(source_file_path):
        return False, f"File not found in workspace, cannot archive: '{safe_filename}'"

    archive_dir_name = config.get('workspace_archive_dir', '_archive') # Get from config
    archive_path = os.path.join(workspace_path, archive_dir_name) # Archive is inside workspace

    try:
        os.makedirs(archive_path, exist_ok=True) # Ensure archive directory exists
        destination_file_path = os.path.join(archive_path, safe_filename)

        # Handle potential name conflict in archive (e.g., append timestamp)
        if os.path.exists(destination_file_path):
            name, ext = os.path.splitext(safe_filename)
            timestamp_suffix = datetime.now().strftime("_%Y%m%d%H%M%S")
            destination_filename = f"{name}{timestamp_suffix}{ext}"
            destination_file_path = os.path.join(archive_path, destination_filename)
            logger.warning(f"File '{safe_filename}' already in archive. Saving as '{destination_filename}'.")

        os.rename(source_file_path, destination_file_path) # Move the file
        msg = f"File '{safe_filename}' successfully archived to '{archive_dir_name}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except Exception as e:
        err_msg = f"Error archiving file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Source: {source_file_path})", exc_info=True)
        return False, err_msg