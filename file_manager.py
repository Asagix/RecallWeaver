# file_manager.py
import os
import json
import logging
from datetime import datetime, timezone, timedelta # Added timedelta for potential future use

logger = logging.getLogger(__name__) # Use logger from this module

DEFAULT_WORKSPACE = "Workspace"
DEFAULT_CALENDAR_FILE = "calendar.jsonl"

def get_workspace_path(config: dict, personality_name: str) -> str | None:
    """
    Gets the absolute workspace path for a specific personality, ensuring it exists.

    Args:
        config: The application configuration dictionary.
        personality_name: The name of the personality.

    Returns:
        The absolute path to the workspace directory, or None if an error occurs.
    """
    if not personality_name:
        logger.error("Cannot get workspace path: Personality name is missing.")
        return None
    base_memory_path = config.get('base_memory_path')
    if not base_memory_path:
        logger.error("Config missing 'base_memory_path'. Cannot determine workspace.")
        return None

    workspace_name = config.get('workspace_dir', DEFAULT_WORKSPACE)
    # Construct path relative to the base memory path and personality
    workspace_path = os.path.join(base_memory_path, personality_name, workspace_name)
    abs_workspace_path = os.path.abspath(workspace_path)
    logger.debug(f"Attempting to ensure workspace directory exists: {abs_workspace_path}")

    try:
        # exist_ok=True prevents error if directory already exists
        os.makedirs(abs_workspace_path, exist_ok=True)
        # Verify write permissions (simple check)
        if not os.access(abs_workspace_path, os.W_OK):
             logger.error(f"Write permissions check failed for workspace directory: {abs_workspace_path}")
             # Optionally raise an error here? For now, return None.
             return None
        return abs_workspace_path
    except OSError as e:
        logger.error(f"OS error creating or accessing workspace directory '{abs_workspace_path}': {e}", exc_info=True)
        return None
    except Exception as e:
         # Catch any other unexpected errors during path creation/verification
         logger.error(f"Unexpected error ensuring workspace directory '{abs_workspace_path}': {e}", exc_info=True)
         return None


def create_or_overwrite_file(config: dict, personality: str, filename: str, content: str) -> tuple[bool, str]:
    """
    Creates or overwrites a file in the specific personality's workspace.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        filename: The desired name of the file.
        content: The string content to write to the file.

    Returns:
        A tuple (success: bool, message: str).
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    # Basic filename sanitization (already done in analyze_action_request, but good safety check)
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip():
        err_msg = f"Invalid or unsafe filename provided: '{filename}'."
        logger.error(err_msg)
        return False, err_msg

    file_path = os.path.join(workspace_path, safe_filename)
    logger.info(f"Attempting write/overwrite: {file_path}")
    logger.debug(f"Content type: {type(content)}, Length: {len(content) if isinstance(content, str) else 'N/A'}")

    if not isinstance(content, str):
        logger.warning(f"Content for file '{safe_filename}' is not a string (type: {type(content)}). Converting to string.")
        content = str(content)

    try:
        # Use 'w' mode to create or overwrite
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        msg = f"File '{safe_filename}' created/overwritten successfully."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except IOError as e:
        err_msg = f"IO error writing file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg
    except PermissionError as e:
         err_msg = f"Permission denied writing file '{safe_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return False, err_msg
    except Exception as e:
        # Catch other potential errors during file writing
        err_msg = f"Unexpected error writing file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg

def append_to_file(config: dict, personality: str, filename: str, content_to_append: str) -> tuple[bool, str]:
    """
    Appends content to a file in the specific personality's workspace. Adds a newline.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        filename: The name of the file to append to.
        content_to_append: The string content to append.

    Returns:
        A tuple (success: bool, message: str).
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip():
        err_msg = f"Invalid or unsafe filename provided: '{filename}'."
        logger.error(err_msg)
        return False, err_msg

    file_path = os.path.join(workspace_path, safe_filename)
    logger.info(f"Attempting append to: {file_path}")
    logger.debug(f"Content type: {type(content_to_append)}, Length: {len(content_to_append) if isinstance(content_to_append, str) else 'N/A'}")

    if not isinstance(content_to_append, str):
        logger.warning(f"Content for appending to '{safe_filename}' is not a string (type: {type(content_to_append)}). Converting to string.")
        content_to_append = str(content_to_append)

    try:
        # Use 'a' mode to append; creates file if it doesn't exist
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content_to_append + "\n") # Add newline for separation
        msg = f"Successfully appended to file '{safe_filename}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except IOError as e:
        err_msg = f"IO error appending to file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg
    except PermissionError as e:
         err_msg = f"Permission denied appending to file '{safe_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return False, err_msg
    except Exception as e:
        err_msg = f"Unexpected error appending to file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg

def add_calendar_event(config: dict, personality: str, event_date: str, event_time: str, description: str) -> tuple[bool, str]:
    """
    Adds an event as a JSON line to the specific personality's calendar file.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        event_date: The date of the event (string, e.g., "2025-12-31", "tomorrow").
        event_time: The time of the event (string, e.g., "14:30", "evening").
        description: The description of the event (string).

    Returns:
        A tuple (success: bool, message: str).
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    calendar_filename = config.get('calendar_file', DEFAULT_CALENDAR_FILE)
    safe_calendar_filename = os.path.basename(calendar_filename) # Sanitize just in case
    file_path = os.path.join(workspace_path, safe_calendar_filename)

    # Validate input types (basic check)
    if not all(isinstance(arg, str) for arg in [event_date, event_time, description]):
         err_msg = "Invalid argument type for add_calendar_event. All args must be strings."
         logger.error(err_msg + f" Received: date({type(event_date)}), time({type(event_time)}), desc({type(description)})")
         return False, err_msg
    if not event_date.strip() or not event_time.strip() or not description.strip():
         err_msg = "Missing required information for calendar event (date, time, or description)."
         logger.error(err_msg)
         return False, err_msg

    event_record = {
        "timestamp_added": datetime.now(timezone.utc).isoformat(),
        "event_date": event_date.strip(),
        "event_time": event_time.strip(),
        "description": description.strip()
    }
    logger.info(f"Attempting to add calendar event to {file_path}: {event_record}")

    try:
        # Use 'a' mode for appending JSON lines
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(event_record, f)
            f.write('\n') # Ensure newline separation for JSONL format
        msg = f"Event '{description[:30]}...' added to calendar '{safe_calendar_filename}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except IOError as e:
        err_msg = f"IO error writing calendar event to '{safe_calendar_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg
    except PermissionError as e:
         err_msg = f"Permission denied writing calendar event to '{safe_calendar_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return False, err_msg
    except TypeError as e: # Catches JSON serialization errors
        err_msg = f"Error serializing calendar event data to JSON: {e}"
        logger.error(err_msg + f" (Data: {event_record})", exc_info=True)
        return False, err_msg
    except Exception as e:
         err_msg = f"Unexpected error adding calendar event: {e}"
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return False, err_msg


def list_files(config: dict, personality: str) -> tuple[list[str] | None, str]:
    """
    Lists files in the specific personality's workspace.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.

    Returns:
        A tuple (file_list: list[str] | None, message: str).
        Returns None for file_list on error.
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return None, f"Could not access workspace for personality '{personality}'."

    logger.info(f"Listing files in workspace: {workspace_path}")
    try:
        # List only files, ignore directories for simplicity for now
        files = [f for f in os.listdir(workspace_path) if os.path.isfile(os.path.join(workspace_path, f))]
        msg = f"Found {len(files)} file(s) in workspace."
        logger.info(msg + f" Files: {files}")
        return files, msg
    except FileNotFoundError:
        # This case should be handled by get_workspace_path, but handle defensively
        err_msg = f"Workspace directory not found: {workspace_path}"
        logger.error(err_msg)
        return None, err_msg
    except PermissionError as e:
        err_msg = f"Permission denied listing files in workspace: {workspace_path}"
        logger.error(err_msg, exc_info=True)
        return None, err_msg
    except Exception as e:
        err_msg = f"Unexpected error listing files in workspace: {e}"
        logger.error(err_msg + f" (Path: {workspace_path})", exc_info=True)
        return None, err_msg


def read_file(config: dict, personality: str, filename: str) -> tuple[str | None, str]:
    """
    Reads the content of a file from the specific personality's workspace.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        filename: The name of the file to read.

    Returns:
        A tuple (content: str | None, message: str).
        Returns None for content on error.
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return None, f"Could not access workspace for personality '{personality}'."

    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip():
        err_msg = f"Invalid or unsafe filename provided for reading: '{filename}'."
        logger.error(err_msg)
        return None, err_msg

    file_path = os.path.join(workspace_path, safe_filename)
    logger.info(f"Attempting to read file: {file_path}")

    if not os.path.exists(file_path):
        err_msg = f"File not found: '{safe_filename}'."
        logger.warning(err_msg + f" (Path: {file_path})")
        return None, err_msg
    if not os.path.isfile(file_path):
        err_msg = f"Path is not a file: '{safe_filename}'."
        logger.warning(err_msg + f" (Path: {file_path})")
        return None, err_msg

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        msg = f"Successfully read content from '{safe_filename}'."
        logger.info(msg + f" (Personality: {personality}, Length: {len(content)})")
        return content, msg
    except IOError as e:
        err_msg = f"IO error reading file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return None, err_msg
    except PermissionError as e:
         err_msg = f"Permission denied reading file '{safe_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return None, err_msg
    except Exception as e:
        err_msg = f"Unexpected error reading file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return None, err_msg


def delete_file(config: dict, personality: str, filename: str) -> tuple[bool, str]:
    """
    Deletes a file from the specific personality's workspace.

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        filename: The name of the file to delete.

    Returns:
        A tuple (success: bool, message: str).
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return False, f"Could not access workspace for personality '{personality}'."

    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..'] or not safe_filename.strip():
        err_msg = f"Invalid or unsafe filename provided for deletion: '{filename}'."
        logger.error(err_msg)
        return False, err_msg

    file_path = os.path.join(workspace_path, safe_filename)
    logger.warning(f"Attempting to DELETE file: {file_path}") # Log deletion attempt as warning

    if not os.path.exists(file_path):
        err_msg = f"Cannot delete, file not found: '{safe_filename}'."
        logger.warning(err_msg + f" (Path: {file_path})")
        # Return success=False but maybe a specific message?
        return False, err_msg
    if not os.path.isfile(file_path):
        err_msg = f"Cannot delete, path is not a file: '{safe_filename}'."
        logger.warning(err_msg + f" (Path: {file_path})")
        return False, err_msg

    try:
        os.remove(file_path)
        msg = f"Successfully deleted file '{safe_filename}'."
        logger.info(msg + f" (Personality: {personality})")
        return True, msg
    except IOError as e:
        err_msg = f"IO error deleting file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg
    except PermissionError as e:
         err_msg = f"Permission denied deleting file '{safe_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return False, err_msg
    except Exception as e:
        err_msg = f"Unexpected error deleting file '{safe_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return False, err_msg


def read_calendar_events(config: dict, personality: str, target_date: str | None = None) -> tuple[list[dict], str]:
    """
    Reads events from the specific personality's calendar file (JSONL format).

    Args:
        config: The application configuration dictionary.
        personality: The name of the personality.
        target_date: Optional string date to filter events by. If None, reads all.

    Returns:
        A tuple (events: list[dict], message: str). The list contains event dictionaries.
        The message indicates success, file not found, or errors encountered.
    """
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path:
        return [], f"Could not access workspace for personality '{personality}'."

    calendar_filename = config.get('calendar_file', DEFAULT_CALENDAR_FILE)
    safe_calendar_filename = os.path.basename(calendar_filename)
    file_path = os.path.join(workspace_path, safe_calendar_filename)

    events = []
    if not os.path.exists(file_path):
        msg = f"Calendar file '{safe_calendar_filename}' not found for personality '{personality}'."
        logger.warning(msg + f" (Path: {file_path})")
        return events, msg # Return empty list and "not found" message

    logger.info(f"Reading calendar events from {file_path} (Filter Date: {target_date or 'All'})")
    lines_read = 0
    lines_failed = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                lines_read += 1
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                             logger.warning(f"Skipping non-dictionary JSON line {line_num} in calendar: {line[:100]}...")
                             lines_failed += 1
                             continue

                        # Filter by date if target_date is provided
                        if target_date:
                            # Simple string comparison for now
                            if event.get("event_date") == target_date:
                                events.append(event)
                        else:
                            events.append(event) # No date filter, add all valid events
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {line_num} in calendar: {line[:100]}...")
                        lines_failed += 1
                        continue

        found_count = len(events)
        filter_str = f" for date '{target_date}'" if target_date else ""
        if lines_failed == 0:
             msg = f"Found {found_count} event(s){filter_str} in '{safe_calendar_filename}'."
        else:
             msg = f"Found {found_count} event(s){filter_str} in '{safe_calendar_filename}'. Skipped {lines_failed} invalid line(s)."
        logger.info(msg + f" (Personality: {personality})")
        return events, msg

    except IOError as e:
        err_msg = f"IO error reading calendar file '{safe_calendar_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return [], err_msg # Return empty list and error message
    except PermissionError as e:
         err_msg = f"Permission denied reading calendar file '{safe_calendar_filename}'."
         logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
         return [], err_msg
    except Exception as e:
        err_msg = f"Unexpected error reading calendar file '{safe_calendar_filename}': {e}"
        logger.error(err_msg + f" (Path: {file_path})", exc_info=True)
        return [], err_msg # Return empty list and error message
