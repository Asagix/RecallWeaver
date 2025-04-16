# file_manager.py
import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__) # Use logger from this module

DEFAULT_WORKSPACE = "Workspace"
DEFAULT_CALENDAR_FILE = "calendar.jsonl"

def get_workspace_path(config: dict, personality_name: str) -> str | None:
    """Gets the workspace path for a specific personality, ensuring it exists."""
    base_memory_path = config.get('base_memory_path')
    if not base_memory_path:
        logger.error("Config missing 'base_memory_path'. Cannot determine workspace.")
        return None
    workspace_name = config.get('workspace_dir', DEFAULT_WORKSPACE)
    workspace_path = os.path.join(base_memory_path, personality_name, workspace_name)
    logger.debug(f"Constructed workspace path: {workspace_path}") # Log constructed path
    try:
        os.makedirs(workspace_path, exist_ok=True)
        return os.path.abspath(workspace_path)
    except OSError as e:
        # *** Log specific OSError ***
        logger.error(f"OSError creating workspace directory '{workspace_path}': {e}", exc_info=True)
        return None
    except Exception as e:
         # *** Log any other unexpected error ***
         logger.error(f"Unexpected error creating workspace directory '{workspace_path}': {e}", exc_info=True)
         return None


def create_or_overwrite_file(config: dict, personality: str, filename: str, content: str) -> bool:
    """Creates/overwrites a file in the specific personality's workspace."""
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path: return False
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..']: logger.error(f"Invalid filename: {filename}"); return False
    file_path = os.path.join(workspace_path, safe_filename)

    # *** Log path and content type/length ***
    logger.info(f"Attempting write/overwrite: {file_path}")
    logger.debug(f"Content type: {type(content)}, Length: {len(content) if isinstance(content, str) else 'N/A'}")

    try:
        with open(file_path, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"Successfully wrote file: {safe_filename} in {personality}'s workspace")
        return True
    except IOError as e:
        # *** Log specific IOError ***
        logger.error(f"IOError writing file '{file_path}': {e}", exc_info=True)
        return False
    except Exception as e:
        # *** Log other exceptions ***
        logger.error(f"Unexpected error writing file '{file_path}': {e}", exc_info=True)
        return False

def append_to_file(config: dict, personality: str, filename: str, content_to_append: str) -> bool:
    """Appends to a file in the specific personality's workspace."""
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path: return False
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in ['.', '..']: logger.error(f"Invalid filename: {filename}"); return False
    file_path = os.path.join(workspace_path, safe_filename)

    # *** Log path and content type/length ***
    logger.info(f"Attempting append to: {file_path}")
    logger.debug(f"Content type: {type(content_to_append)}, Length: {len(content_to_append) if isinstance(content_to_append, str) else 'N/A'}")

    try:
        with open(file_path, 'a', encoding='utf-8') as f: f.write(content_to_append + "\n")
        logger.info(f"Successfully appended to file: {safe_filename} in {personality}'s workspace")
        return True
    except IOError as e:
        # *** Log specific IOError ***
        logger.error(f"IOError appending to file '{file_path}': {e}", exc_info=True)
        return False
    except Exception as e:
         # *** Log other exceptions ***
        logger.error(f"Unexpected error appending to file '{file_path}': {e}", exc_info=True)
        return False

def add_calendar_event(config: dict, personality: str, event_date: str, event_time: str, description: str) -> bool:
    """Adds an event to the specific personality's calendar file."""
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path: return False
    calendar_filename = config.get('calendar_file', DEFAULT_CALENDAR_FILE)
    file_path = os.path.join(workspace_path, os.path.basename(calendar_filename))
    event_record = {"timestamp_added": datetime.now(timezone.utc).isoformat(), "event_date": event_date, "event_time": event_time, "description": description}
    logger.info(f"Adding calendar event to {file_path}: {event_record}")

    try:
        with open(file_path, 'a', encoding='utf-8') as f: json.dump(event_record, f); f.write('\n')
        logger.info(f"Successfully added event to {calendar_filename} for {personality}")
        return True
    except IOError as e:
        # *** Log specific IOError ***
        logger.error(f"IOError writing calendar event to '{file_path}': {e}", exc_info=True)
        return False
    except Exception as e:
         # *** Log other exceptions ***
         logger.error(f"Unexpected error adding calendar event: {e}", exc_info=True)
         return False

def read_calendar_events(config: dict, personality: str, target_date: str = None) -> list[dict]:
    """Reads events from the specific personality's calendar file."""
    workspace_path = get_workspace_path(config, personality)
    if not workspace_path: return []
    calendar_filename = config.get('calendar_file', DEFAULT_CALENDAR_FILE)
    file_path = os.path.join(workspace_path, os.path.basename(calendar_filename))
    events = []
    if not os.path.exists(file_path): logger.warning(f"Calendar file not found: {file_path}"); return events
    logger.info(f"Reading calendar events from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try: event = json.loads(line);
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line {line_num} in calendar: {line[:100]}..."); continue
                    if target_date:
                        if event.get("event_date") == target_date: events.append(event)
                    else: events.append(event)
        logger.info(f"Found {len(events)} events (Date filter: {target_date or 'All'}) for {personality}.")
        return events
    except IOError as e: logger.error(f"IOError reading calendar file '{file_path}': {e}", exc_info=True); return []
    except Exception as e: logger.error(f"Unexpected error reading calendar file: {e}", exc_info=True); return []