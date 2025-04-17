import logging
import file_manager # Import the existing file manager

logger = logging.getLogger(__name__)

class WorkspaceAgent:
    """
    Handles the execution of a planned sequence of workspace actions.
    """
    def __init__(self, client_config: dict, personality: str):
        """
        Initializes the WorkspaceAgent.

        Args:
            client_config: The main configuration dictionary (from GraphMemoryClient).
            personality: The name of the current personality.
        """
        self.config = client_config
        self.personality = personality
        logger.info(f"WorkspaceAgent initialized for personality '{self.personality}'.")

    def execute_plan(self, plan: list) -> list[tuple[bool, str, str]]:
        """
        Executes a list of planned actions sequentially.

        Args:
            plan: A list of action dictionaries, e.g.,
                  [{"action": "action_name", "args": {...}}, ...]

        Returns:
            A list of result tuples for each executed action:
            [(success: bool, message: str, action_suffix: str), ...]
            Execution stops after the first failure.
        """
        results = []
        if not isinstance(plan, list):
            logger.error(f"Invalid plan format received: Expected list, got {type(plan)}")
            results.append((False, "Internal Error: Invalid plan format received.", "plan_error"))
            return results

        logger.info(f"Executing workspace plan with {len(plan)} step(s)...")

        for i, step in enumerate(plan):
            action_result = None
            action_name = "unknown"
            try:
                if not isinstance(step, dict) or "action" not in step or "args" not in step:
                    logger.error(f"Invalid action step format in plan (Step {i+1}): {step}")
                    action_result = (False, f"Internal Error: Invalid action format in step {i+1}.", "step_error")
                else:
                    action_name = step.get("action", "unknown")
                    args = step.get("args", {})
                    logger.info(f"Executing Step {i+1}/{len(plan)}: Action='{action_name}', Args={args}")

                    # Dispatch to specific execution method
                    if action_name == "create_file":
                        action_result = self._execute_create_file(args)
                    elif action_name == "append_file":
                        action_result = self._execute_append_file(args)
                    elif action_name == "list_files":
                        action_result = self._execute_list_files(args)
                    elif action_name == "read_file":
                        action_result = self._execute_read_file(args)
                    elif action_name == "delete_file":
                        action_result = self._execute_delete_file(args)
                    elif action_name == "add_calendar_event":
                        action_result = self._execute_add_calendar_event(args)
                    elif action_name == "read_calendar":
                        action_result = self._execute_read_calendar(args)
                    else:
                        logger.error(f"Unsupported action '{action_name}' in plan (Step {i+1}).")
                        action_result = (False, f"Error: Action '{action_name}' is not supported.", f"{action_name}_unsupported")

            except Exception as e:
                logger.error(f"Unexpected exception executing plan step {i+1} (Action: {action_name}): {e}", exc_info=True)
                action_result = (False, f"Internal error during execution of '{action_name}': {e}", f"{action_name}_exception")

            # Append result and check for failure
            if action_result:
                results.append(action_result)
                if not action_result[0]: # Check success flag (index 0)
                    logger.warning(f"Plan execution stopped at step {i+1} due to failure.")
                    break # Stop execution on first failure
            else:
                # Should not happen if logic is correct, but handle defensively
                logger.error(f"Action result was None for step {i+1} (Action: {action_name}). Stopping plan.")
                results.append((False, f"Internal Error: No result returned for action '{action_name}'.", f"{action_name}_internal_error"))
                break

        logger.info(f"Workspace plan execution finished. {len(results)} step(s) attempted.")
        return results

    # --- Private Execution Helper Methods ---

    def _execute_create_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        content = args.get("content")
        action_name = "create_file"
        if filename and content is not None: # Allow empty string content
            # Note: We assume overwrite is intended here, as planning should handle existence checks if needed.
            # The file_manager function handles the actual creation/overwrite.
            success, message = file_manager.create_or_overwrite_file(self.config, self.personality, filename, str(content))
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'filename' or 'content' argument for create_file action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_append_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        content = args.get("content")
        action_name = "append_file"
        if filename and content is not None: # Allow empty string content
            success, message = file_manager.append_to_file(self.config, self.personality, filename, str(content))
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'filename' or 'content' argument for append_file action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_list_files(self, args: dict) -> tuple[bool, str, str]:
        action_name = "list_files"
        file_list, message = file_manager.list_files(self.config, self.personality)
        if file_list is not None:
            success = True
            # Format the message for the user if successful
            if file_list: message = f"Files in workspace:\n- " + "\n- ".join(file_list)
            else: message = "Workspace is empty."
        else:
            success = False
            # message already contains the error from file_manager
        suffix = f"{action_name}_{'success' if success else 'fail'}"
        return success, message, suffix

    def _execute_read_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        action_name = "read_file"
        if filename:
            file_content, message = file_manager.read_file(self.config, self.personality, filename)
            if file_content is not None:
                success = True
                # Return truncated content in the message for display
                content_preview = file_content[:500] + ('...' if len(file_content) > 500 else '')
                message = f"Content of '{filename}':\n---\n{content_preview}\n---"
                # Note: The full content isn't directly returned here, only the success message.
                # If the AI needs the content for subsequent steps, the *plan* needs to include reading it.
            else:
                success = False
                # message already contains the error from file_manager
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'filename' argument for read_file action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_delete_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        action_name = "delete_file"
        if filename:
            # Add confirmation step? No, assume plan is confirmed by AI generation.
            logger.warning(f"WorkspaceAgent executing delete_file for: {filename}")
            success, message = file_manager.delete_file(self.config, self.personality, filename)
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'filename' argument for delete_file action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_add_calendar_event(self, args: dict) -> tuple[bool, str, str]:
        date = args.get("date")
        time_str = args.get("time")
        desc = args.get("description")
        action_name = "add_calendar_event"
        if date and time_str and desc:
            success, message = file_manager.add_calendar_event(self.config, self.personality, date, time_str, desc)
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'date', 'time', or 'description' argument for add_calendar_event action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_read_calendar(self, args: dict) -> tuple[bool, str, str]:
        date = args.get("date") # Optional
        action_name = "read_calendar"
        events, message = file_manager.read_calendar_events(self.config, self.personality, date)
        # Success is true if the read operation didn't fail, message contains details/errors
        if "error" in message.lower() or "failed" in message.lower() or "denied" in message.lower():
             success = False
        else:
             success = True
             # Reformat message slightly for consistency if successful
             date_str = f" for {date}" if date else " (all dates)"
             if not events:
                 message = f"No calendar events found{date_str}."
             else:
                 event_lines = []
                 for e in sorted(events, key=lambda x: (x.get('event_date', ''), x.get('event_time', ''))):
                     event_lines.append(f"- {e.get('event_time', '?')}: {e.get('description', '?')} ({e.get('event_date', '?')})")
                 message = f"Found {len(events)} event(s){date_str}:\n" + "\n".join(event_lines)

        suffix = f"{action_name}_{'success' if success else 'fail'}"
        return success, message, suffix
