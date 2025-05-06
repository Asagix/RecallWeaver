import logging
import json
import re
import file_manager # Import the existing file manager
# Removed: from persistent_backend_graph import GraphMemoryClient

logger = logging.getLogger(__name__)

class WorkspaceAgent:
    """
    Handles the execution of a planned sequence of workspace actions.
    """
    def __init__(self, client_instance):
        """
        Initializes the WorkspaceAgent.

        Args:
            client_instance: An instance of GraphMemoryClient.
        """
        if client_instance is None:
             raise ValueError("WorkspaceAgent requires a valid GraphMemoryClient instance.")
        self.client = client_instance # Store the passed client instance
        self.config = self.client.config # Get config from the client instance
        self.personality = self.client.personality # Get personality from the client instance
        logger.info(f"WorkspaceAgent initialized for personality '{self.personality}' using provided client instance.")

    def execute_plan(self, plan: list) -> list[tuple[bool, str, str, bool]]: # Return 4-tuple list
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
            # Renamed action_result to indicate its potential length
            action_result_variable_tuple = None
            final_result_4_tuple = None # Variable to hold the final 4-tuple
            action_name = "unknown"
            args = {} # Initialize args here
            is_silent_request = False # Initialize silent flag
            try:
                if not isinstance(step, dict) or "action" not in step or "args" not in step:
                    logger.error(f"Invalid action step format in plan (Step {i+1}): {step}")
                    # Directly assign the 4-tuple error result
                    action_result_variable_tuple = (False, f"Internal Error: Invalid action format in step {i+1}.", "step_error", False)
                else:
                    action_name = step.get("action", "unknown")
                    args = step.get("args", {}) # Assign args here
                    is_silent_request = args.get("silent", False) is True # Check silent flag from args
                    logger.info(f"Executing Step {i+1}/{len(plan)}: Action='{action_name}', Args={args}, Silent={is_silent_request}")

                    # Dispatch to specific execution method (these return 3-tuples)
                    if action_name == "create_file":
                        action_result_variable_tuple = self._execute_create_file(args)
                    elif action_name == "append_file":
                        action_result_variable_tuple = self._execute_append_file(args)
                    elif action_name == "list_files":
                        action_result_variable_tuple = self._execute_list_files(args)
                    elif action_name == "read_file":
                        action_result_variable_tuple = self._execute_read_file(args)
                    elif action_name == "delete_file":
                        action_result_variable_tuple = self._execute_delete_file(args)
                    elif action_name == "add_calendar_event":
                        action_result_variable_tuple = self._execute_add_calendar_event(args)
                    elif action_name == "read_calendar":
                        action_result_variable_tuple = self._execute_read_calendar(args)
                    elif action_name == "consolidate_files":
                        action_result_variable_tuple = self._execute_consolidate_files(args)
                    else:
                        logger.error(f"Unsupported action '{action_name}' in plan (Step {i+1}).")
                        # Directly assign the 4-tuple error result
                        action_result_variable_tuple = (False, f"Error: Action '{action_name}' is not supported.", f"{action_name}_unsupported", False)

            except Exception as e:
                logger.error(f"Unexpected exception executing plan step {i+1} (Action: {action_name}): {e}", exc_info=True)
                # Directly assign the 4-tuple error result
                action_result_variable_tuple = (False, f"Internal error during execution of '{action_name}': {e}", f"{action_name}_exception", False)

            # --- Process the result (could be 3-tuple or 4-tuple) ---
            if action_result_variable_tuple is not None:
                if isinstance(action_result, tuple) and len(action_result) == 3:
                    success, message, action_suffix = action_result
                    # Determine the 4th element based on helper success and request
                    silent_and_successful = success and is_silent_request
                    # Construct the final 4-tuple
                    final_result_tuple = (success, message, action_suffix, silent_and_successful)
                    results.append(final_result_tuple)
                    # Check success flag (index 0) from the original 3-tuple result
                    if not success:
                        logger.warning(f"Plan execution stopped at step {i+1} due to failure.")
                        break # Stop execution on first failure
                else:
                    # Handle unexpected return type from helper defensively
                    logger.error(f"Helper for action '{action_name}' returned unexpected value: {action_result}. Expected 3-tuple.")
                    # Append an error tuple (as 4-tuple)
                    results.append((False, f"Internal Error: Invalid return from helper for '{action_name}'.", f"{action_name}_helper_error", False))
                    break # Stop execution
                # --- END CORRECTED BLOCK ---
            else:
                # Should not happen if logic is correct, but handle defensively
                logger.error(f"Action result was None for step {i+1} (Action: {action_name}). Stopping plan.")
                # Return 4-tuple for consistency
                results.append((False, f"Internal Error: No result returned for action '{action_name}'.", f"{action_name}_internal_error", False))
                break

            # --- Append the final 4-tuple result ---
            if final_result_4_tuple is not None:
                 # >>> FINAL CHECK <<<
                 if not isinstance(final_result_4_tuple, tuple) or len(final_result_4_tuple) != 4:
                      logger.critical(f"CRITICAL ERROR before append: final_result_4_tuple is not a 4-tuple! Type={type(final_result_4_tuple)}, Len={len(final_result_4_tuple) if isinstance(final_result_4_tuple, tuple) else 'N/A'}, Value={final_result_4_tuple}")
                      results.append((False, "Internal Agent Error: Invalid final tuple generated.", "internal_final_tuple_error", False))
                      break # Stop processing plan
                 else:
                      results.append(final_result_4_tuple)
                      # Check success flag (index 0) to stop on failure
                      if not final_result_4_tuple[0]:
                           logger.warning(f"Plan execution stopped at step {i+1} due to failure.")
                           break
            else:
                 # This case indicates a logic error where no result tuple was generated
                 logger.error(f"Result tuple (final_result_4_tuple) was None for step {i+1} (Action: {action_name}). Stopping plan.")
                 results.append((False, f"Internal Error: No result tuple generated for action '{action_name}'.", f"{action_name}_internal_none_error", False))
                 break

        logger.info(f"Workspace plan execution finished. {len(results)} step(s) attempted.")
        silent_success_count = sum(1 for r in results if len(r) == 4 and r[3]) # Ensure check against 4-tuple
        if silent_success_count > 0:
            logger.info(f"  {silent_success_count} action(s) were executed silently and successfully.")
        return results

    # --- Private Execution Helper Methods (return 3-tuple: success, message, suffix) ---

    def _execute_create_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        content = args.get("content")
        action_name = "create_file"
        if filename and content is not None: # Allow empty string content
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
            if file_list: message = f"Files in workspace:\n- " + "\n- ".join(file_list)
            else: message = "Workspace is empty."
        else:
            success = False
        suffix = f"{action_name}_{'success' if success else 'fail'}"
        return success, message, suffix

    def _execute_consolidate_files(self, args: dict) -> tuple[bool, str, str]:
        input_filenames = args.get("input_filenames")
        output_filename = args.get("output_filename")
        action_name = "consolidate_files"
        if not isinstance(input_filenames, list) or not input_filenames:
            msg = "Error: 'input_filenames' must be a non-empty list."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_missing"
        if not output_filename or not isinstance(output_filename, str):
            msg = "Error: Missing or invalid 'output_filename'."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_missing"
        if not all(isinstance(fname, str) for fname in input_filenames):
            msg = "Error: All filenames in 'input_filenames' must be strings."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_invalid"
        if output_filename in input_filenames:
            msg = "Error: Output filename cannot be one of the input filenames."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_conflict"

        logger.info(f"Executing consolidate_files: Inputs={input_filenames}, Output='{output_filename}'")
        combined_content, read_errors, successful_reads = "", [], []
        for fname in input_filenames:
            content, read_msg = file_manager.read_file(self.config, self.personality, fname)
            if content is not None:
                combined_content += f"\n\n--- Content from: {fname} ---\n{content}"
                successful_reads.append(fname)
            else:
                logger.error(f"Failed read '{fname}': {read_msg}"); read_errors.append(f"'{fname}': {read_msg}")
        if not successful_reads:
            return False, "Error: Failed to read any input files.", f"{action_name}_read_fail"
        if read_errors: logger.warning("Consolidating partial content due to read errors: " + "; ".join(read_errors))

        consolidated_content = None
        if not self.client: return False, "Internal Error: Backend client missing.", f"{action_name}_internal_error"
        try:
            prompt_template = self.client._load_prompt("consolidate_files_content_prompt.txt")
            if not prompt_template: raise ValueError("Consolidation prompt template missing.")
            consolidation_prompt = prompt_template.format(combined_input_content=combined_content.strip())
            llm_response = self.client._call_configured_llm('workspace_file_consolidation', prompt=consolidation_prompt)
            if llm_response and not llm_response.startswith("Error:"): consolidated_content = llm_response.strip()
            else: raise ValueError(f"LLM failed: {llm_response}")
        except Exception as e: msg = f"Error during content generation: {e}"; logger.error(msg, exc_info=True); return False, msg, f"{action_name}_llm_fail"

        write_success, write_message = file_manager.create_or_overwrite_file(self.config, self.personality, output_filename, consolidated_content)
        if not write_success: return False, f"Error writing consolidated file: {write_message}", f"{action_name}_write_fail"

        delete_errors, deleted_count = [], 0
        for fname in successful_reads:
            del_success, del_msg = file_manager.delete_file(self.config, self.personality, fname)
            if not del_success: logger.error(f"Failed delete '{fname}': {del_msg}"); delete_errors.append(f"'{fname}': {del_msg}")
            else: deleted_count += 1

        final_message = f"Consolidated {len(successful_reads)} file(s) into '{output_filename}'."
        if read_errors: final_message += f" (Read Failures: {len(read_errors)})"
        if delete_errors: final_message += f" (Delete Failures: {len(delete_errors)})"
        elif deleted_count == len(successful_reads): final_message += f" Originals deleted."
        return True, final_message, f"{action_name}_success"

    def _execute_read_file(self, args: dict) -> tuple[bool, str, str]:
        filename = args.get("filename")
        action_name = "read_file"
        if filename:
            file_content, message = file_manager.read_file(self.config, self.personality, filename)
            if file_content is not None:
                success = True
                content_preview = file_content[:500] + ('...' if len(file_content) > 500 else '')
                message = f"Content of '{filename}':\n---\n{content_preview}\n---"
            else:
                success = False
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
            logger.warning(f"WorkspaceAgent executing delete_file for: {filename}")
            success, message = file_manager.delete_file(self.config, self.personality, filename)
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'filename' argument for delete_file action."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_add_calendar_event(self, args: dict) -> tuple[bool, str, str]:
        date, time_str, desc = args.get("date"), args.get("time"), args.get("description")
        action_name = "add_calendar_event"
        if date and time_str and desc:
            success, message = file_manager.add_calendar_event(self.config, self.personality, date, time_str, desc)
            suffix = f"{action_name}_{'success' if success else 'fail'}"
            return success, message, suffix
        else:
            message = "Error: Missing 'date', 'time', or 'description'."
            logger.error(message + f" Args: {args}")
            return False, message, f"{action_name}_arg_missing"

    def _execute_read_calendar(self, args: dict) -> tuple[bool, str, str]:
        date = args.get("date") # Optional
        action_name = "read_calendar"
        events, message = file_manager.read_calendar_events(self.config, self.personality, date)
        success = not ("error" in message.lower() or "failed" in message.lower() or "denied" in message.lower())
        if success:
             date_str = f" for {date}" if date else " (all dates)"
             if not events: message = f"No calendar events found{date_str}."
             else:
                 event_lines = [f"- {e.get('event_time', '?')}: {e.get('description', '?')} ({e.get('event_date', '?')})"
                                for e in sorted(events, key=lambda x: (x.get('event_date', ''), x.get('event_time', '')))]
                 message = f"Found {len(events)} event(s){date_str}:\n" + "\n".join(event_lines)
        suffix = f"{action_name}_{'success' if success else 'fail'}"
        return success, message, suffix
    
    
    """
    Handles the execution of a planned sequence of workspace actions.
    """
    def __init__(self, client_instance):
        """
        Initializes the WorkspaceAgent.

        Args:
            client_instance: An instance of GraphMemoryClient.
        """
        if client_instance is None:
             raise ValueError("WorkspaceAgent requires a valid GraphMemoryClient instance.")
        self.client = client_instance # Store the passed client instance
        self.config = self.client.config # Get config from the client instance
        self.personality = self.client.personality # Get personality from the client instance
        logger.info(f"WorkspaceAgent initialized for personality '{self.personality}' using provided client instance.")

    def execute_plan(self, plan: list) -> list[tuple[bool, str, str, bool]]: # Return 4-tuple list
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
            # Renamed variable to avoid confusion with the final 4-tuple
            helper_result_tuple = None
            final_result_4_tuple = None # Variable to hold the final 4-tuple
            action_name = "unknown"
            args = {}
            is_silent_request = False

            try:
                if not isinstance(step, dict) or "action" not in step or "args" not in step:
                    logger.error(f"Invalid action step format in plan (Step {i+1}): {step}")
                    # Directly assign the 4-tuple error result
                    final_result_4_tuple = (False, f"Internal Error: Invalid action format in step {i+1}.", "step_error", False)
                else:
                    action_name = step.get("action", "unknown")
                    args = step.get("args", {})
                    is_silent_request = args.get("silent", False) is True
                    logger.info(f"Executing Step {i+1}/{len(plan)}: Action='{action_name}', Args={args}, Silent={is_silent_request}")

                    # Dispatch to specific execution method (these return 3-tuples)
                    if action_name == "create_file":
                        helper_result_tuple = self._execute_create_file(args)
                    elif action_name == "append_file":
                        helper_result_tuple = self._execute_append_file(args)
                    elif action_name == "list_files":
                        helper_result_tuple = self._execute_list_files(args)
                    elif action_name == "read_file":
                        helper_result_tuple = self._execute_read_file(args)
                    elif action_name == "delete_file":
                        helper_result_tuple = self._execute_delete_file(args)
                    elif action_name == "add_calendar_event":
                        helper_result_tuple = self._execute_add_calendar_event(args)
                    elif action_name == "read_calendar":
                        helper_result_tuple = self._execute_read_calendar(args)
                    elif action_name == "consolidate_files":
                        helper_result_tuple = self._execute_consolidate_files(args)
                    else:
                        logger.error(f"Unsupported action '{action_name}' in plan (Step {i+1}).")
                        # Directly assign the 4-tuple error result
                        final_result_4_tuple = (False, f"Error: Action '{action_name}' is not supported.", f"{action_name}_unsupported", False)

                    # --- Process successful helper result (if no error occurred above) ---
                    if helper_result_tuple is not None:
                        # <<< DEBUG POINT >>>
                        logger.debug(f"[DEBUG] Step {i+1} Success: Helper returned helper_result_tuple: Type={type(helper_result_tuple)}, Value={helper_result_tuple}")
                        if isinstance(helper_result_tuple, tuple) and len(helper_result_tuple) == 3:
                            # --- CORRECT UNPACKING ---
                            success, message, action_suffix = helper_result_tuple
                            # -------------------------
                            silent_and_successful = success and is_silent_request
                            final_result_4_tuple = (success, message, action_suffix, silent_and_successful)
                            logger.debug(f"[DEBUG] Step {i+1} Success: Constructed 4-tuple: {final_result_4_tuple}")
                        else:
                            logger.error(f"!!! Helper for '{action_name}' returned unexpected value: {helper_result_tuple}. Expected 3-tuple. !!!")
                            final_result_4_tuple = (False, f"Internal Error: Helper for '{action_name}' returned invalid data.", f"{action_name}_helper_error", False)

            except Exception as e:
                logger.error(f"Unexpected exception executing plan step {i+1} (Action: {action_name}): {e}", exc_info=True)
                # Directly assign the 4-tuple error result
                final_result_4_tuple = (False, f"Internal error during execution of '{action_name}': {e}", f"{action_name}_exception", False)

            # --- Append the final 4-tuple result ---
            if final_result_4_tuple is not None:
                if not isinstance(final_result_4_tuple, tuple) or len(final_result_4_tuple) != 4:
                    logger.critical(f"CRITICAL ERROR before append: final_result_4_tuple is not a 4-tuple! Type={type(final_result_4_tuple)}, Len={len(final_result_4_tuple) if isinstance(final_result_4_tuple, tuple) else 'N/A'}, Value={final_result_4_tuple}")
                    results.append((False, "Internal Agent Error: Invalid final tuple generated.", "internal_final_tuple_error", False))
                    break
                else:
                    results.append(final_result_4_tuple)
                    # Check success flag (index 0) to stop on failure
                    if not final_result_4_tuple[0]:
                        logger.warning(f"Plan execution stopped at step {i+1} due to failure.")
                        break
            else:
                # This indicates a logic error where no result tuple was assigned
                logger.error(f"Result tuple (final_result_4_tuple) was None for step {i+1} (Action: {action_name}). Stopping plan.")
                results.append((False, f"Internal Error: No result tuple generated for action '{action_name}'.", f"{action_name}_internal_none_error", False))
                break

        logger.info(f"Workspace plan execution finished. {len(results)} step(s) attempted.")
        silent_success_count = sum(1 for r in results if len(r) == 4 and r[3])
        if silent_success_count > 0:
            logger.info(f"  {silent_success_count} action(s) were executed silently and successfully.")
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

    def _execute_consolidate_files(self, args: dict) -> tuple[bool, str, str]:
        """Consolidates multiple input files into a single output file."""
        input_filenames = args.get("input_filenames")
        output_filename = args.get("output_filename")
        action_name = "consolidate_files"

        # --- Argument Validation ---
        if not isinstance(input_filenames, list) or not input_filenames:
            msg = "Error: 'input_filenames' must be a non-empty list for consolidate_files."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_missing"
        if not output_filename or not isinstance(output_filename, str):
            msg = "Error: Missing or invalid 'output_filename' argument for consolidate_files."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_missing"
        if not all(isinstance(fname, str) for fname in input_filenames):
            msg = "Error: All filenames in 'input_filenames' must be strings."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_invalid"
        if output_filename in input_filenames:
            msg = "Error: Output filename cannot be one of the input filenames."
            logger.error(msg + f" Args: {args}")
            return False, msg, f"{action_name}_arg_conflict"

        logger.info(f"Executing consolidate_files: Inputs={input_filenames}, Output='{output_filename}'")

        # --- Read Input Files ---
        combined_content = ""
        read_errors = []
        successful_reads = []
        for fname in input_filenames:
            content, read_msg = file_manager.read_file(self.config, self.personality, fname)
            if content is not None:
                combined_content += f"\n\n--- Content from: {fname} ---\n{content}"
                successful_reads.append(fname)
            else:
                logger.error(f"Failed to read input file '{fname}' for consolidation: {read_msg}")
                read_errors.append(f"Could not read '{fname}': {read_msg}")

        if not successful_reads:
            msg = "Error: Failed to read any of the input files for consolidation. " + " ".join(read_errors)
            return False, msg, f"{action_name}_read_fail"
        if read_errors:
            # Proceed with consolidation but warn about missing files
            logger.warning("Consolidating partial content due to read errors: " + "; ".join(read_errors))

        # --- Generate Consolidated Content via LLM ---
        consolidated_content = None
        if not self.client:
             # Handle case where internal client failed to initialize
             msg = "Internal Error: Cannot generate consolidated content (Backend client missing)."
             logger.error(msg)
             return False, msg, f"{action_name}_internal_error"

        try:
            prompt_template = self.client._load_prompt("consolidate_files_content_prompt.txt")
            if not prompt_template:
                raise ValueError("Consolidation content prompt template missing.")

            consolidation_prompt = prompt_template.format(combined_input_content=combined_content.strip())
            logger.debug(f"Sending file consolidation content prompt (Input length: {len(combined_content)} chars)...")

            # Use the client's configured LLM call method
            llm_response = self.client._call_configured_llm('workspace_file_consolidation', prompt=consolidation_prompt)

            if llm_response and not llm_response.startswith("Error:"):
                consolidated_content = llm_response.strip()
                logger.info(f"LLM generated consolidated content (Length: {len(consolidated_content)} chars).")
            else:
                raise ValueError(f"LLM failed to generate consolidated content: {llm_response}")

        except Exception as e:
            msg = f"Error during consolidated content generation: {e}"
            logger.error(msg, exc_info=True)
            return False, msg, f"{action_name}_llm_fail"

        # --- Write Output File ---
        write_success, write_message = file_manager.create_or_overwrite_file(
            self.config, self.personality, output_filename, consolidated_content
        )
        if not write_success:
            msg = f"Error writing consolidated file '{output_filename}': {write_message}"
            logger.error(msg)
            # Don't delete inputs if output failed
            return False, msg, f"{action_name}_write_fail"

        logger.info(f"Successfully wrote consolidated file: '{output_filename}'")

        # --- Delete Input Files ---
        delete_errors = []
        deleted_count = 0
        for fname in successful_reads: # Only delete files that were successfully read
            del_success, del_msg = file_manager.delete_file(self.config, self.personality, fname)
            if not del_success:
                logger.error(f"Failed to delete input file '{fname}' after consolidation: {del_msg}")
                delete_errors.append(f"Could not delete '{fname}': {del_msg}")
            else:
                deleted_count += 1

        # --- Final Message ---
        final_message = f"Successfully consolidated {len(successful_reads)} file(s) into '{output_filename}'."
        if read_errors:
            final_message += f" (Warning: Failed to read {len(read_errors)} file(s): {'; '.join(read_errors)})"
        if delete_errors:
            final_message += f" (Warning: Failed to delete {len(delete_errors)} original file(s): {'; '.join(delete_errors)})"
        elif deleted_count == len(successful_reads):
             final_message += f" Original {deleted_count} file(s) deleted."


        return True, final_message, f"{action_name}_success"

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
