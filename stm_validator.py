# src/stm_validator.py

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Set, Optional

def validate_stm_xml(xml_string: str, final_state_keywords: Optional[List[str]] = None) -> Tuple[bool, List[str], List[str]]:
    """
    Performs comprehensive structural and logical validation of an STM XML string.

    Args:
        xml_string: The XML content of the State Transition Machine.
        final_state_keywords: A list of keywords (e.g., 'Closed', 'End') that indicate a final state.
                              States with these keywords in their ID will not be flagged as dead-ends.

    Returns:
        A tuple: (is_valid, errors, warnings)
        is_valid (bool): True if no critical errors are found.
        errors (List[str]): A list of error messages found.
        warnings (List[str]): A list of warning messages found.
    """
    errors: List[str] = []
    warnings: List[str] = []
    is_valid = True

    if final_state_keywords is None:
        # Default keywords for states that are considered final/closed
        final_state_keywords = ["Closed", "End", "Final", "Complete", "Resolved", "Failed", "Rejected", "Terminated", "Stop"]

    if not xml_string.strip():
        errors.append("Input XML string is empty or contains only whitespace.")
        return False, errors, warnings

    try:
        # 1. Basic XML parsing (checks for well-formedness)
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        errors.append(f"XML parsing error (not well-formed): {e}. Please ensure XML syntax is correct.")
        return False, errors, warnings

    # 2. Check root element
    if root.tag != "workflow":
        errors.append(f"Root element must be '<workflow>', but found '<{root.tag}>'.")
        is_valid = False

    # Data structures for comprehensive checks
    all_state_ids: Set[str] = set()
    initial_state_id: Optional[str] = None
    states_with_outgoing_transitions: Dict[str, List[str]] = {} # {source_state_id: [target_state_id1, ...]}
    states_with_incoming_transitions: Set[str] = set() # Set of all states that are targets of any transition
    transitions_defined_count = 0

    # First pass: Collect state info and basic validation
    for state_elem in root.findall("state"):
        state_id = state_elem.get("id")

        if not state_id:
            errors.append(f"Found a <state> element missing its 'id' attribute.")
            is_valid = False
            continue # Cannot process this state further without an ID

        if state_id in all_state_ids:
            errors.append(f"Duplicate state ID found: '{state_id}'. State IDs must be unique.")
            is_valid = False
        all_state_ids.add(state_id)

        if state_elem.get("initial") == "true":
            if initial_state_id is not None:
                warnings.append("Multiple states marked as 'initial'. An STM typically has only one initial state.")
            initial_state_id = state_id

        current_state_outgoing_targets: List[str] = []
        for transition_elem in state_elem.findall("transition"):
            transitions_defined_count += 1
            event = transition_elem.get("event")
            target = transition_elem.get("target")

            if not event:
                errors.append(f"Transition in state '{state_id}' is missing 'event' attribute.")
                is_valid = False
            if not target:
                errors.append(f"Transition in state '{state_id}' is missing 'target' attribute.")
                is_valid = False
            else:
                current_state_outgoing_targets.append(target)
                states_with_incoming_transitions.add(target)
        states_with_outgoing_transitions[state_id] = current_state_outgoing_targets

    # If no states are found at all
    if not all_state_ids:
        errors.append("No <state> elements found in the workflow. A workflow must define at least one state.")
        is_valid = False

    # Second pass: Perform cross-referencing and logical validations
    if initial_state_id is None and len(all_state_ids) > 0:
        warnings.append("No 'initial' state found. A workflow typically starts from a defined initial state. The entry point is ambiguous.")

    if transitions_defined_count == 0 and len(all_state_ids) > 1:
        warnings.append("No transitions found. A workflow with multiple states should define transitions between them.")

    # 3. Validate target states (check if all targets exist as defined states)
    for state_id, targets in states_with_outgoing_transitions.items():
        for target_id in targets:
            if target_id not in all_state_ids:
                errors.append(f"Transition from state '{state_id}' targets undefined state: '{target_id}'. All target states must be defined.")
                is_valid = False

    # 4. Check for unreachable states (excluding the initial state)
    if initial_state_id: # Only check if an initial state is explicitly defined
        for state_id in all_state_ids:
            if state_id == initial_state_id:
                continue # Initial state is always reachable by definition
            if state_id not in states_with_incoming_transitions:
                warnings.append(f"State '{state_id}' is unreachable. No transition leads to it.")
    elif len(all_state_ids) > 0 and transitions_defined_count > 0:
        # If no initial state, but there are states and transitions, all states should ideally have incoming transitions
        # unless it's a "start" implicitly. This is a weaker check than with an explicit initial state.
        pass # Will rely on the "No initial state found" warning.

    # 5. Check for dead-end states (states with no outgoing transitions that are not considered final)
    for state_id in all_state_ids:
        # Check if the state has no outgoing transitions
        if state_id not in states_with_outgoing_transitions or not states_with_outgoing_transitions[state_id]:
            # Check if it's explicitly named as a final state (case-insensitive check)
            is_final_state_by_name = any(keyword.lower() in state_id.lower() for keyword in final_state_keywords)
            if not is_final_state_by_name:
                warnings.append(f"State '{state_id}' is a potential dead-end. It has no outgoing transitions and its ID does not suggest it is a final/closed state.")

    return is_valid, errors, warnings

# Example Usage (for testing the module directly) - unchanged
if __name__ == "__main__":
    print("--- Testing STM Validator ---")

    valid_xml = """
    <workflow id="TestWorkflow">
        <state id="Start" initial="true">
            <transition event="init" target="Active"/>
        </state>
        <state id="Active">
            <transition event="complete" target="End"/>
            <transition event="fail" target="Error"/>
        </state>
        <state id="End"/>
        <state id="Error"/>
    </workflow>
    """
    print("\nValid XML (Should be valid, one warning for implicit 'End'/'Error' being final):")
    is_valid, errors, warnings = validate_stm_xml(valid_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    invalid_root_xml = """
    <process id="InvalidRoot">
        <state id="Start"/>
    </process>
    """
    print("\nInvalid Root XML (Should be invalid):")
    is_valid, errors, warnings = validate_stm_xml(invalid_root_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    missing_target_xml = """
    <workflow id="MissingTarget">
        <state id="Start" initial="true">
            <transition event="next" target="NonExistentState"/>
        </state>
        <state id="ActualEnd"/>
    </workflow>
    """
    print("\nMissing Target XML (Should be invalid, unreachable warning):")
    is_valid, errors, warnings = validate_stm_xml(missing_target_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    duplicate_state_xml = """
    <workflow id="DuplicateState">
        <state id="Start" initial="true"/>
        <state id="Start"/>
        <state id="End"/>
    </workflow>
    """
    print("\nDuplicate State XML (Should be invalid, unreachable warning):")
    is_valid, errors, warnings = validate_stm_xml(duplicate_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    empty_xml = ""
    print("\nEmpty XML (Should be invalid):")
    is_valid, errors, warnings = validate_stm_xml(empty_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    unreachable_state_xml = """
    <workflow id="Unreachable">
        <state id="Start" initial="true">
            <transition event="go" target="A"/>
        </state>
        <state id="A">
            <transition event="done" target="End"/>
        </state>
        <state id="B"/> <!-- Unreachable -->
        <state id="End"/>
    </workflow>
    """
    print("\nUnreachable State XML (Should be valid, unreachable warning for B):")
    is_valid, errors, warnings = validate_stm_xml(unreachable_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    dead_end_xml = """
    <workflow id="DeadEnd">
        <state id="Start" initial="true">
            <transition event="go" target="Middle"/>
        </state>
        <state id="Middle"/> <!-- Dead-end, not a final state keyword -->
        <state id="FinalState"/>
    </workflow>
    """
    print("\nDead-End State XML (Should be valid, dead-end warning for Middle):")
    is_valid, errors, warnings = validate_stm_xml(dead_end_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    single_state_xml = """
    <workflow id="SingleState">
        <state id="OnlyState" initial="true"/>
    </workflow>
    """
    print("\nSingle State XML (Should be valid, no transitions warning if only one state):")
    is_valid, errors, warnings = validate_stm_xml(single_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
