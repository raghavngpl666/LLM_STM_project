# src/stm_validator.py

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

def validate_stm_xml(xml_string: str) -> Tuple[bool, List[str], List[str]]:
    """
    Performs basic structural and logical validation of an STM XML string.

    Args:
        xml_string: The XML content of the State Transition Machine.

    Returns:
        A tuple: (is_valid, errors, warnings)
        is_valid (bool): True if no critical errors are found.
        errors (List[str]): A list of error messages found.
        warnings (List[str]): A list of warning messages found.
    """
    errors: List[str] = []
    warnings: List[str] = []
    is_valid = True

    if not xml_string.strip():
        errors.append("Input XML string is empty.")
        return False, errors, warnings

    try:
        # 1. Basic XML parsing (well-formedness check)
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        errors.append(f"XML parsing error (not well-formed): {e}")
        return False, errors, warnings

    # 2. Check root element
    if root.tag != "workflow":
        errors.append(f"Root element must be '<workflow>', but found '<{root.tag}>'.")
        is_valid = False

    # Store states and their transitions for cross-referencing
    states: Dict[str, ET.Element] = {}
    target_states: List[str] = []
    initial_state_found = False
    transitions_exist = False

    # 3. Validate states and gather information
    for state_elem in root.findall("state"):
        state_id = state_elem.get("id")
        if not state_id:
            errors.append(f"State element missing 'id' attribute.")
            is_valid = False
            continue

        if state_id in states:
            errors.append(f"Duplicate state ID found: '{state_id}'. State IDs must be unique.")
            is_valid = False
        states[state_id] = state_elem

        # Check for initial state (a simple convention for now, can be expanded)
        if state_elem.get("initial") == "true":
            if initial_state_found:
                warnings.append("Multiple initial states found. An STM typically has only one initial state.")
            initial_state_found = True

        # 4. Validate transitions within each state
        for transition_elem in state_elem.findall("transition"):
            transitions_exist = True
            event = transition_elem.get("event")
            target = transition_elem.get("target")

            if not event:
                errors.append(f"Transition in state '{state_id}' is missing 'event' attribute.")
                is_valid = False
            if not target:
                errors.append(f"Transition in state '{state_id}' is missing 'target' attribute.")
                is_valid = False
            else:
                target_states.append(target) # Collect all target states

    if not initial_state_found:
        warnings.append("No 'initial' state found. A workflow usually starts from a defined initial state.")

    if not transitions_exist and len(states) > 1:
        warnings.append("No transitions found. A workflow with multiple states should have transitions.")


    # 5. Validate target states (check if all targets exist as defined states)
    for target_id in set(target_states): # Use set to avoid redundant checks
        if target_id not in states:
            errors.append(f"Transition targets undefined state: '{target_id}'. All target states must be defined.")
            is_valid = False

    # Additional potential future checks (just as comments for now)
    # - Check for unreachable states (states with no incoming transitions, unless it's initial)
    # - Check for dead-end states (states with no outgoing transitions, unless it's a final/closed state)
    # - Check for valid event names / conditions (requires defined schema/grammar)

    return is_valid, errors, warnings

# Example Usage (for testing the module directly)
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
    print("\nValid XML:")
    is_valid, errors, warnings = validate_stm_xml(valid_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    invalid_root_xml = """
    <process id="InvalidRoot">
        <state id="Start"/>
    </process>
    """
    print("\nInvalid Root XML:")
    is_valid, errors, warnings = validate_stm_xml(invalid_root_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")


    missing_target_xml = """
    <workflow id="MissingTarget">
        <state id="Start" initial="true">
            <transition event="next" target="NonExistentState"/>
        </state>
        <state id="End"/>
    </workflow>
    """
    print("\nMissing Target XML:")
    is_valid, errors, warnings = validate_stm_xml(missing_target_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    duplicate_state_xml = """
    <workflow id="DuplicateState">
        <state id="Start" initial="true"/>
        <state id="Start"/>
    </workflow>
    """
    print("\nDuplicate State XML:")
    is_valid, errors, warnings = validate_stm_xml(duplicate_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    empty_xml = ""
    print("\nEmpty XML:")
    is_valid, errors, warnings = validate_stm_xml(empty_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")