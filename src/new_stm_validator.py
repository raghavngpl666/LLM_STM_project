# src/new_stm_validator.py

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Set, Optional

def validate_stm_xml(xml_string: str) -> Tuple[bool, List[str], List[str]]:
    """
    Performs structural and basic logical validation of an STM XML string
    adhering to the schema defined in the Deepseek-Coder Modelfile.

    Expected Schema:
    <states>
        <flow id="..." default="true">
            <manual-state id="..." initialState="true|false">
                <on eventId="..." newStateId="..." componentName="..." />
            </manual-state>
            <auto-state id="..." code="payload.someField">
                <on eventId="..." newStateId="..." />
            </auto-state>
            ...
        </flow>
    </states>

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
        errors.append("Input XML string is empty or contains only whitespace.")
        return False, errors, warnings

    try:
        # 1. Basic XML parsing (checks for well-formedness)
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        errors.append(f"XML parsing error (not well-formed): {e}. Please ensure XML syntax is correct.")
        return False, errors, warnings

    # Data structures for validation
    all_state_ids: Set[str] = set()
    initial_state_found = False
    transitions_from_state: Dict[str, List[str]] = {} # {source_state_id: [target_state_id1, ...]}
    transitions_to_state: Set[str] = set() # All states that are targets of any 'on' transition

    # 2. Validate root element: Must be <states>
    if root.tag != "states":
        errors.append(f"Root element must be '<states>', but found '<{root.tag}>'.")
        is_valid = False

    # 3. Iterate through <flow> elements (expected to be direct children of <states>)
    flow_elements = root.findall("flow")
    if not flow_elements:
        warnings.append("No '<flow>' element found directly under '<states>'. A workflow typically has at least one flow.")
    elif len(flow_elements) > 1:
        # Warning, as the schema example shows only one, but multiple are technically possible for complex STMs
        warnings.append("Multiple '<flow>' elements found. Ensure this is intended for your workflow's structure.")


    for flow_elem in flow_elements:
        flow_id = flow_elem.get("id")
        if not flow_id:
            warnings.append(f"Found a '<flow>' element without an 'id' attribute.") # Warning for now, could be error
        
        # Check for 'default' attribute on flow
        if flow_elem.get("default") == "true":
            # No specific validation, but note its presence
            pass

        # 4. Iterate through state elements (<manual-state> and <auto-state>)
        state_elements = flow_elem.findall("manual-state") + flow_elem.findall("auto-state")
        if not state_elements:
            errors.append(f"No '<manual-state>' or '<auto-state>' elements found within flow '{flow_id}'. A flow must define states.")
            is_valid = False
            continue

        for state_elem in state_elements:
            state_id = state_elem.get("id")
            if not state_id:
                errors.append(f"A '<{state_elem.tag}>' element within flow '{flow_id}' is missing its 'id' attribute.")
                is_valid = False
                continue

            if state_id in all_state_ids:
                errors.append(f"Duplicate state ID found: '{state_id}'. State IDs must be unique across all states.")
                is_valid = False
            all_state_ids.add(state_id)

            # Check for initial state
            if state_elem.get("initialState") == "true":
                if initial_state_found:
                    warnings.append(f"Multiple states marked as 'initialState=\"true\"'. An STM typically has only one initial state. (State ID: '{state_id}')")
                initial_state_found = True

            # Additional checks for auto-state
            if state_elem.tag == "auto-state":
                if not state_elem.get("code"):
                    errors.append(f"An '<auto-state>' with ID '{state_id}' is missing its 'code' attribute.")
                    is_valid = False

            # 5. Iterate through <on> transitions within states
            current_state_outgoing_targets: List[str] = []
            on_elements = state_elem.findall("on")
            if not on_elements:
                # This could be a dead-end, handled later. Not an error here.
                pass

            for on_elem in on_elements:
                event_id = on_elem.get("eventId")
                new_state_id = on_elem.get("newStateId")

                if not event_id:
                    errors.append(f"An '<on>' transition in state '{state_id}' is missing its 'eventId' attribute.")
                    is_valid = False
                if not new_state_id:
                    errors.append(f"An '<on>' transition in state '{state_id}' is missing its 'newStateId' attribute.")
                    is_valid = False
                else:
                    current_state_outgoing_targets.append(new_state_id)
                    transitions_to_state.add(new_state_id)
                
                # Check for componentName in manual-state's 'on' element
                if state_elem.tag == "manual-state":
                    if not on_elem.get("componentName"):
                        warnings.append(f"A '<manual-state>' (ID: '{state_id}') has an '<on>' transition (Event: '{event_id}') without a 'componentName' attribute. Modelfile rules state it's required.")


            transitions_from_state[state_id] = current_state_outgoing_targets

    # 6. Post-processing and logical checks across all states
    if not initial_state_found and all_state_ids:
        warnings.append("No state found with 'initialState=\"true\"'. A workflow typically starts from a defined initial state.")

    # 7. Validate target states: Ensure all 'newStateId' attributes refer to existing state IDs
    for source_state_id, targets in transitions_from_state.items():
        for target_id in targets:
            if target_id not in all_state_ids:
                errors.append(f"Transition from state '{source_state_id}' targets undefined state: '{target_id}'. All target states must be defined.")
                is_valid = False

    # 8. Check for unreachable states (excluding the initial state, if explicitly found)
    if initial_state_found:
        reachable_states = set()
        # Find the actual initial state ID
        for flow_elem in flow_elements:
            for state_elem in flow_elem.findall("manual-state") + flow_elem.findall("auto-state"):
                if state_elem.get("initialState") == "true":
                    reachable_states.add(state_elem.get("id"))
                    break
            if reachable_states: break # Found initial, no need to search further

        # Use a simple BFS/DFS to find all reachable states
        q = list(reachable_states)
        while q:
            current_state = q.pop(0)
            for next_state in transitions_from_state.get(current_state, []):
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    q.append(next_state)
        
        unreachable_states = all_state_ids - reachable_states
        for state_id in unreachable_states:
            warnings.append(f"State '{state_id}' is unreachable. No transition leads to it from the initial state.")
    else:
        # If no initial state, can't determine reachability properly from a single start point
        warnings.append("Cannot fully check for unreachable states because no 'initialState=\"true\"' was found.")


    # 9. Check for dead-end states (states with no outgoing transitions that are not clearly terminal)
    # For now, we don't have a clear "final" state definition in the Modelfile, so we'll warn on any state without outgoing transitions.
    for state_id in all_state_ids:
        if state_id not in transitions_from_state or not transitions_from_state[state_id]:
            # This is a warning because sometimes a state is *meant* to be a terminal state without explicit outgoing transitions.
            # If the Modelfile had a rule for 'finalState="true"', we'd check against that.
            warnings.append(f"State '{state_id}' is a potential dead-end. It has no outgoing transitions. Ensure this is intended.")

    # Determine overall validity
    if errors:
        is_valid = False

    return is_valid, errors, warnings

# --- Example Usage (for testing the module directly) ---
if __name__ == "__main__":
    print("--- Testing STM Validator (Ollama Schema) ---")

    # Example 1: Valid basic workflow
    valid_xml = """
    <states>
        <flow id="mainFlow" default="true">
            <manual-state id="Start" initialState="true">
                <on eventId="init" newStateId="Active" componentName="ProcessHandler"/>
            </manual-state>
            <auto-state id="Active" code="payload.data">
                <on eventId="complete" newStateId="End"/>
                <on eventId="fail" newStateId="Error"/>
            </auto-state>
            <manual-state id="End"/>
            <manual-state id="Error"/>
        </flow>
    </states>
    """
    print("\nExample 1: Valid Basic Workflow (should be valid, no errors/warnings if componentName is seen as optional for 'on')")
    is_valid, errors, warnings = validate_stm_xml(valid_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 2: Missing initial state
    missing_initial_xml = """
    <states>
        <flow id="mainFlow">
            <manual-state id="Start">
                <on eventId="init" newStateId="Active" componentName="ProcessHandler"/>
            </manual-state>
            <manual-state id="Active"/>
        </flow>
    </states>
    """
    print("\nExample 2: Missing Initial State (should warn)")
    is_valid, errors, warnings = validate_stm_xml(missing_initial_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 3: Undefined target state
    undefined_target_xml = """
    <states>
        <flow id="mainFlow">
            <manual-state id="Start" initialState="true">
                <on eventId="init" newStateId="NonExistent" componentName="Handler"/>
            </manual-state>
        </flow>
    </states>
    """
    print("\nExample 3: Undefined Target State (should error)")
    is_valid, errors, warnings = validate_stm_xml(undefined_target_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 4: Unreachable state
    unreachable_state_xml = """
    <states>
        <flow id="mainFlow">
            <manual-state id="Start" initialState="true">
                <on eventId="go" newStateId="A" componentName="Comp"/>
            </manual-state>
            <manual-state id="A"/>
            <manual-state id="B"/> <!-- Unreachable -->
        </flow>
    </states>
    """
    print("\nExample 4: Unreachable State (should warn for 'B')")
    is_valid, errors, warnings = validate_stm_xml(unreachable_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 5: Dead-end state
    dead_end_state_xml = """
    <states>
        <flow id="mainFlow">
            <manual-state id="Start" initialState="true">
                <on eventId="start" newStateId="Middle" componentName="Comp"/>
            </manual-state>
            <manual-state id="Middle"/> <!-- No outgoing transitions -->
        </flow>
    </states>
    """
    print("\nExample 5: Dead-end State (should warn for 'Middle')")
    is_valid, errors, warnings = validate_stm_xml(dead_end_state_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 6: Invalid root
    invalid_root_xml = """
    <workflow>
        <flow id="mainFlow"/>
    </workflow>
    """
    print("\nExample 6: Invalid Root Element (should error)")
    is_valid, errors, warnings = validate_stm_xml(invalid_root_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Example 7: Missing 'code' attribute for auto-state
    missing_code_xml = """
    <states>
        <flow id="mainFlow">
            <auto-state id="Process" initialState="true">
                <on eventId="trigger" newStateId="End"/>
            </auto-state>
            <manual-state id="End"/>
        </flow>
    </states>
    """
    print("\nExample 7: Missing 'code' for auto-state (should error)")
    is_valid, errors, warnings = validate_stm_xml(missing_code_xml)
    print(f"Is Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
