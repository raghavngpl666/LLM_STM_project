# src/hello_llm.py

import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List, Optional, Tuple

# Import the STM validator module
from stm_validator import validate_stm_xml

# Helper function to generate LLM response
def _generate_llm_response(
    model, tokenizer, messages: List[dict], device: str, max_new_tokens: int = 700
) -> str:
    """
    Generates a response from the LLM based on the current messages history.
    """
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    print("\n[DEBUG] --- LLM Generation Call Initiated ---")
    try:
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_response.strip()
    except Exception as e:
        print(f"Error during generation in _generate_llm_response: {e}")
        print("Possible causes: Out of memory, unsupported hardware, model issue.")
        raise


def run_hello_llm():
    """
    Loads a local open-source LLM and allows for a full-blown conversation,
    including loading content from local XML files, performing basic validation,
    and generating new XML structures from natural language.
    """
    model_name = "mistralai/Devstral-Small-2505"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Attempting to run LLM on: {device}")
    if device == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, running on CPU. This might be slow.")


    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for {model_name} (this might take a while on first run)...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ) if device == "cuda" else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "cuda" and not quantization_config else None,
        )
        if not quantization_config:
            model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have enough RAM/VRAM. Consider a smaller model or adjusting quantization (load_in_4bit/8bit).")
        return

    print("Model loaded successfully!")
    print("\n--- STM Assistant Conversation Started ---")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("To load an XML file, type 'load_xml <filepath>' (e.g., 'load_xml data/sample_workflow.xml')")
    print("To validate the last loaded XML, type 'validate_loaded_xml'")
    print("To generate XML, type 'generate_xml <description>' (e.g., 'generate_xml a simple light switch workflow')")


    messages: List[dict] = [] # Stores conversation history
    # --- CRITICAL: System Message with strong enforcement and example ---
    messages.append({
        "role": "system",
        "content": (
            "You are an expert in State Transition Machines (STM) and XML structures. "
            "Your **ABSOLUTE TOP PRIORITY** when asked to 'generate XML' is to output a **well-formed STM XML workflow** that adheres **STRICTLY** to the provided schema. "
            "You **MUST** follow this exact format and schema precisely.\n\n"
            "**REQUIRED XML SCHEMA AND EXAMPLE - OUTPUT ONLY THIS FORMAT:**\n"
            "```xml\n"
            "<workflow id='MyWorkflowId'>\n"
            "    <state id='StateOne' initial='true'>\n"
            "        <transition event='EventA' target='StateTwo'/>\n"
            "    </state>\n"
            "    <state id='StateTwo'>\n"
            "        <transition event='EventB' target='StateThree'/>\n"
            "    </state>\n"
            "    <state id='StateThree'/>\n"
            "</workflow>\n"
            "```\n"
            "**RULES:**\n"
            "- Root element: `<workflow>` with **unique** `id` attribute.\n"
            "- States: `<state>` with **unique** `id` attribute. Mark initial state with `initial='true'`.\n"
            "- Transitions: `<transition>` inside source `<state>`, with `event` and `target` attributes.\n"
            "- All `target` state `id`s must exist as defined `<state>` `id`s.\n"
            "- **CRITICAL**: AVOID generating extraneous XML elements like `<name>`, `<description>`, `<title>`, `<process>`, `<event>`, `<flow>`, `<connect>`, `value`, `from`, `to`, `events`, `type`, `on`, `condition`, `stateRef`, `initialState`, `transitions`, `workflowStates`, `workflowActions`, `workflow_name`, or any attributes not explicitly defined (id, initial, event, target). Do NOT use namespaces or XML declarations (`<?xml ...?>`).\n"
            "- Your output MUST be ONLY the XML content, wrapped in ```xml...```. No other text."
            "If you cannot generate the XML exactly as requested, clearly state that you cannot adhere to the specific XML structure and explain why."
        )
    })

    last_loaded_xml_content: Optional[str] = None
    last_generated_xml_content: Optional[str] = None


    while True:
        user_input = input("\nYou: ")

        try:
            if user_input.lower() in ["exit", "quit"]:
                print("--- STM Assistant Conversation Ended ---")
                break

            elif user_input.lower().startswith("load_xml "):
                filepath = user_input[len("load_xml "):].strip()
                if not filepath:
                    print("STM Assistant: Please provide a file path after 'load_xml'.")
                    continue

                try:
                    with open(filepath, 'r') as f:
                        xml_content = f.read()

                    last_loaded_xml_content = xml_content
                    last_generated_xml_content = None # Reset generated XML when new is loaded
                    print(f"\nSTM Assistant: Successfully loaded XML from '{filepath}'.")
                    print(f"--- XML Content Preview ({len(xml_content)} chars) ---")
                    print(xml_content[:500] + ('...' if len(xml_content) > 500 else ''))
                    print("--- End XML Content Preview ---")

                    # --- CRITICAL FIX: Changed prompt after loading XML to be more direct ---
                    messages.append({
                        "role": "user",
                        "content": (
                            f"I have successfully loaded the following STM XML content:\n```xml\n{xml_content}\n```\n"
                            f"The XML preview is above. What would you like to do with this workflow? You can ask me to 'validate it', 'explain it', or 'suggest improvements'."
                        )
                    })
                    # --- END CRITICAL FIX ---

                    llm_response = _generate_llm_response(model, tokenizer, messages, device)
                    print(f"STM Assistant: {llm_response}") # Print the LLM's response to loaded XML
                    messages.append({"role": "assistant", "content": llm_response})


                except FileNotFoundError:
                    print(f"STM Assistant: Error: File not found at '{filepath}'. Please check the path.")
                    continue
                except Exception as e:
                    print(f"STM Assistant: Error reading file: {e}")
                    continue

            elif user_input.lower() == "validate_loaded_xml":
                xml_to_validate = last_loaded_xml_content or last_generated_xml_content
                if xml_to_validate is None:
                    print("STM Assistant: No XML content has been loaded or generated yet. Please use 'load_xml <filepath>' or 'generate_xml <description>' first.")
                    continue

                print("\nSTM Assistant: Validating the current XML content...")
                is_valid, errors, warnings = validate_stm_xml(xml_to_validate)

                validation_report = "Here's a validation report for the STM XML:\n\n"
                if is_valid:
                    validation_report += "Overall Status: The XML appears to be structurally valid based on initial checks.\n"
                else:
                    validation_report += "Overall Status: The XML has structural errors.\n"

                if errors:
                    validation_report += "\nErrors Found:\n" + "\n".join([f"- {e}" for e in errors])
                else:
                    validation_report += "\nNo Critical Errors Found.\n"

                if warnings:
                    validation_report += "\nWarnings Found:\n" + "\n".join([f"- {w}" for w in warnings])
                else:
                    validation_report += "\nNo Warnings Found.\n"

                # --- CRITICAL FIX: Make the LLM explicitly list errors/warnings ---
                analysis_request_message = {
                    "role": "user",
                    "content": (
                        f"The content I asked you to validate has been programmatically validated. "
                        f"Here is the validation summary:\n{validation_summary}\n"
                        f"Please analyze this summary, focusing on **listing each specific error and warning message exactly as provided**, "
                        f"and then provide clear, actionable suggestions for improving the XML to conform to the specified STM schema."
                    )
                }
                # --- END CRITICAL FIX ---
                messages.append(analysis_request_message)

                llm_response = _generate_llm_response(model, tokenizer, messages, device)
                print(f"STM Assistant: {llm_response}") # Print the LLM's analysis of the validation report
                messages.append({"role": "assistant", "content": llm_response})


            elif user_input.lower().startswith("generate_xml "):
                description = user_input[len("generate_xml "):].strip()
                if not description:
                    print("STM Assistant: Please provide a description after 'generate_xml'.")
                    continue

                # The system prompt handles the detailed XML generation instructions and example.
                # Here, we just give the description as the user's intent.
                messages.append({"role": "user", "content": f"Generate STM XML based on this description: {description}"})

                # --- First LLM call: Generate the XML (increased max_new_tokens for generation) ---
                llm_generated_response_full = _generate_llm_response(model, tokenizer, messages, device, max_new_tokens=1500)
                
                # --- CRITICAL FIX: Add the full raw LLM response to history IMMEDIATELY ---
                # This ensures the LLM has context of its own output, even if it was conversational.
                # This also ensures the turn structure is correct for the next analysis call.
                messages.append({"role": "assistant", "content": llm_generated_response_full})
                # --- END CRITICAL FIX ---

                xml_match = re.search(r"```xml\s*(.*?)\s*```", llm_generated_response_full, re.DOTALL)
                
                extracted_xml = "" # Initialize
                if xml_match:
                    extracted_xml = xml_match.group(1).strip()
                    last_generated_xml_content = extracted_xml # Store the newly generated XML
                    last_loaded_xml_content = None # Reset loaded XML when new is generated

                    print(f"\nSTM Assistant: Generated XML:\n{extracted_xml}\n") # Print the extracted XML for the user

                else: # If XML block not found in LLM's raw response
                    print("STM Assistant: Could not extract valid XML content (```xml``` block not found). The model might not have adhered to the output format.")
                    print(f"STM Assistant: Raw LLM response: \n{llm_generated_response_full}\n") # Show raw response for debugging
                    # If no XML was extracted, we'll still attempt to send a "validation failed" report to LLM for analysis.
                    # The `extracted_xml` will be empty, and `validate_stm_xml` will handle empty string.


                print("STM Assistant: Attempting to validate the generated content...")
                # --- Always validate, even if XML extraction failed, using the (possibly empty) extracted_xml ---
                is_valid, errors, warnings = validate_stm_xml(extracted_xml)

                validation_summary = "Generated Content Validation Status:\n" # Changed from XML Validation Status
                if xml_match:
                    if is_valid:
                        validation_summary += "Overall Status: The extracted XML appears to be structurally valid based on initial checks.\n"
                    else:
                        validation_summary += "Overall Status: The extracted XML has structural errors.\n"
                else:
                    validation_summary += "Overall Status: No valid XML block was found in the model's response. Please re-check the model's output and your request.\n"

                if errors:
                    validation_summary += "\nErrors Found:\n" + "\n".join([f"- {e}" for e in errors])
                else:
                    validation_summary += "\nNo Critical Errors Found.\n"

                if warnings:
                    validation_summary += "\nWarnings Found:\n" + "\n".join([f"- {w}" for w in warnings])
                else:
                    validation_summary += "\nNo Warnings Found.\n"

                # --- CRITICAL FIX: Make the LLM explicitly list errors/warnings ---
                analysis_request_message = {
                    "role": "user",
                    "content": (
                        f"The content I asked you to generate has been programmatically validated. "
                        f"Here is the validation summary:\n{validation_summary}\n"
                        f"Please analyze this summary, focusing on **listing each specific error and warning message exactly as provided**, "
                        f"and then provide clear, actionable suggestions for improving the generated content to conform to the specified STM XML schema. "
                        f"Focus ONLY on the structural XML requirements from the system message. If no XML was found, advise on how to better prompt for it."
                    )
                }
                # --- END CRITICAL FIX ---
                messages.append(analysis_request_message)

                # --- Second LLM call: Get analysis of validation ---
                print("\n[DEBUG] --- Triggering Second LLM Call for Validation Analysis ---") # DEBUG
                llm_analysis_response = _generate_llm_response(model, tokenizer, messages, device)
                print(f"STM Assistant: {llm_analysis_response}") # Print the LLM's analysis
                messages.append({"role": "assistant", "content": llm_analysis_response})


            else: # General LLM Response for regular chat input
                messages.append({"role": "user", "content": user_input})
                llm_response = _generate_llm_response(model, tokenizer, messages, device)
                print(f"STM Assistant: {llm_response}")
                messages.append({"role": "assistant", "content": llm_response})

        except Exception as e:
            print(f"An unexpected error occurred during conversation: {e}")
            print("Restarting conversation loop. Please try your input again or 'exit'.")
            break


if __name__ == "__main__":
    run_hello_llm()
