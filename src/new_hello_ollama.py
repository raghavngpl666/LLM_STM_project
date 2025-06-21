# src/new_hello_ollama.py

import os
import re
import json
import requests
from typing import List, Dict, Tuple, Optional

from new_stm_validator import validate_stm_xml

# --- Configuration for Ollama ---
OLLAMA_API_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "stm-deepseek:latest"

# Helper function to generate LLM response from Ollama
def _generate_ollama_response(messages: List[dict], stream: bool = False, raw: bool = False, max_new_tokens: int = 2000, expect_json_output: bool = False) -> str:
    """
    Sends a request to the Ollama API and retrieves the response.

    Args:
        messages: A list of message dictionaries for the conversation history.
                  The last message should be the user's current prompt.
        stream: Whether to stream the response (not used for this specific app structure, but good to have).
        raw: Whether to get raw response (useful for debugging).
        max_new_tokens: Maximum number of tokens to generate.
        expect_json_output: If True, the function will attempt to extract and parse a JSON block.
                            If False, it will return the raw text response.

    Returns:
        The generated text content from the Ollama model.
    """
    headers = {"Content-Type": "application/json"}
    
    current_prompt_content = messages[-1]["content"] if messages else ""

    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": current_prompt_content,
        "stream": stream,
        "raw": raw,
        "options": {
            "temperature": 0.2,
            "num_predict": max_new_tokens,
        }
    }

    print(f"\n[DEBUG] Sending request to Ollama API for model: {OLLAMA_MODEL_NAME} with num_predict={max_new_tokens}, expect_json_output={expect_json_output}")

    try:
        response = requests.post(OLLAMA_API_BASE_URL, headers=headers, json=data)
        response.raise_for_status()

        full_response_content = ""
        json_response = response.json()
        
        if 'response' in json_response:
            full_response_content = json_response['response']
        elif 'message' in json_response and 'content' in json_response['message']:
            full_response_content = json_response['message']['content']
        else:
            print(f"[ERROR] Unexpected Ollama API response format: {json_response}")
            return "Error: Unexpected response format from Ollama."

        if expect_json_output:
            # Attempt to extract JSON block using regex if expected
            json_match = re.search(r"```json\s*(.*?)\s*```", full_response_content, re.DOTALL)
            extracted_json_str = ""
            if json_match:
                extracted_json_str = json_match.group(1).strip()
            else:
                # Fallback: if no ```json``` block found, try to assume the whole response is JSON
                if full_response_content.strip().startswith('{') and full_response_content.strip().endswith('}'):
                    extracted_json_str = full_response_content.strip()
            
            try:
                if not extracted_json_str:
                    raise json.JSONDecodeError("No JSON block found or extracted.", full_response_content, 0)
                return json.dumps(json.loads(extracted_json_str)) # Return as compact JSON string
            except json.JSONDecodeError as e:
                print(f"STM Assistant: Error: Failed to parse model's response as JSON: {e}")
                print(f"Raw model response: {full_response_content}")
                return f"Error: Model did not provide valid JSON as expected. Details: {e}"
        else:
            # If not expecting JSON, return the full content as plain text,
            # but try to strip any leading/trailing markdown code blocks if they accidentally appear.
            # This is a heuristic to clean up unintended markdown from the model.
            cleaned_response = re.sub(r"```[a-zA-Z]*\s*(.*?)\s*```", r"\1", full_response_content, flags=re.DOTALL)
            # Also remove specific Modelfile instructions if they appear outside a code block
            cleaned_response = re.sub(r"CRITICAL STM XML SCHEMA RULES.*|ABSOLUTE RULES FOR GENERATION.*|Your response MUST be the JSON object.*", "", cleaned_response, flags=re.DOTALL)
            return cleaned_response.strip()


    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}. Is Ollama running and the model '{OLLAMA_MODEL_NAME}' available?")
        return f"Error: Could not connect to Ollama. Please ensure Ollama is running and '{OLLAMA_MODEL_NAME}' is available. Details: {e}"


def run_new_ollama_assistant():
    """
    Main conversational loop for the Ollama-powered STM Assistant.
    """
    print("\n--- Ollama STM Assistant Conversation Started ---")
    print(f"Targeting Ollama model: {OLLAMA_MODEL_NAME}")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("To load an XML file, type 'load_xml <filepath>' (e.g., 'load_xml data/sample_workflow.xml')")
    print("To validate the last loaded XML, type 'validate_loaded_xml'")
    print("To generate XML, type 'generate_xml <description>' (e.g., 'generate_xml a simple light switch workflow')")
    print("To suggest optimizations, type 'optimize_xml'")
    print("To generate a test case, type 'generate_test_case <description>' (e.g., 'generate_test_case for the door workflow')")

    messages: List[dict] = []
    
    last_loaded_xml_content: Optional[str] = None
    last_generated_xml_content: Optional[str] = None

    while True:
        user_input = input("\nYou: ")

        try:
            if user_input.lower() in ["exit", "quit"]:
                print("--- Ollama STM Assistant Conversation Ended ---")
                break

            elif user_input.lower().startswith("load_xml "):
                filepath = user_input[len("load_xml "):].strip()
                if not filepath:
                    print("STM Assistant: Please provide a file path after 'load_xml'.")
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        xml_content = f.read()

                    last_loaded_xml_content = xml_content
                    last_generated_xml_content = None
                    print(f"\nSTM Assistant: Successfully loaded XML from '{filepath}'.")
                    print(f"--- XML Content Preview ({len(xml_content)} chars) ---")
                    print(xml_content[:500] + ('...' if len(xml_content) > 500 else ''))
                    print("--- End XML Content Preview ---")

                    load_ack_prompt = (
                        f"I have just loaded an XML file. The preview is:\n```xml\n{xml_content[:200]}...\n```\n"
                        f"What would you like to do with this workflow? You can ask me to 'validate it', 'explain it', or 'suggest improvements'."
                    )
                    messages.append({"role": "user", "content": load_ack_prompt})

                    # Expect narrative for load_ack_prompt response
                    ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                    print(f"STM Assistant: {ollama_response_content}")
                    messages.append({"role": "assistant", "content": ollama_response_content})


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

                print("\nSTM Assistant: Validating the current XML content against Modelfile schema...")
                is_valid, errors, warnings = validate_stm_xml(xml_to_validate)

                validation_summary = "Validation Report:\n"
                if is_valid:
                    validation_summary += "Overall Status: The XML appears to be structurally valid based on Deepseek-Coder's Modelfile schema checks.\n"
                else:
                    validation_summary += "Overall Status: The XML has structural errors or deviations from the Modelfile schema.\n"

                if errors:
                    validation_summary += "\nErrors Found:\n" + "\n".join([f"- {e}" for e in errors])
                else:
                    validation_summary += "\nNo Critical Errors Found.\n"

                if warnings:
                    validation_summary += "\nWarnings Found:\n" + "\n".join([f"- {w}" for w in warnings])
                else:
                    validation_summary += "\nNo Warnings Found.\n"

                analysis_prompt = (
                    f"A programmatic validation has been performed on the STM XML. "
                    f"Here is the validation summary:\n{validation_summary}\n\n"
                    f"Please analyze this summary. **Explicitly list each specific error and warning message exactly as provided.** "
                    f"Then, provide clear, actionable suggestions for improving the XML to conform to the Modelfile's STM XML schema."
                )
                messages.append({"role": "user", "content": analysis_prompt})

                # Expect narrative for validation analysis response
                ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                print(f"STM Assistant: {ollama_response_content}")
                messages.append({"role": "assistant", "content": ollama_response_content})


            elif user_input.lower().startswith("generate_xml "):
                description = user_input[len("generate_xml "):].strip()
                if not description:
                    print("STM Assistant: Please provide a description after 'generate_xml'.")
                    continue

                generation_prompt = f"Create an STM XML workflow for: {description}"
                messages.append({"role": "user", "content": generation_prompt})

                print("\nSTM Assistant: Generating STM XML workflow...")
                
                # IMPORTANT: Expect JSON output for generate_xml
                ollama_raw_response_str = _generate_ollama_response(messages, max_new_tokens=2000, expect_json_output=True)

                # Update messages history with the full raw response for context (if it was valid JSON)
                try:
                    ollama_json_response_parsed = json.loads(ollama_raw_response_str) # Re-parse for structure check
                    messages.append({"role": "assistant", "content": ollama_raw_response_str})
                except json.JSONDecodeError:
                    print("STM Assistant: Model's response was not valid JSON, cannot add to history for context.")
                    messages.append({"role": "assistant", "content": f"ERROR: Invalid JSON response: {ollama_raw_response_str}"})


                # Attempt to parse the JSON response
                try:
                    # If ollama_raw_response_str is not valid JSON string, this will fail.
                    # _generate_ollama_response now handles the JSON parsing and error, returning a specific error string.
                    if ollama_raw_response_str.startswith("Error:"):
                        print(ollama_raw_response_str) # Print the error directly from _generate_ollama_response
                        continue # Skip further processing if JSON extraction failed

                    ollama_json_response = json.loads(ollama_raw_response_str) # Already extracted and verified JSON

                    generated_text_summary = ollama_json_response.get("text", "No text summary provided by model.")
                    generated_xml = ollama_json_response.get("xml", "")
                    generated_agent_question = ollama_json_response.get("agent Question", "")
                    generated_context = ollama_json_response.get("context", "")

                    if not generated_xml:
                        print("STM Assistant: Model generated response but XML field is empty or missing.")
                        print(f"Model's full response (JSON): {ollama_raw_response_str}")
                    else:
                        last_generated_xml_content = generated_xml
                        last_loaded_xml_content = None

                        print(f"\nSTM Assistant: Model's Summary: {generated_text_summary}")
                        print(f"STM Assistant: Model's Context: {generated_context}")
                        print(f"STM Assistant: Generated XML:\n{generated_xml}\n")
                        if generated_agent_question:
                            print(f"STM Assistant: Model's Question: {generated_agent_question}")
                        
                        print("\nSTM Assistant: Attempting to validate the generated XML...")
                        is_valid, errors, warnings = validate_stm_xml(generated_xml)

                        validation_summary = "Generated XML Validation Status:\n"
                        if is_valid:
                            validation_summary += "Overall Status: The generated XML appears to be structurally valid based on Deepseek-Coder's Modelfile schema checks.\n"
                        else:
                            validation_summary += "Overall Status: The XML has structural errors or deviations from the Modelfile schema.\n"

                        if errors:
                            validation_summary += "\nErrors Found:\n" + "\n".join([f"- {e}" for e in errors])
                        else:
                            validation_summary += "\nNo Critical Errors Found.\n"

                        if warnings:
                            validation_summary += "\nWarnings Found:\n" + "\n".join([f"- {w}" for w in warnings])
                        else:
                            validation_summary += "\nNo Warnings Found.\n"

                        analysis_prompt = (
                            f"The STM XML workflow you just generated has been programmatically validated. "
                            f"Here is the validation summary:\n{validation_summary}\n\n"
                            f"Please analyze this summary. **Explicitly list each specific error and warning message exactly as provided.** "
                            f"Then, provide clear, actionable suggestions for improving the generated XML to conform to the Modelfile's STM XML schema."
                        )
                        messages.append({"role": "user", "content": analysis_prompt})

                        ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                        print(f"STM Assistant: {ollama_response_content}")
                        messages.append({"role": "assistant", "content": ollama_response_content})

                except json.JSONDecodeError as e: # This block is now less likely to be hit as _generate_ollama_response handles it
                    print(f"STM Assistant: Error: Failed to parse model's response as JSON (unexpected path): {e}")
                    print(f"Raw model response: {ollama_raw_response_str}")
                    analysis_prompt = (
                        f"I asked you to generate STM XML in JSON format, but your response was not valid JSON or was incomplete. "
                        f"Raw response received:\n```\n{ollama_raw_response_str[:500]}...\n```\n"
                        f"Please analyze this issue. Why might you have failed to produce valid JSON as per your Modelfile instructions? "
                        f"Provide suggestions on how I can better prompt or what might be wrong."
                    )
                    messages.append({"role": "user", "content": analysis_prompt})
                    ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                    print(f"STM Assistant: {ollama_response_content}")
                    messages.append({"role": "assistant", "content": ollama_response_content})


            elif user_input.lower() == "optimize_xml":
                if last_loaded_xml_content is None and last_generated_xml_content is None:
                    print("STM Assistant: No XML content has been loaded or generated yet to optimize.")
                    continue

                xml_to_optimize = last_loaded_xml_content or last_generated_xml_content
                
                print("\nSTM Assistant: Suggesting optimizations for the current STM XML workflow...")
                optimization_prompt = (
                    f"Analyze the following STM XML workflow for potential optimizations. "
                    f"Consider aspects like state minimization, transition efficiency, clarity, and adherence to best practices for the Modelfile's schema. "
                    f"The XML is:\n```xml\n{xml_to_optimize}\n```\n"
                    f"Provide actionable suggestions as a narrative, not XML code."
                )
                messages.append({"role": "user", "content": optimization_prompt})
                
                # IMPORTANT: Expect narrative output for optimize_xml
                ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                print(f"STM Assistant: {ollama_response_content}")
                messages.append({"role": "assistant", "content": ollama_response_content})

            elif user_input.lower().startswith("generate_test_case "):
                description = user_input[len("generate_test_case "):].strip()
                if not description:
                    print("STM Assistant: Please provide a description for the test case scenario.")
                    continue

                xml_context = ""
                if last_loaded_xml_content:
                    xml_context = f"\nThe last loaded XML was:\n```xml\n{last_loaded_xml_content}\n```\n"
                elif last_generated_xml_content:
                    xml_context = f"\nThe last generated XML was:\n```xml\n{last_generated_xml_content}\n```\n"

                print("\nSTM Assistant: Generating a test case scenario...")
                test_case_prompt = (
                    f"Create a simple, step-by-step test case *scenario* for the workflow described as: '{description}'. "
                    f"Focus on a narrative sequence of events and expected state changes. DO NOT generate any XML or JSON. "
                    f"Provide the test case as a numbered list of steps (e.g., '1. Start in X state.', '2. Trigger Y event.', '3. Expect Z state.')."
                    f"{xml_context}"
                )
                messages.append({"role": "user", "content": test_case_prompt})

                # IMPORTANT: Expect narrative output for generate_test_case
                ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                print(f"STM Assistant: {ollama_response_content}")
                messages.append({"role": "assistant", "content": ollama_response_content})


            else: # General LLM Response for regular chat input
                messages.append({"role": "user", "content": user_input})
                # Expect narrative output for general chat
                ollama_response_content = _generate_ollama_response(messages, expect_json_output=False)
                print(f"STM Assistant: {ollama_response_content}")
                messages.append({"role": "assistant", "content": ollama_response_content})

        except Exception as e:
            print(f"An unexpected error occurred during conversation: {e}")
            print("Restarting conversation loop. Please try your input again or 'exit'.")
            messages = []


if __name__ == "__main__":
    run_new_ollama_assistant()
