# src/hello_llm.py

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List, Optional # Import Optional for type hinting

# --- NEW: Import the STM validator module ---
from stm_validator import validate_stm_xml
# --- END NEW ---

def run_hello_llm():
    """
    Loads a local open-source LLM and allows for a full-blown conversation,
    including loading content from local XML files and performing basic validation.
    """
    # Define the model to use
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # --- Determine the device to run on (GPU if available, else CPU) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Attempting to run LLM on: {device}")
    if device == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, running on CPU. This might be slow.")


    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- FIX: Set pad_token if it's not defined (common for generative models) ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- END FIX ---

    print(f"Loading model for {model_name} (this might take a while on first run)...")
    try:
        # Configure 4-bit quantization for GPU to save VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 # Improves compute speed for 4-bit operations
        ) if device == "cuda" else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            # If not using quantization (e.g., on CPU), explicitly set torch_dtype
            torch_dtype=torch.float16 if device == "cuda" and not quantization_config else None,
        )
        # Move model to device only if not loaded via quantization (bitsandbytes handles device placement)
        if not quantization_config:
            model.to(device)
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have enough RAM/VRAM. Consider a smaller model or adjusting quantization (load_in_4bit/8bit).")
        return

    print("Model loaded successfully!")
    print("\n--- STM Assistant Conversation Started ---")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("To load an XML file, type 'load_xml <filepath>' (e.g., 'load_xml data/sample_workflow.xml')")
    # --- NEW INSTRUCTION FOR VALIDATION ---
    print("To validate the last loaded XML, type 'validate_loaded_xml'")
    # --- END NEW INSTRUCTION ---


    messages: List[dict] = [] # Stores conversation history
    last_loaded_xml_content: Optional[str] = None # --- NEW: Store the last loaded XML ---


    while True:
        user_input = input("\nYou: ")

        # --- Check for termination command ---
        if user_input.lower() in ["exit", "quit"]:
            print("--- STM Assistant Conversation Ended ---")
            break

        # --- Check for file load command ---
        if user_input.lower().startswith("load_xml "):
            filepath = user_input[len("load_xml "):].strip()
            if not filepath:
                print("STM Assistant: Please provide a file path after 'load_xml'.")
                continue

            try:
                with open(filepath, 'r') as f:
                    xml_content = f.read()

                last_loaded_xml_content = xml_content # Store the content
                print(f"\nSTM Assistant: Successfully loaded XML from '{filepath}'.")
                print(f"--- XML Content Preview ({len(xml_content)} chars) ---")
                print(xml_content[:500] + ('...' if len(xml_content) > 500 else ''))
                print("--- End XML Content Preview ---")

                # Add the loaded XML content to the conversation history, then prompt LLM
                messages.append({"role": "user", "content": f"I have loaded the following STM XML content:\n```xml\n{xml_content}\n```\nWhat would you like to do with this XML? (e.g., 'validate it', 'explain it', 'suggest improvements')"})

            except FileNotFoundError:
                print(f"STM Assistant: Error: File not found at '{filepath}'. Please check the path.")
                continue
            except Exception as e:
                print(f"STM Assistant: Error reading file: {e}")
                continue
        # --- NEW: Handle validate_loaded_xml command ---
        elif user_input.lower() == "validate_loaded_xml":
            if last_loaded_xml_content is None:
                print("STM Assistant: No XML content has been loaded yet. Please use 'load_xml <filepath>' first.")
                continue

            print("\nSTM Assistant: Validating the loaded XML content...")
            is_valid, errors, warnings = validate_stm_xml(last_loaded_xml_content)

            validation_report = "Here's a validation report for the loaded STM XML:\n\n"
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

            validation_report += "\nPlease analyze this report and provide further insights or suggestions."

            # Send the validation report to the LLM for analysis and natural language output
            messages.append({"role": "user", "content": validation_report})
        # --- END NEW VALIDATION COMMAND ---

        else:
            # If not a special command, add the user's regular input to history
            messages.append({"role": "user", "content": user_input})

        # --- Generate LLM response (for both normal input and loaded XML/validation follow-up) ---
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        print("\nSTM Assistant (Generating)...")
        try:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=700, # Increased for more comprehensive responses
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode only the newly generated part of the text
            generated_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

            print(f"STM Assistant: {generated_response.strip()}")

            # Add assistant's response to history
            messages.append({"role": "assistant", "content": generated_response.strip()})

        except Exception as e:
            print(f"Error during generation: {e}")
            print("Possible causes: Out of memory, unsupported hardware, model issue.")
            break


if __name__ == "__main__":
    run_hello_llm()