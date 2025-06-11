# src/hello_llm.py

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List

def run_hello_llm():
    """
    Loads a local open-source LLM and allows for a full-blown conversation.
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

    # --- FIX: Set pad_token if it's not defined ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- END FIX ---

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

    messages: List[dict] = []

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("--- STM Assistant Conversation Ended ---")
            break

        messages.append({"role": "user", "content": user_input})

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        print("\nSTM Assistant (Generating)...")
        try:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

            print(f"STM Assistant: {generated_response.strip()}")

            # --- FIX: Change role from "model" to "assistant" ---
            messages.append({"role": "assistant", "content": generated_response.strip()})
            # --- END FIX ---

        except Exception as e:
            print(f"Error during generation: {e}")
            print("Possible causes: Out of memory, unsupported hardware, model issue.")
            break


if __name__ == "__main__":
    run_hello_llm()