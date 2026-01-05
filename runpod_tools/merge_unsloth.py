#!/usr/bin/env python3
import argparse
import os
from unsloth import FastModel
import torch

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA with base model using Unsloth (Memory Efficient)")
    parser.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged model")
    args = parser.parse_args()

    print(f"Loading model {args.base_model} and adapter {args.adapter_dir}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = 2048,
        load_in_4bit = True,
        token = os.environ.get("HF_TOKEN"),
    )
    
    # Load the adapter
    model = FastModel.for_inference(model) # Enable native 2x faster inference
    model.load_adapter(args.adapter_dir)

    print(f"Merging and saving to {args.output_dir}...")
    # save_pretrained_merged will merge the LoRA weights into the base model
    # and save as float16/bfloat16
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method = "merged_16bit")
    
    print("Done!")

if __name__ == "__main__":
    main()
