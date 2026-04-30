import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from llama import Llama3
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).parent

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--use_kv_cache", action="store_true")
    args = args.parse_args()

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir=models_dir,
    )

    print(f"Downloaded models to {models_dir}")

    tokenizer = AutoTokenizer.from_pretrained(models_dir)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "What is the capital of France?",
        "What is the capital of United States of America?",
    ]
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )

    print("inputs", inputs)

    model = Llama3.from_pretrained(models_dir)

    start_time = time.monotonic()
    generated_tokens = model.generate(
        inputs["input_ids"], inputs["attention_mask"], use_kv_cache=args.use_kv_cache
    )

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_time = time.monotonic()
    print(f"Time taken: {end_time - start_time} seconds")
    print("response", response)
