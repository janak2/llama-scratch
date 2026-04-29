from pathlib import Path

from huggingface_hub import snapshot_download
from llama import Llama3
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).parent

if __name__ == "__main__":
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir=models_dir,
    )

    print(f"Downloaded models to {models_dir}")

    model = Llama3.from_pretrained(models_dir)
    tokenizer = AutoTokenizer.from_pretrained(models_dir)

    prompt = "What is the capital of France?"

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )

    print(inputs)

    response = model.generate(
        inputs["input_ids"].unsqueeze(0), inputs["attention_mask"].unsqueeze(0)
    )
    print(response)
