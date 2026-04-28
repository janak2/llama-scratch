from pathlib import Path

from huggingface_hub import snapshot_download
from llama import Llama3

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

    prompt = "What is the capital of France?"
    response = model.generate(prompt)
    print(response)
