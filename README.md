API for Templated LLM Access to Sources

commands required to use GGUF model, tested on Windows 10.

- Install dependencies: 
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv add accelerate bert-score lxml
winget install llama.cpp
uv pip install huggingface_hub hf_xet openai

- create ATLAS\models\gemma3 folder

- This command downloads the model:
uvx --from huggingface_hub hf download unsloth/gemma-3-4b-it-GGUF --include "gemma-3-4b-it-Q4_K_M.gguf" --local-dir ATLAS\models\gemma3





