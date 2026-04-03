#!/bin/bash
# If invoked via `sh`, re-exec with bash to honour the shebang
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# Ensure ~/.local/bin is on PATH (needed when script runs without a login shell)
export PATH="$HOME/.local/bin:$PATH"

# ── uv ────────────────────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
echo "Creating virtual environment with Python 3.12..."
uv venv --python 3.12 --clear .venv
. .venv/bin/activate

echo "Installing dependencies..."
uv pip install torch torchvision torchaudio pillow ipykernel huggingface_hub

# ── Hugging Face login ────────────────────────────────────────────────────────
echo ""
echo "Logging in to Hugging Face..."
echo "Generate a Read token at: https://huggingface.co/settings/tokens"
huggingface-cli login

# ── Install transformers  ────────────────────────────────────────────────
uv pip install transformers tqdm accelerate datasets hf_transfer matplotlib
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── ipykernel ─────────────────────────────────────────────────────────────────
echo "Setting up ipykernel for the notebook..."
python -m ipykernel install --user --name=gemma4 --display-name "Python 3.12 (Gemma 4)"
uv pip install ipywidgets

echo ""
echo "Environment setup complete! Run 'source .venv/bin/activate' to start."