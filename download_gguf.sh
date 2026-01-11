#!/bin/bash

echo "=== Téléchargement de modèles GGUF ==="

# Choisissez un modèle (décommentez une ligne)
# MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.gguf"
# MODEL_URL="https://huggingface.co/bartowski/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf"

# Nom du fichier
MODEL_FILE=$(basename "$MODEL_URL")

echo "Téléchargement de $MODEL_FILE..."
wget "$MODEL_URL" -O "$MODEL_FILE"

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modèle téléchargé: $MODEL_FILE"
    echo "Taille: $(ls -lh "$MODEL_FILE" | awk '{print $5}')"
else
    echo "❌ Échec du téléchargement"
fi
