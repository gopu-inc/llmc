
#!/bin/bash

# Télécharger un petit modèle
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model

# Ou pour un très petit modèle
wget https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors
wget https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json

echo "Modèles téléchargés!"
