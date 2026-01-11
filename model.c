
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    float* token_embedding;  // [vocab_size, hidden_size]
    float* weights;         // Tous les poids
} Transformer;

Transformer* load_model(const char* model_path) {
    FILE* f = fopen(model_path, "rb");
    if(!f) return NULL;
    
    Transformer* model = malloc(sizeof(Transformer));
    
    // Lire l'en-tête
    fread(&model->vocab_size, sizeof(int), 1, f);
    fread(&model->hidden_size, sizeof(int), 1, f);
    fread(&model->num_layers, sizeof(int), 1, f);
    fread(&model->num_heads, sizeof(int), 1, f);
    
    // Allouer et lire les poids
    size_t embed_size = model->vocab_size * model->hidden_size;
    model->token_embedding = malloc(embed_size * sizeof(float));
    fread(model->token_embedding, sizeof(float), embed_size, f);
    
    fclose(f);
    return model;
}

float* get_token_embedding(Transformer* model, int token_id) {
    return &model->token_embedding[token_id * model->hidden_size];
}
