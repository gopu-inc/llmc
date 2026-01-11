
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define EMBED_DIM 64
#define SEQ_LEN 10

// Softmax simple
void softmax(float* x, int n) {
    if (n == 0) return;
    
    float max = x[0];
    for(int i = 1; i < n; i++) if(x[i] > max) max = x[i];
    
    float sum = 0.0f;
    for(int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    
    for(int i = 0; i < n; i++) x[i] /= sum;
}

// LayerNorm simplifié
void layer_norm(float* output, float* input, float* gamma, float* beta, int n) {
    float mean = 0.0f;
    float var = 0.0f;
    
    for(int i = 0; i < n; i++) mean += input[i];
    mean /= n;
    
    for(int i = 0; i < n; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= n;
    
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    
    for(int i = 0; i < n; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// GELU activation
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Feed Forward Network simple
void feed_forward(float* output, float* input, float* W1, float* W2, 
                  float* b1, float* b2, int dim, int hidden_dim) {
    // hidden = gelu(input * W1 + b1)
    // output = hidden * W2 + b2
    
    float* hidden = (float*)malloc(hidden_dim * sizeof(float));
    
    // hidden = input * W1 + b1
    for(int i = 0; i < hidden_dim; i++) {
        hidden[i] = b1[i];
        for(int j = 0; j < dim; j++) {
            hidden[i] += input[j] * W1[j * hidden_dim + i];
        }
        hidden[i] = gelu(hidden[i]);
    }
    
    // output = hidden * W2 + b2
    for(int i = 0; i < dim; i++) {
        output[i] = b2[i];
        for(int j = 0; j < hidden_dim; j++) {
            output[i] += hidden[j] * W2[j * dim + i];
        }
    }
    
    free(hidden);
}

int main() {
    printf("=== Mini Transformer en C ===\n");
    srand(time(NULL));
    
    // Dimensions
    int dim = EMBED_DIM;
    int seq_len = SEQ_LEN;
    int hidden_dim = dim * 4;
    
    printf("Configuration:\n");
    printf("- Embedding dim: %d\n", dim);
    printf("- Séquence length: %d\n", seq_len);
    printf("- Hidden dim: %d\n\n", hidden_dim);
    
    // Allocation mémoire
    float* input = (float*)malloc(dim * sizeof(float));
    float* output = (float*)malloc(dim * sizeof(float));
    float* gamma = (float*)malloc(dim * sizeof(float));
    float* beta = (float*)malloc(dim * sizeof(float));
    float* W1 = (float*)malloc(dim * hidden_dim * sizeof(float));
    float* W2 = (float*)malloc(hidden_dim * dim * sizeof(float));
    float* b1 = (float*)malloc(hidden_dim * sizeof(float));
    float* b2 = (float*)malloc(dim * sizeof(float));
    
    // Initialisation aléatoire simple
    for(int i = 0; i < dim; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }
    
    for(int i = 0; i < dim * hidden_dim; i++) W1[i] = ((float)rand() / RAND_MAX) * 0.1f;
    for(int i = 0; i < hidden_dim * dim; i++) W2[i] = ((float)rand() / RAND_MAX) * 0.1f;
    for(int i = 0; i < hidden_dim; i++) b1[i] = ((float)rand() / RAND_MAX) * 0.1f;
    for(int i = 0; i < dim; i++) b2[i] = ((float)rand() / RAND_MAX) * 0.1f;
    
    // Test LayerNorm
    printf("Test LayerNorm:\n");
    layer_norm(output, input, gamma, beta, dim);
    printf("Input[0]: %.3f, Output[0]: %.3f\n\n", input[0], output[0]);
    
    // Test GELU
    printf("Test GELU:\n");
    for(int i = 0; i < 3; i++) {
        float x = i - 1.0f;
        printf("gelu(%.1f) = %.3f\n", x, gelu(x));
    }
    printf("\n");
    
    // Test Feed Forward
    printf("Test Feed Forward Network:\n");
    feed_forward(output, input, W1, W2, b1, b2, dim, hidden_dim);
    printf("FFN Output[0]: %.3f\n", output[0]);
    
    // Libération mémoire
    free(input);
    free(output);
    free(gamma);
    free(beta);
    free(W1);
    free(W2);
    free(b1);
    free(b2);
    
    printf("\n=== Programme terminé avec succès ===\n");
    return 0;
}
