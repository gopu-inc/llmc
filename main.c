#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define EMBED_DIM 64
#define N_HEADS 4
#define SEQ_LEN 10

// Softmax simple
void softmax(float* x, int n) {
    if (n <= 0) return;
    
    float max = x[0];
    float sum = 0.0f;
    
    for(int i = 1; i < n; i++) {
        if(x[i] > max) max = x[i];
    }
    
    for(int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    
    for(int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// Multiplication matricielle simple
void matmul(float* out, float* a, float* b, int m, int n, int k) {
    // out = a * b
    // a: m x n, b: n x k, out: m x k
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            float sum = 0.0f;
            for(int l = 0; l < n; l++) {
                sum += a[i * n + l] * b[l * k + j];
            }
            out[i * k + j] = sum;
        }
    }
}

// Implémentation simple d'une tête d'attention
void attention_head(float* output, float* Q, float* K, float* V, 
                    int seq_len, int d_head) {
    // Allouer mémoire pour les scores
    float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    if (!scores) {
        printf("Erreur d'allocation mémoire!\n");
        return;
    }
    
    // scores = Q * K^T / sqrt(d_head)
    for(int i = 0; i < seq_len; i++) {
        for(int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for(int kk = 0; kk < d_head; kk++) {
                sum += Q[i * d_head + kk] * K[j * d_head + kk];
            }
            scores[i * seq_len + j] = sum / sqrtf((float)d_head);
        }
    }
    
    // Softmax sur chaque ligne
    for(int i = 0; i < seq_len; i++) {
        softmax(&scores[i * seq_len], seq_len);
    }
    
    // output = scores * V
    matmul(output, scores, V, seq_len, seq_len, d_head);
    
    free(scores);
}

// Fonction principale REQUISE pour l'exécution
int main() {
    printf("=== Démarrage du Mini LLM en C ===\n");
    
    srand(time(NULL)); // Initialisation du générateur aléatoire
    
    int d_head = EMBED_DIM / N_HEADS;
    
    // Allocation mémoire
    float* input = (float*)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));
    float* output = (float*)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));
    
    if (!input || !output) {
        printf("Erreur d'allocation mémoire!\n");
        return 1;
    }
    
    // Initialisation aléatoire
    printf("Initialisation des données...\n");
    for(int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        output[i] = 0.0f;
    }
    
    printf("Dimensions:\n");
    printf("  Longueur de séquence (SEQ_LEN): %d\n", SEQ_LEN);
    printf("  Dimension d'embedding (EMBED_DIM): %d\n", EMBED_DIM);
    printf("  Nombre de têtes (N_HEADS): %d\n", N_HEADS);
    printf("  Dimension par tête (d_head): %d\n", d_head);
    
    printf("\nCalcul de l'attention...\n");
    
    // Calcul pour une seule tête
