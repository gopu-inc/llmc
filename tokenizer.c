
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 32000
#define MAX_TOKEN_LEN 100

typedef struct {
    char* tokens[VOCAB_SIZE];
    int vocab_size;
} Tokenizer;

Tokenizer* load_tokenizer(const char* vocab_file) {
    Tokenizer* tok = malloc(sizeof(Tokenizer));
    // Ici: charger le vocab depuis un fichier
    // Pour l'exemple, vocab minimal
    tok->tokens[0] = strdup("<unk>");
    tok->tokens[1] = strdup(" ");
    tok->tokens[2] = strdup("!");
    tok->tokens[3] = strdup("Bonjour");
    tok->tokens[4] = strdup("monde");
    tok->vocab_size = 5;
    return tok;
}

int encode(Tokenizer* tok, const char* text, int* output_ids, int max_len) {
    // Tokenisation simple par mots
    char* copy = strdup(text);
    char* token = strtok(copy, " ");
    int count = 0;
    
    while(token && count < max_len) {
        // Recherche naïve dans le vocab
        for(int i = 0; i < tok->vocab_size; i++) {
            if(strcmp(tok->tokens[i], token) == 0) {
                output_ids[count++] = i;
                break;
            }
        }
        token = strtok(NULL, " ");
    }
    
    free(copy);
    return count;
}

char* decode(Tokenizer* tok, int id) {
    if(id >= 0 && id < tok->vocab_size) {
        return tok->tokens[id];
    }
    return "<unk>";
}
