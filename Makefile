CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm

# Fichiers source
SRCS = main.c tokenizer.c model.c
OBJS = $(SRCS:.c=.o)
TARGET = llm

# Règles
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
