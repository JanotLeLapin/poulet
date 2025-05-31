CC = gcc
EMCC = emcc
LDFLAGS = -lm
LDFLAGS_WASM = -lm -s ALLOW_MEMORY_GROWTH=1
CFLAGS = -Wall -Wextra -g
CFLAGS_WASM = -Wall -Wextra -O3

COMMON_SRCS = poulet.c game.c ai.c
WRAPPER_SRC = wrapper.c
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
WASM_OBJS = $(COMMON_SRCS:.c=.wasm.o) $(WRAPPER_SRC:.c=.wasm.o)

TEST_TARGET = test
MAIN_TARGET = main
WASM_TARGET = poulet.js

all: $(TEST_TARGET) $(MAIN_TARGET)

wasm: $(WASM_TARGET)

$(TEST_TARGET): test.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(MAIN_TARGET): main.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(WASM_TARGET): $(WASM_OBJS)
	$(EMCC) $^ -o $@ \
		-s EXPORTED_FUNCTIONS='["_w_ai_layer_t_size","_w_ai_brain_t_size","_w_chess_color_t_size","_w_chess_move_t_size","_w_chess_game_t_size","_chess_init","_chess_is_check","_chess_legal_move","_chess_safe_move","_chess_do_move","_ai_layer_free","_poulet_brain_init","_poulet_next_move","_malloc","_free"]' \
		-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
		$(LDFLAGS_WASM)

%.wasm.o: %.c
	$(EMCC) $(CFLAGS_WASM) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(COMMON_OBJS) test.o main.o $(TEST_TARGET) $(MAIN_TARGET)
	rm -f $(WASM_OBJS) $(WASM_TARGET) poulet.wasm

.PHONY: all clean wasm
