#ifndef _POULET_H
#define _POULET_H

#include <stdint.h>
#include <stdlib.h>

static const char *LETTERS = "abcdefgh";

typedef enum {
  AI_ACTIVATION_NONE = 0,
  AI_ACTIVATION_RELU,
  AI_ACTIVATION_SOFTMAX,
} ai_activation_type_t;

typedef struct {
  float temperature;
} ai_activation_data_softmax;

typedef struct {
  ai_activation_type_t type;
  union {
    ai_activation_data_softmax softmax;
  } data;
} ai_activation_t;

typedef struct {
  uint64_t input_size;
  uint64_t output_size;
  float *weights;
  float *biases;
  float *outputs;
  ai_activation_t activation;
} ai_layer_t;

typedef struct {
  uint64_t layer_count;
  ai_layer_t *layers;
} ai_brain_t;


typedef enum {
  CHESS_PIECE_PAWN = 1,
  CHESS_PIECE_BISHOP,
  CHESS_PIECE_KNIGHT,
  CHESS_PIECE_ROOK,
  CHESS_PIECE_QUEEN,
  CHESS_PIECE_KING,
} chess_piece_t;

typedef enum {
  CHESS_COLOR_BLACK = 0,
  CHESS_COLOR_WHITE,
} chess_color_t;

typedef enum {
  CHESS_MOVE_ILLEGAL = 0,
  CHESS_MOVE_LEGAL,
  CHESS_MOVE_TAKE,
  CHESS_MOVE_TAKE_ENPASSANT,
  CHESS_MOVE_CASTLE,

  CHESS_MOVE_UNSAFE,
} chess_move_t;

typedef uint8_t chess_square_t;
typedef chess_square_t chess_board_t[8][8];

typedef uint8_t chess_meta_t;

typedef struct {
  chess_board_t board;
  chess_meta_t meta;
} chess_game_t;

inline static chess_square_t
chess_new_square(chess_piece_t piece, chess_color_t color)
{
  chess_square_t res = 0;
  res |= piece;
  res |= color << 3;
  return res;
}

inline static chess_piece_t
chess_piece_from_square(chess_square_t square)
{
  return (chess_piece_t) (square & 0x07);
}

inline static chess_color_t
chess_color_from_square(chess_square_t square)
{
  return (chess_color_t) (square >> 3);
}

inline static void
chess_pretty_square(char *buf, uint8_t x, uint8_t y)
{
  buf[0] = LETTERS[x];
  buf[1] = 8 - y + 48;
  buf[2] = '\0';
}

inline static const char *
chess_stringify_piece(chess_piece_t piece)
{
  switch (piece) {
  case CHESS_PIECE_PAWN:
    return "pawn";
  case CHESS_PIECE_BISHOP:
    return "bishop";
  case CHESS_PIECE_KNIGHT:
    return "knight";
  case CHESS_PIECE_ROOK:
    return "rook";
  case CHESS_PIECE_QUEEN:
    return "queen";
  case CHESS_PIECE_KING:
    return "king";
  }
}

inline static const char *
chess_stringify_color(chess_color_t color)
{
  switch (color) {
  case CHESS_COLOR_WHITE:
    return "white";
  case CHESS_COLOR_BLACK:
    return "black";
  }
}

inline static int
chess_get_enpassant_file(chess_meta_t meta)
{
  return (0x01 == (meta & 0x01)) ? ((meta >> 2) & 0x07) : -1;
}

inline static int
chess_get_enpassant_color(chess_meta_t meta)
{
  return (0x01 == (meta & 0x01)) ? ((meta >> 1) & 0x01) : -1;
}

inline static int
chess_get_castling_rights(chess_meta_t meta, chess_color_t color)
{
  return 0 != (meta & 0x01 << (color + 5));
}

void chess_init(chess_game_t *game);

int chess_is_check(chess_game_t *game, chess_color_t color);
chess_move_t chess_legal_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by);
chess_move_t chess_safe_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by);

void chess_do_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by);

void act_softmax(float *logits, size_t logit_count, float temperature);

int ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size, ai_activation_t activation);
void ai_layer_offspring(ai_layer_t *child, ai_layer_t *a, ai_layer_t *b);
void ai_layer_forward(ai_layer_t *layer, float *input);
void ai_layer_free(ai_layer_t *layer);

int ai_brain_init(ai_brain_t *brain, size_t layer_count);
void ai_brain_offspring(ai_brain_t *a, ai_brain_t *b, ai_brain_t *child);
void ai_brain_forward(ai_brain_t *brain, float *input);
int ai_brain_save(ai_brain_t *brain, const char *filename);
int ai_brain_load(ai_brain_t *brain, const char *filename);
int ai_brain_load_from_buffer(ai_brain_t *brain, const uint8_t *buffer, size_t buffer_length);
void ai_brain_free(ai_brain_t *brain);

void poulet_brain_init(ai_brain_t *brain);
int poulet_next_move(uint8_t *res, chess_game_t *game, ai_brain_t *brain, chess_color_t color, float temperature);

#endif
