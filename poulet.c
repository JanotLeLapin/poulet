#include <math.h>

#include "poulet.h"

typedef struct {
  size_t index;
  float score;
} scored_move_t;

static int
compare_moves(const void *a, const void *b)
{
  const scored_move_t *move_a = a;
  const scored_move_t *move_b = b;
  if (move_a->score > move_b->score) return -1;
  if (move_a->score < move_b->score) return 1;
  return 0;
}

void
encode_board(chess_board_t board, float *inputs)
{
  size_t i, j;
  chess_square_t s;

  for (i = 0; i < 768; i++) {
    inputs[i] = 0.0f;
  }

  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      s = board[i][j];
      if (0 == s) {
        continue;
      }

      inputs[(i * 8 + j) * 12 + chess_piece_from_square(s) + chess_color_from_square(s) * 6] = 1.0f;
    }
  }
}

static inline void
move_from_index(uint8_t *res, size_t idx)
{
  size_t src_index, dst_index;

  src_index = idx / 64;
  dst_index = idx % 64;

  res[0] = src_index / 8;
  res[1] = src_index % 8;
  res[2] = dst_index / 8;
  res[3] = dst_index % 8;
}

void
poulet_brain_init(ai_brain_t *brain)
{
  ai_brain_init(brain, 3);
  ai_layer_init(&brain->layers[0], 768, 1024, (ai_activation_t) { .type = AI_ACTIVATION_RELU });
  ai_layer_init(&brain->layers[1], 1024, 512, (ai_activation_t) { .type = AI_ACTIVATION_RELU });
  ai_layer_init(&brain->layers[2], 512, 4096, (ai_activation_t) { .type = AI_ACTIVATION_NONE });
}

int
poulet_next_move(uint8_t *res, chess_game_t *game, ai_brain_t *brain, chess_color_t color, float temperature)
{
  size_t i;
  float inputs[768];
  uint8_t tmp[4];
  scored_move_t scored_moves[4096];
  chess_move_t move;

  encode_board(game->board, inputs);
  ai_brain_forward(brain, inputs);

  for (i = 0; i < brain->layers[2].output_size; i++) {
    move_from_index(tmp, i);
    move = chess_safe_move(game, tmp[1], tmp[0], tmp[3], tmp[2]);
    if (color != chess_color_from_square(game->board[tmp[0]][tmp[1]]) || CHESS_MOVE_ILLEGAL == move || CHESS_MOVE_UNSAFE == move) {
      brain->layers[2].outputs[i] = -INFINITY;
    }
  }

  act_softmax(brain->layers[2].outputs, brain->layers[2].output_size, temperature);

  for (i = 0; i < 4096; i++) {
    scored_moves[i].index = i;
    scored_moves[i].score = brain->layers[2].outputs[i];
  }

  qsort(scored_moves, 4096, sizeof(scored_move_t), compare_moves);

  for (i = 0; i < brain->layers[2].output_size; i++) {
    move_from_index(res, scored_moves[i].index);
    move = chess_safe_move(game, res[1], res[0], res[3], res[2]);
    if (color == chess_color_from_square(game->board[res[0]][res[1]]) && CHESS_MOVE_ILLEGAL != move && CHESS_MOVE_UNSAFE != move) {
      return 0;
    }
  }

  return -1;
}
