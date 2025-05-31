#include <math.h>

#include "poulet.h"

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
  size_t i, invalid_moves = 0, selected_index = 0;
  float inputs[768], r, total_weight = 0.0f, cumulative_weight = 0.0f, max_score = -INFINITY;
  uint8_t tmp[4];
  chess_move_t move;

  encode_board(game->board, inputs);
  ai_brain_forward(brain, inputs);

  for (i = 0; i < brain->layers[2].output_size; i++) {
    move_from_index(tmp, i);
    move = chess_safe_move(game, tmp[1], tmp[0], tmp[3], tmp[2]);
    if (color != chess_color_from_square(game->board[tmp[0]][tmp[1]]) || CHESS_MOVE_ILLEGAL == move || CHESS_MOVE_UNSAFE == move) {
      brain->layers[2].outputs[i] = -INFINITY;
      invalid_moves++;
    }
  }

  if (brain->layers[2].output_size <= invalid_moves) {
    return -1;
  }

  act_softmax(brain->layers[2].outputs, brain->layers[2].output_size, temperature);

  for (i = 0; i < brain->layers[2].output_size; i++) {
    if (0 < brain->layers[2].outputs[i]) {
      total_weight += brain->layers[2].outputs[i];
    }
  }

  r = (float) rand() / (float) RAND_MAX * total_weight;
  cumulative_weight = 0.0f;

  for (i = 0; i < brain->layers[2].output_size; i++) {
    if (0 >= brain->layers[2].outputs) {
      continue;
    }

    cumulative_weight += brain->layers[2].outputs[i];
    if (r <= cumulative_weight) {
      selected_index = i;
      break;
    }
  }

  if (cumulative_weight < r) {
    for (i = 0; i < brain->layers[2].output_size; i++) {
      if (max_score < brain->layers[2].outputs[i]) {
        max_score = brain->layers[2].outputs[i];
        selected_index = i;
      }
    }
  }

  move_from_index(res, brain->layers[2].outputs[selected_index]);

  return 0;
}
