#include <stdio.h>
#include <stdlib.h>

#include "ai.h"
#include "game.h"

typedef struct {
  size_t index;
  float score;
} scored_move_t;

int
compare_moves(const void *a, const void *b)
{
  const scored_move_t *move_a = a;
  const scored_move_t *move_b = b;
  if (move_a->score > move_b->score) return -1;
  if (move_a->score < move_b->score) return 1;
  return 0;
}

void
init_chess_brain(ai_brain_t *brain)
{
  ai_brain_init(brain, 3);
  ai_layer_init(&brain->layers[0], 768, 1024, ai_activation_relu);
  ai_layer_init(&brain->layers[1], 1024, 512, ai_activation_relu);
  ai_layer_init(&brain->layers[2], 512, 4096, ai_activation_relu);
}

void
encode_chess_board(chess_board_t board, float *inputs)
{
  uint8_t i, j;
  chess_square_t s;

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

int
main()
{
  ai_brain_t brain;
  chess_game_t game;
  float inputs[768];
  size_t i, best_move_index, src_index, dst_index;
  scored_move_t scored_moves[4096];
  uint8_t src_x, src_y, dst_x, dst_y;

  for (i = 0; i < 768; i++) {
    inputs[i] = 0.0f;
  }

  init_chess_brain(&brain);
  chess_init(&game);
  encode_chess_board(game.board, inputs);
  ai_brain_forward(&brain, inputs);

  for (i = 0; i < 4096; i++) {
    scored_moves[i].index = i;
    scored_moves[i].score = brain.layers[2].outputs[i];
  }

  qsort(scored_moves, 4096, sizeof(scored_move_t), compare_moves);

  for (i = 0; i < 4096; i++) {
    size_t move_index = scored_moves[i].index;

    src_index = move_index / 64;
    dst_index = move_index % 64;

    src_x = src_index / 8;
    src_y = src_index % 8;
    dst_x = dst_index / 8;
    dst_y = dst_index % 8;

    if (chess_safe_move(&game, src_x, src_y, dst_x, dst_y)) {
      break;
    }
  }

  printf("best valid move: (x: %d, y: %d to x: %d, y: %d)\n", src_x, src_y, dst_x, dst_y);

  ai_brain_free(&brain);
}
