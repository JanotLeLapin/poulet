#include <stdio.h>

#include "ai.h"
#include "game.h"

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
  size_t i, highest_score = 0, src_index, dst_index;
  uint8_t src_x, src_y, dst_x, dst_y;

  for (i = 0; i < 768; i++) {
    inputs[i] = 0.0f;
  }

  init_chess_brain(&brain);
  chess_init(&game);
  encode_chess_board(game.board, inputs);
  ai_brain_forward(&brain, inputs);

  for (i = 0; i < 4096; i++) {
    if (brain.layers[2].outputs[i] > brain.layers[2].outputs[highest_score]) {
      highest_score = i;
    }
  }

  src_index = highest_score / 64;
  dst_index = highest_score % 64;

  src_x = src_index / 8;
  src_y = src_index % 8;
  dst_x = dst_index / 8;
  dst_y = dst_index % 8;

  printf("highest score: %ld (x: %d, y: %d to x: %d, y: %d)\n", highest_score, src_x, src_y, dst_x, dst_y);

  ai_brain_free(&brain);
}
