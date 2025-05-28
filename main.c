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
  init_chess_brain(&brain);
  ai_brain_free(&brain);
}
