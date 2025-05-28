#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ai.h"
#include "game.h"

typedef struct {
  size_t index;
  float score;
} scored_move_t;

typedef struct {
  uint8_t src_x;
  uint8_t src_y;
  uint8_t dst_x;
  uint8_t dst_y;
} move_t;

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

move_t
predict_next_move(chess_game_t *game, ai_brain_t *brain, chess_color_t color)
{
  size_t i, src_index, dst_index;
  float inputs[768];
  scored_move_t scored_moves[4096];
  move_t res;

  for (i = 0; i < 768; i++) {
    inputs[i] = 0.0f;
  }

  encode_chess_board(game->board, inputs);
  ai_brain_forward(brain, inputs);

  for (i = 0; i < 4096; i++) {
    scored_moves[i].index = i;
    scored_moves[i].score = brain->layers[2].outputs[i];
  }

  qsort(scored_moves, 4096, sizeof(scored_move_t), compare_moves);

  for (i = 0; i < 4096; i++) {
    size_t move_index = scored_moves[i].index;

    src_index = move_index / 64;
    dst_index = move_index % 64;

    res.src_x = src_index / 8;
    res.src_y = src_index % 8;
    res.dst_x = dst_index / 8;
    res.dst_y = dst_index % 8;

    if (color == chess_color_from_square(game->board[res.src_y][res.src_x]) && chess_safe_move(game, res.src_x, res.src_y, res.dst_x, res.dst_y)) {
      break;
    }
  }

  return res;
}

int
main()
{
  ai_brain_t brain_a, brain_b;
  chess_game_t game;
  move_t move;
  char src[3], dst[3];

  srand(time(NULL));

  init_chess_brain(&brain_a);
  init_chess_brain(&brain_b);
  chess_init(&game);

  for (;;) {
    if ('q' == fgetc(stdin)) {
      break;
    }

    move = predict_next_move(&game, &brain_a, CHESS_COLOR_WHITE);
    chess_pretty_square(src, move.src_x, move.src_y);
    chess_pretty_square(dst, move.dst_x, move.dst_y);
    printf("white: %s to %s\n", src, dst);
    chess_do_move(&game, move.src_x, move.src_y, move.dst_x, move.dst_y);

    move = predict_next_move(&game, &brain_a, CHESS_COLOR_BLACK);
    chess_pretty_square(src, move.src_x, move.src_y);
    chess_pretty_square(dst, move.dst_x, move.dst_y);
    printf("black: %s to %s\n", src, dst);
    chess_do_move(&game, move.src_x, move.src_y, move.dst_x, move.dst_y);
  }

  ai_brain_free(&brain_a);
  ai_brain_free(&brain_b);
}
