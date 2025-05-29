#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ai.h"
#include "game.h"

#define POPULATION_SIZE 128
#define GROUP_SIZE 8

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

typedef struct {
  size_t offset;
  ai_brain_t *brains;
  int results[GROUP_SIZE][GROUP_SIZE];
} thread_data_t;

typedef struct {
  size_t index;
  int score;
} ranked_brain_t;

int
compare_moves(const void *a, const void *b)
{
  const scored_move_t *move_a = a;
  const scored_move_t *move_b = b;
  if (move_a->score > move_b->score) return -1;
  if (move_a->score < move_b->score) return 1;
  return 0;
}

int
compare_ranked_brains(const void *a, const void *b)
{
  const ranked_brain_t *brain_a = a;
  const ranked_brain_t *brain_b = b;
  if (brain_a->score > brain_b->score) return -1;
  if (brain_a->score < brain_b->score) return 1;
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
predict_next_move(move_t *res, chess_game_t *game, ai_brain_t *brain, chess_color_t color)
{
  size_t i, src_index, dst_index;
  float inputs[768];
  scored_move_t scored_moves[4096];
  chess_move_t move;

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

    res->src_x = src_index / 8;
    res->src_y = src_index % 8;
    res->dst_x = dst_index / 8;
    res->dst_y = dst_index % 8;

    move = chess_safe_move(game, res->src_x, res->src_y, res->dst_x, res->dst_y);
    if (color == chess_color_from_square(game->board[res->src_y][res->src_x]) && CHESS_MOVE_ILLEGAL != move && CHESS_MOVE_UNSAFE != move) {
      return 0;
    }
  }

  return -1;
}

int
game_loop(chess_game_t *game, ai_brain_t *a, ai_brain_t *b)
{
  size_t i;
  chess_color_t c;
  int has_prediction, score = 0;
  move_t move;
  chess_move_t move_data;
  uint8_t piece_value, until_stalemate = 0;
  size_t total_moves = 0;
  // char src[3], dst[3];

  for (;;) {
    // if ('q' == fgetc(stdin)) {
    //   return;
    // }

    for (i = 0; i < 2; i++)  {
      // printf("computing\n");
      c = (i + 1) % 2;
      has_prediction = predict_next_move(&move, game, i == 0 ? a : b, c);
      move_data = chess_legal_move(game, move.src_x, move.src_y, move.dst_x, move.dst_y);

      // chess_pretty_square(src, move.src_x, move.src_y);
      // chess_pretty_square(dst, move.dst_x, move.dst_y);

      // printf("%ld: %s -> %s\n", i, src, dst);

      switch (move_data) {
      case CHESS_MOVE_TAKE:
        until_stalemate = 0;

        switch (chess_piece_from_square(game->board[move.dst_y][move.dst_x])) {
        case CHESS_PIECE_PAWN:
          piece_value = 1;
          break;
        case CHESS_PIECE_BISHOP:
        case CHESS_PIECE_KNIGHT:
          piece_value = 3;
          break;
        case CHESS_PIECE_ROOK:
          piece_value = 5;
          break;
        case CHESS_PIECE_QUEEN:
          piece_value = 9;
          break;
        default:
          break;
        }

        score += (CHESS_COLOR_WHITE == c ? 1 : -1) * piece_value;
        break;
      case CHESS_MOVE_TAKE_ENPASSANT:
        until_stalemate = 0;

        score += (CHESS_COLOR_WHITE == c ? 1 : -1);
        break;
      default:
        until_stalemate++;
        break;
      }

      if (40 <= until_stalemate) {
        return score;
      }

      if (-1 == has_prediction) {
        score += (CHESS_COLOR_WHITE == c ? -50 : 50);
        return score;
      }

      chess_do_move(game, move.src_x, move.src_y, move.dst_x, move.dst_y);
    }

    total_moves += 2;
    if (total_moves > 2048) {
      printf("exceeded max moves\n");
      return 0;
    }
  }
}

void *
training_thread(void *arg)
{
  thread_data_t *data = arg;
  size_t i, j;
  ai_brain_t a, b;
  chess_game_t game;

  for (i = 0; i < GROUP_SIZE; i++) {
    for (j = 0; j < GROUP_SIZE; j++) {
      if (i == j) {
        continue;
      }

      a = data->brains[i + data->offset * GROUP_SIZE];
      b = data->brains[j + data->offset * GROUP_SIZE];
      chess_init(&game);


      data->results[i][j] = game_loop(&game, &a, &b);
      printf("game over (%ld,%ld: %d)\n", i + data->offset * GROUP_SIZE, j + data->offset * GROUP_SIZE, data->results[i][j]);
    }
  }

  return 0;
}

int
main(int argc, char **argv)
{
  int start_gen, stop_gen;
  ai_brain_t brains[POPULATION_SIZE];
  pthread_t threads[POPULATION_SIZE / GROUP_SIZE];
  thread_data_t data[POPULATION_SIZE / GROUP_SIZE];
  ranked_brain_t ranked_brains[POPULATION_SIZE];
  size_t i, j, k, parent_a, parent_b;
  char filename[32];

  start_gen = argc <= 1 ? 0 : atoi(argv[1]);
  stop_gen = argc <= 2 ? start_gen + 10 : atoi(argv[2]);

  if (0 < start_gen) {
    for (i = 0; i < 8; i++) {
      snprintf(filename, sizeof(filename), "models/%d-%ld.model", start_gen, i);
      printf("loading brain: %s\n", filename);
      ai_brain_load(&brains[i], filename);
    }

    printf("loaded gen %d brains\n", start_gen);

    for (i = 8; i < POPULATION_SIZE; i++) {
      do {
        parent_a = rand() % 8;
        parent_b = rand() % 8;
      } while (parent_a != parent_b);
      init_chess_brain(&brains[i]);
      ai_brain_offspring(&brains[parent_a], &brains[parent_b], &brains[i]);
    }
  } else {
    for (i = 0; i < POPULATION_SIZE; i++) {
      init_chess_brain(&brains[i]);
    }
  }

  while (start_gen < stop_gen) {
    printf("-- GENERATION %d/%d --\n", start_gen, stop_gen);
    for (i = 0; i < POPULATION_SIZE / GROUP_SIZE; i++) {
      data[i].brains = brains;
      data[i].offset = i;
      memset(data[i].results, 0, 8 * 8 * sizeof(int));
      pthread_create(&threads[i], NULL, training_thread, &data[i]);
    }

    for (i = 0; i < POPULATION_SIZE / GROUP_SIZE; i++) {
      pthread_join(threads[i], NULL);
    }

    printf("done with groups!\n");

    for (i = 0; i < POPULATION_SIZE / GROUP_SIZE; i++) {
      for (j = 0; j < GROUP_SIZE; j++) {
        ranked_brains[j + i * GROUP_SIZE].index = j + i * GROUP_SIZE;
        ranked_brains[j + i * GROUP_SIZE].score = 0;
        for (k = 0; k < GROUP_SIZE; k++) {
          ranked_brains[j + i * GROUP_SIZE].score += data[i].results[j][k];
          ranked_brains[j + i * GROUP_SIZE].score += -data[i].results[k][j];
        }
      }
    }

    qsort(ranked_brains, POPULATION_SIZE, sizeof(ranked_brain_t), compare_ranked_brains);

    start_gen++;

    for (i = 0; i < 8; i++) {
      snprintf(filename, 32, "models/%d-%ld.model", start_gen, i);
      printf("saving %ld\n", ranked_brains[i].index);
      ai_brain_save(&brains[ranked_brains[i].index], filename);
    }

    printf("saved elite brains\n");

    for (i = 8; i < POPULATION_SIZE; i++) {
      do {
        parent_a = rand() % 8;
        parent_b = rand() % 8;
      } while (parent_a != parent_b);
      ai_brain_offspring(&brains[ranked_brains[parent_a].index], &brains[ranked_brains[parent_b].index], &brains[ranked_brains[i].index]);
    }
  }

  printf("done\n");

  for (i = 0; i < POPULATION_SIZE; i++) {
    ai_brain_free(&brains[i]);
  }

  // srand(time(NULL));
}
