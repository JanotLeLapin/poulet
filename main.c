#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "poulet.h"

#define POPULATION_SIZE 256
#define GROUP_SIZE 16
#define ELITE_SIZE 8

typedef struct {
  uint8_t src_x;
  uint8_t src_y;
  uint8_t dst_x;
  uint8_t dst_y;
} move_t;

typedef struct {
  size_t offset;
  ai_brain_t *brains;
  float results[GROUP_SIZE][GROUP_SIZE][2];
} thread_data_t;

typedef struct {
  size_t index;
  float score;
} ranked_brain_t;

volatile sig_atomic_t running = 1;

void
handle_sigint(int sig)
{
  if (SIGINT == sig) {
    running = 0;
    printf("got sigint, stopping after this generation\n");
  }
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

static inline void
normalize_scores(float *scores, size_t total_moves)
{
  scores[0] /= total_moves;
  scores[1] /= total_moves;
}

void
game_loop(float *scores, chess_game_t *game, ai_brain_t *a, ai_brain_t *b)
{
  size_t i;
  chess_color_t c;
  int has_prediction;
  uint8_t move[4], piece_value, until_stalemate = 0;
  chess_move_t move_data;
  size_t total_moves = 0;
  // char src[3], dst[3];

  for (;;) {
    // if ('q' == fgetc(stdin)) {
    //   return;
    // }

    for (i = 0; i < 2; i++)  {
      // printf("computing\n");
      c = (i + 1) % 2;
      has_prediction = poulet_next_move(move, game, i == 0 ? a : b, c, 1.2f);
      move_data = chess_legal_move(game, move[1], move[0], move[3], move[2]);

      // chess_pretty_square(src, move[1], move[0]);
      // chess_pretty_square(dst, move[3], move[2]);

      // printf("%ld: %s -> %s\n", i, src, dst);

      switch (move_data) {
      case CHESS_MOVE_TAKE:
        until_stalemate = 0;

        switch (chess_piece_from_square(game->board[move[2]][move[3]])) {
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

        scores[c] += piece_value;
        scores[i] -= piece_value;
        break;
      case CHESS_MOVE_TAKE_ENPASSANT:
        until_stalemate = 0;

        scores[c]++;
        scores[i]--;
        break;
      default:
        until_stalemate = CHESS_PIECE_PAWN == chess_piece_from_square(game->board[move[0]][move[1]]) ? 0 : until_stalemate + 1;
        break;
      }

      if (50 <= until_stalemate || total_moves > 2048) {
        if (scores[c] > scores[i]) {
          scores[c] -= 100;
          scores[i] += 100;
        } else if (scores[c] < scores[i]) {
          scores[c] += 100;
          scores[i] -= 100;
        }
        normalize_scores(scores, total_moves);
        return;
      }

      if (-1 == has_prediction) {
        scores[c] -= 1000;
        scores[i] += 1000;
        normalize_scores(scores, total_moves);
        return;
      }

      chess_do_move(game, move[1], move[0], move[3], move[2]);

      total_moves++;
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

      game_loop(data->results[i][j], &game, &a, &b);
      printf("game over (%ld,%ld: white: %f, black: %f)\n", i + data->offset * GROUP_SIZE, j + data->offset * GROUP_SIZE, data->results[i][j][0], data->results[i][j][1]);
    }
  }

  return 0;
}

int
main(int argc, char **argv)
{
  struct sigaction act;
  int start_gen, stop_gen;
  ai_brain_t brains[POPULATION_SIZE], tmp_brain;
  pthread_t threads[POPULATION_SIZE / GROUP_SIZE];
  thread_data_t data[POPULATION_SIZE / GROUP_SIZE];
  ranked_brain_t ranked_brains[POPULATION_SIZE];
  size_t i, j, k, parent_a, parent_b;
  char filename[32];

  act.sa_handler = handle_sigint;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESTART;

  sigaction(SIGINT, &act, NULL);

  start_gen = argc <= 1 ? 0 : atoi(argv[1]);
  stop_gen = argc <= 2 ? start_gen + 10 : atoi(argv[2]);

  srand(time(0));

  if (0 < start_gen) {
    for (i = 0; i < ELITE_SIZE; i++) {
      snprintf(filename, sizeof(filename), "models/%d-%ld.model", start_gen, i);
      printf("loading brain: %s\n", filename);
      ai_brain_load(&brains[i], filename);
    }

    printf("loaded gen %d brains\n", start_gen);

    for (i = ELITE_SIZE; i < POPULATION_SIZE; i++) {
      do {
        parent_a = rand() % ELITE_SIZE;
        parent_b = rand() % ELITE_SIZE;
      } while (parent_a != parent_b);
      poulet_brain_init(&brains[i]);
      ai_brain_offspring(&brains[parent_a], &brains[parent_b], &brains[i]);
    }
  } else {
    for (i = 0; i < POPULATION_SIZE; i++) {
      poulet_brain_init(&brains[i]);
    }
  }

  while (start_gen < stop_gen && running) {
    printf("-- GENERATION %d/%d --\n", start_gen, stop_gen);
    for (i = 0; i < POPULATION_SIZE / GROUP_SIZE; i++) {
      data[i].brains = brains;
      data[i].offset = i;
      memset(data[i].results, 0, sizeof(int) * 2 * GROUP_SIZE * GROUP_SIZE);
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
          if (k == j) {
            continue;
          }
          ranked_brains[j + i * GROUP_SIZE].score += data[i].results[j][k][0];
          ranked_brains[j + i * GROUP_SIZE].score += data[i].results[k][j][1];
        }
      }
    }

    qsort(ranked_brains, POPULATION_SIZE, sizeof(ranked_brain_t), compare_ranked_brains);

    start_gen++;

    if (start_gen % 5 == 0) {
      printf("saving elite\n");

      for (i = 0; i < ELITE_SIZE; i++) {
        snprintf(filename, sizeof(filename), "models/%d-%ld.model", start_gen, i);
        printf("saving %ld\n", ranked_brains[i].index);
        ai_brain_save(&brains[ranked_brains[i].index], filename);
      }

      printf("saved elite brains\n");
    }

    printf("generating offspring\n");
    for (i = ELITE_SIZE; i < POPULATION_SIZE; i++) {
      do {
        parent_a = rand() % ELITE_SIZE;
        parent_b = rand() % ELITE_SIZE;
      } while (parent_a != parent_b);
      ai_brain_offspring(&brains[ranked_brains[parent_a].index], &brains[ranked_brains[parent_b].index], &brains[ranked_brains[i].index]);
    }

    printf("shuffling new brains\n");
    for (i = POPULATION_SIZE - 1; i > 0; i--) {
      j = rand() % (i + 1);

      tmp_brain = brains[i];
      brains[i] = brains[j];
      brains[j] = tmp_brain;
    }
  }

  printf("done\n");

  for (i = 0; i < POPULATION_SIZE; i++) {
    ai_brain_free(&brains[i]);
  }
}
