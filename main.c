#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "poulet.h"

#define POPULATION_SIZE 256
#define GROUP_SIZE 16
#define GAME_COUNT 32
#define MATCH_COUNT (POPULATION_SIZE * GAME_COUNT) / 2
#define THREAD_COUNT 16
#define ELITE_SIZE 8

const int HEAT_MAP[8][8] = {
  { 0, 0, 1, 2, 2, 1, 0, 0 },
  { 0, 1, 2, 3, 3, 2, 1, 0 },
  { 1, 2, 3, 4, 4, 3, 2, 1 },
  { 2, 3, 4, 5, 5, 4, 3, 2 },
  { 2, 3, 4, 5, 5, 4, 3, 2 },
  { 1, 2, 3, 4, 4, 3, 2, 1 },
  { 0, 1, 2, 3, 3, 2, 1, 0 },
  { 0, 0, 1, 2, 2, 1, 0, 0 },
};

typedef struct {
  size_t brain_a;
  size_t brain_b;
  float scores[2];
} match_t;

typedef struct {
  uint8_t src_x;
  uint8_t src_y;
  uint8_t dst_x;
  uint8_t dst_y;
} move_t;

typedef struct {
  size_t start;
  size_t end;
  match_t *matches;
  ai_brain_t *brains;
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

void
init_matches(match_t *matches)
{
  size_t match_count = 0, brain_matches[POPULATION_SIZE], brain_a, brain_b;

  memset(brain_matches, 0, POPULATION_SIZE * sizeof(size_t));
  while (match_count < MATCH_COUNT) {
    do {
      brain_a = rand() % POPULATION_SIZE;
      brain_b = rand() % POPULATION_SIZE;
    } while (brain_matches[brain_a] >= GAME_COUNT || brain_matches[brain_b] >= GAME_COUNT);

    matches[match_count].brain_a = brain_a;
    matches[match_count].brain_b = brain_b;
    matches[match_count].scores[0] = 0;
    matches[match_count].scores[1] = 0;
    brain_matches[brain_a]++;
    brain_matches[brain_b]++;
    match_count++;
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

void
game_loop(float *scores, chess_game_t *game, ai_brain_t *a, ai_brain_t *b)
{
  size_t i;
  chess_color_t c;
  int has_prediction, is_check;
  uint8_t move[4], piece_value, until_stalemate = 0;
  chess_move_t move_data;
  chess_piece_t piece;
  size_t total_moves = 0;
  // char src[3], dst[3];

  for (;;) {
    // if ('q' == fgetc(stdin)) {
    //   return;
    // }

    for (i = 0; i < 2; i++)  {
      // printf("computing\n");
      c = 1 - i;
      has_prediction = poulet_next_move(move, game, i == 0 ? a : b, c, 0.8f);
      is_check = chess_is_check(game, c);

      if ((-1 == has_prediction && !is_check) || (50 <= until_stalemate || total_moves > 2048)) {
        if (scores[c] > scores[i]) {
          scores[c] -= 10;
          scores[i] += 10;
        } else if (scores[c] < scores[i]) {
          scores[c] += 10;
          scores[i] -= 10;
        }
        return;
      }

      if (-1 == has_prediction) {
        scores[c] -= 1000;
        scores[i] += 1000;
        return;
      }

      scores[c] += (total_moves < 40 ? 1.0f : 0.3f) * (((float) HEAT_MAP[move[2]][move[3]]) / 24.0f);
      move_data = chess_legal_move(game, move[1], move[0], move[3], move[2]);

      // chess_pretty_square(src, move[1], move[0]);
      // chess_pretty_square(dst, move[3], move[2]);

      // printf("%ld: %s -> %s\n", i, src, dst);

      piece = chess_piece_from_square(game->board[move[1]][move[0]]);

      if (CHESS_PIECE_PAWN == piece && (c == CHESS_COLOR_WHITE ? 0 : 7) == move[2]) {
        scores[c] += 8;
        scores[i] -= 8;
      }

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
          piece_value = 0;
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
        until_stalemate = CHESS_PIECE_PAWN == piece ? 0 : until_stalemate + 1;
        break;
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
  size_t i;
  ai_brain_t *a, *b;
  chess_game_t game;

  for (i = data->start; i < data->end; i++) {
    a = &data->brains[data->matches[i].brain_a];
    b = &data->brains[data->matches[i].brain_b];

    chess_init(&game);
    game_loop(data->matches[i].scores, &game, a, b);
    printf("game over (%ld vs %ld: white: %f, black: %f)\n", data->matches[i].brain_a, data->matches[i].brain_b, data->matches[i].scores[0], data->matches[i].scores[1]);
  }

  return 0;
}

int
main(int argc, char **argv)
{
  struct sigaction act;
  int start_gen, stop_gen;
  match_t matches[POPULATION_SIZE * GAME_COUNT];
  ai_brain_t brains[POPULATION_SIZE], tmp_brain;
  pthread_t threads[POPULATION_SIZE / GROUP_SIZE];
  thread_data_t data[POPULATION_SIZE / GROUP_SIZE];
  ranked_brain_t ranked_brains[POPULATION_SIZE];
  size_t i, j, parent_a, parent_b;
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
    init_matches(matches);
    for (i = 0; i < THREAD_COUNT; i++) {
      data[i].brains = brains;
      data[i].start = i * (MATCH_COUNT / THREAD_COUNT);
      data[i].end = (i + 1) * (MATCH_COUNT / THREAD_COUNT);
      data[i].matches = matches;
      pthread_create(&threads[i], NULL, training_thread, &data[i]);
    }

    for (i = 0; i < THREAD_COUNT; i++) {
      pthread_join(threads[i], NULL);
    }

    printf("done with groups!\n");

    for (i = 0; i < POPULATION_SIZE; i++) {
      ranked_brains[i].index = i;
      ranked_brains[i].score = 0;
    }

    for (i = 0; i < MATCH_COUNT; i++) {
      ranked_brains[matches[i].brain_a].score += matches[i].scores[0];
      ranked_brains[matches[i].brain_b].score += matches[i].scores[1];
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
