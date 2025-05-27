#include <stdio.h>
#include <stdlib.h>

#include "game.h"

#define ASSERT_EQ(a, b, fmt) \
  if ((a) != (b)) { \
    fprintf(stderr, "expected to be '" fmt "', found '" fmt "'\n", a, b); \
    exit(1); \
  }

#define ASSERT_INT_EQ(a, b) ASSERT_EQ(a, b, "%d")

#define ASSERT(bool) \
  if (!(bool)) { \
    fprintf(stderr, "bool is false\n"); \
    exit(1); \
  }

void
test_pawn()
{
  chess_board_t board;
  chess_init_board(board);

  ASSERT(chess_legal_move(board, 3, 6, 3, 5));
  ASSERT(chess_legal_move(board, 3, 6, 3, 4));
}

int
main()
{
  test_pawn();

  fprintf(stderr, "ran tests successfully\n");
}
