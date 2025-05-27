#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game.h"

#define COLOR_RESET "\x1b[0m"
#define COLOR_INFO  "\x1b[32m"
#define COLOR_ERROR "\x1b[31m"

#define LOG(color, tag, fmt, ...) \
  fprintf(stderr, "[" color tag COLOR_RESET ":%s:%d]: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
  LOG(COLOR_ERROR, "error", fmt, ##__VA_ARGS__)

#define LOG_INFO(fmt, ...) \
  LOG(COLOR_INFO, "info", fmt, ##__VA_ARGS__)

#define ASSERT_EQ(a, b, fmt) \
  if ((a) != (b)) { \
    LOG_ERROR("expected '" #b "' to be '" fmt "', found '" fmt "'", a, b); \
    exit(1); \
  }

#define ASSERT_INT_EQ(a, b) ASSERT_EQ(a, b, "%d")

#define ASSERT(bool) \
  if (!(bool)) { \
    LOG_ERROR("assertion failed: " #bool); \
    exit(1); \
  }

static inline void
empty_board(chess_board_t board)
{
  memset(board, 0, sizeof(chess_board_t));
}

void
test_pawn()
{
  chess_board_t board;

  LOG_INFO("testing pawn");

  empty_board(board);
  board[5][0] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_BLACK);
  board[6][0] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  board[6][1] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT(!chess_legal_move(board, 0, 6, 0, 4));
  ASSERT(chess_legal_move(board, 1, 6, 1, 4));
  ASSERT(chess_legal_move(board, 1, 6, 1, 5));
  ASSERT(chess_legal_move(board, 1, 6, 0, 5));
  ASSERT(!chess_legal_move(board, 1, 6, 2, 5));
  ASSERT(!chess_legal_move(board, 1, 6, 3, 4));
}

void
test_bishop()
{
  chess_board_t board;

  LOG_INFO("testing bishop");

  empty_board(board);
  board[3][2] = chess_new_square(CHESS_PIECE_BISHOP, CHESS_COLOR_BLACK);
  ASSERT(chess_legal_move(board, 2, 3, 4, 5));
  ASSERT(chess_legal_move(board, 2, 3, 3, 4));
  ASSERT(chess_legal_move(board, 2, 3, 0, 5));
  ASSERT(chess_legal_move(board, 2, 3, 1, 2));
  ASSERT(chess_legal_move(board, 2, 3, 4, 1));
  ASSERT(!chess_legal_move(board, 2, 3, 7, 1));

  board[4][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT(!chess_legal_move(board, 2, 3, 4, 5));
  ASSERT(!chess_legal_move(board, 2, 3, 3, 4));
  ASSERT(chess_legal_move(board, 2, 3, 0, 5));
  ASSERT(chess_legal_move(board, 2, 3, 1, 2));
  ASSERT(chess_legal_move(board, 2, 3, 4, 1));
  ASSERT(!chess_legal_move(board, 2, 3, 7, 1));

  board[4][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT(!chess_legal_move(board, 2, 3, 4, 5));
  ASSERT(chess_legal_move(board, 2, 3, 3, 4));
  ASSERT(chess_legal_move(board, 2, 3, 0, 5));
  ASSERT(chess_legal_move(board, 2, 3, 1, 2));
  ASSERT(chess_legal_move(board, 2, 3, 4, 1));
  ASSERT(!chess_legal_move(board, 2, 3, 7, 1));
}

void
test_knight()
{
  chess_board_t board;

  LOG_INFO("testing knight");

  empty_board(board);
  board[3][2] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_WHITE);
  ASSERT(chess_legal_move(board, 2, 3, 0, 2));
  ASSERT(chess_legal_move(board, 2, 3, 0, 4));
  ASSERT(chess_legal_move(board, 2, 3, 4, 4));
  ASSERT(chess_legal_move(board, 2, 3, 4, 2));
  ASSERT(chess_legal_move(board, 2, 3, 4, 4));
  ASSERT(!chess_legal_move(board, 2, 3, 4, 5));
  ASSERT(!chess_legal_move(board, 2, 3, 2, 6));

  board[4][4] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_WHITE);
  ASSERT(!chess_legal_move(board, 2, 3, 4, 4));

  board[4][4] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_BLACK);
  ASSERT(chess_legal_move(board, 2, 3, 4, 4));
}

int
main()
{
  test_pawn();
  test_bishop();
  test_knight();

  LOG_INFO("successfully ran test cases");
}
