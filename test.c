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
    LOG_ERROR("expected: " #b " to be: " #a " (" fmt "), found: " fmt, a, b); \
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
  chess_game_t game;

  LOG_INFO("testing pawn");

  empty_board(game.board);
  game.meta = 0;
  game.board[5][0] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_BLACK);
  game.board[6][0] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  game.board[6][1] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 0, 6, 0, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 1, 6, 1, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 1, 6, 1, 5));
  ASSERT_INT_EQ(CHESS_MOVE_TAKE, chess_legal_move(&game, 1, 6, 0, 5));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 1, 6, 2, 5));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 1, 6, 3, 4));

  game.board[6][0] = 0;
  game.board[5][0] = 0;
  game.board[4][1] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  game.board[4][0] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  game.meta |= 0x01;
  ASSERT_INT_EQ(CHESS_MOVE_TAKE_ENPASSANT, chess_legal_move(&game, 1, 4, 0, 5));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 1, 4, 2, 5));
}

void
test_bishop()
{
  chess_game_t game;

  LOG_INFO("testing bishop");

  empty_board(game.board);
  game.meta = 0;
  game.board[3][2] = chess_new_square(CHESS_PIECE_BISHOP, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 5));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 3, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 0, 5));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 1, 2));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 1));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 3, 2));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 7, 1));

  game.board[4][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 4, 5));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 3, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 0, 5));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 1, 2));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 1));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 7, 1));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 3, 2, 5, 4));

  game.board[4][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 4, 5));
  ASSERT_INT_EQ(CHESS_MOVE_TAKE, chess_legal_move(&game, 2, 3, 3, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 0, 5));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 1, 2));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 1));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 7, 1));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 3, 2, 5, 4));
}

void
test_knight()
{
  chess_game_t game;

  LOG_INFO("testing knight");

  empty_board(game.board);
  game.meta = 0;
  game.board[3][2] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 0, 2));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 0, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 2));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 2, 3, 4, 4));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 4, 5));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 2, 6));

  game.board[4][4] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 2, 3, 4, 4));

  game.board[4][4] = chess_new_square(CHESS_PIECE_KNIGHT, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_TAKE, chess_legal_move(&game, 2, 3, 4, 4));
}

void
test_rook()
{
  chess_game_t game;

  LOG_INFO("testing rook");

  empty_board(game.board);
  game.meta = 0;
  game.board[4][3] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 3, 7));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 2, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 7, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 3, 1));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 3, 4, 4, 3));

  game.board[6][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 6, 4));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 3, 4, 3, 5));
  ASSERT_INT_EQ(CHESS_MOVE_TAKE, chess_legal_move(&game, 3, 4, 3, 6));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 3, 4, 3, 7));

  game.board[6][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 3, 4, 3, 6));
}

void
test_check()
{
  chess_game_t game;

  LOG_INFO("testing check");

  empty_board(game.board);
  game.meta = 0;
  game.board[2][2] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_WHITE);
  game.board[4][5] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_BLACK);
  ASSERT(!chess_is_check(&game, CHESS_COLOR_WHITE));
  ASSERT(!chess_is_check(&game, CHESS_COLOR_BLACK));

  game.board[2][4] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_BLACK);
  ASSERT(chess_is_check(&game, CHESS_COLOR_WHITE));
  ASSERT(!chess_is_check(&game, CHESS_COLOR_BLACK));

  game.board[7][2] = chess_new_square(CHESS_PIECE_BISHOP, CHESS_COLOR_WHITE);
  ASSERT(chess_is_check(&game, CHESS_COLOR_WHITE));
  ASSERT(chess_is_check(&game, CHESS_COLOR_BLACK));
}

void
test_safe()
{
  chess_game_t game;

  LOG_INFO("testing safe");

  empty_board(game.board);
  game.meta = 0;
  game.board[4][0] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_WHITE);
  game.board[7][7] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_BLACK);
  game.board[4][1] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  game.board[4][4] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_BLACK);
  game.board[4][7] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_safe_move(&game, 7, 4, 7, 3));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_safe_move(&game, 7, 4, 7, 2));
  ASSERT_INT_EQ(CHESS_MOVE_UNSAFE, chess_safe_move(&game, 1, 4, 1, 3));

  game.board[4][1] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_safe_move(&game, 7, 4, 7, 3));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_safe_move(&game, 1, 4, 1, 5));

  empty_board(game.board);
  game.board[3][1] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_WHITE);
  game.board[3][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  game.board[3][4] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  game.board[3][7] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_BLACK);
  game.meta = (4 << 1) | 1;
  ASSERT_INT_EQ(CHESS_MOVE_UNSAFE, chess_safe_move(&game, 3, 3, 4, 2));

  game.board[3][7] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_TAKE_ENPASSANT, chess_safe_move(&game, 3, 3, 4, 2));

  empty_board(game.board);
  game.board[4][1] = chess_new_square(CHESS_PIECE_KING, CHESS_COLOR_BLACK);
  game.board[4][3] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  game.board[4][4] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_WHITE);
  game.board[4][7] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_WHITE);
  game.meta |= (4 << 1) | 1;
  ASSERT_INT_EQ(CHESS_MOVE_UNSAFE, chess_safe_move(&game, 3, 4, 4, 5));

  game.board[4][7] = chess_new_square(CHESS_PIECE_PAWN, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_TAKE_ENPASSANT, chess_safe_move(&game, 3, 4, 4, 5));
}

void
test_castle()
{
  chess_game_t game;

  LOG_INFO("testing castle");

  chess_init(&game);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[7][5] = 0;
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[7][6] = 0;
  ASSERT_INT_EQ(CHESS_MOVE_CASTLE, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[6][4] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[6][4] = 0;
  game.board[6][5] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_BLACK);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[6][5] = 0;
  game.board[7][7] = 0;
  game.board[7][6] = chess_new_square(CHESS_PIECE_ROOK, CHESS_COLOR_WHITE);
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  chess_init(&game);
  game.board[7][5] = 0;
  game.board[7][6] = 0;
  game.board[7][7] = 0;
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 2, 7));

  game.board[7][3] = 0;
  game.board[7][2] = 0;
  game.board[7][1] = 0;
  ASSERT_INT_EQ(CHESS_MOVE_ILLEGAL, chess_legal_move(&game, 4, 7, 6, 7));
  ASSERT_INT_EQ(CHESS_MOVE_CASTLE, chess_legal_move(&game, 4, 7, 2, 7));
}

void
test_main()
{
  chess_game_t game;

  LOG_INFO("testing main");

  chess_init(&game);
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 4, 1, 4, 3));
  ASSERT_INT_EQ(CHESS_MOVE_LEGAL, chess_legal_move(&game, 4, 6, 4, 4));
}

int
main()
{
  test_pawn();
  test_bishop();
  test_knight();
  test_rook();

  test_check();
  test_safe();
  test_castle();

  test_main();

  LOG_INFO("successfully ran test cases");
}
