#include <stdint.h>
#include <stdlib.h>

#include "game.h"

void
chess_init_board(chess_board_t board)
{
  uint8_t i, j;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      board[i * 7][j * 7] = chess_new_square(CHESS_PIECE_ROOK, (chess_color_t) i);
      board[i * 7][j * 5 + 1] = chess_new_square(CHESS_PIECE_KNIGHT, (chess_color_t) i);
      board[i * 7][j * 3 + 2] = chess_new_square(CHESS_PIECE_BISHOP, (chess_color_t) i);
    }
    board[i * 7][3] = chess_new_square(CHESS_PIECE_QUEEN, (chess_color_t) i);
    board[i * 7][4] = chess_new_square(CHESS_PIECE_KING, (chess_color_t) i);
  }

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 8; j++) {
      board[i * 5 + 1][j] = chess_new_square(CHESS_PIECE_PAWN, (chess_color_t) i);
    }
  }

  for (i = 2; i < 6; i++) {
    for (j = 0; j < 8; j++) {
      board[i][j] = 0;
    }
  }
}

static int
pawn_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s = board[ay][ax];
  chess_color_t c = chess_color_from_square(s);
  char direction = c * -2 + 1;

  if (0 == board[by][bx]) {
    // implement en passant later
    return bx == ax && (by == ay + direction || ((c == CHESS_COLOR_WHITE ? 6 : 1) == ay && by == ay + direction * 2 && 0 == board[ay + direction][ax]));
  } else {
    return c != chess_color_from_square(board[by][bx]) && 1 == abs(ax - bx) && by == ay + direction;
  }
}

static int
bishop_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s, target;
  char vx, vy, i;

  if (abs(bx - ax) != abs(by - ay)) {
    return 0;
  }

  s = board[ay][ax];
  target = board[by][bx];
  if (0 != target && chess_color_from_square(target) == chess_color_from_square(s)) {
    return 0;
  }

  vx = bx < ax ? -1 : 1;
  vy = by < ay ? -1 : 1;
  for (i = 1; i < abs(bx - ax); i++) {
    if (0 != board[ay + vy][ax + vx]) {
      return 0;
    }
  }

  return 1;
}

static int
knight_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s, target;
  int dx;
  int dy;

  s = board[ay][ax];
  target = board[by][bx];

  if (0 != target && chess_color_from_square(target) == chess_color_from_square(s)) {
    return 0;
  }

  dx = abs(bx - ax);
  dy = abs(by - ay);

  return 2 == dx && 1 == dy || 2 == dy && 1 == dx;
}

static int
rook_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s, target;
  char step, i;

  if (bx != ax && by != ay) {
    return 0;
  }

  s = board[ay][ax];
  target = board[by][bx];
  if (0 != target && chess_color_from_square(target) == chess_color_from_square(s)) {
    return 0;
  }

  if (bx == ax) {
    step = by < ay ? -1 : 1;
    for (i = 1; i < abs(by - ay); i++) {
      if (0 != board[ay + step * i][ax]) {
        return 0;
      }
    }
  } else {
    step = bx < ax ? -1 : 1;
    for (i = 1; i < abs(bx - ax); i++){
      if (0 != board[ay][ax + step * i]) {
        return 0;
      }
    }
  }

  return 1;
}

int
chess_legal_move(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t a = board[ay][ax];

  if (ax > 7 || ay > 7 || bx > 7 || by > 7) {
    return 0;
  }

  switch (chess_piece_from_square(a)) {
  case CHESS_PIECE_PAWN:
    return pawn_legal(board, ax, ay, bx, by);
  case CHESS_PIECE_BISHOP:
    return bishop_legal(board, ax, ay, bx, by);
  case CHESS_PIECE_KNIGHT:
    return knight_legal(board, ax, ay, bx, by);
  case CHESS_PIECE_ROOK:
    return rook_legal(board, ax, ay, bx, by);
  default:
    return 0;
  }
}
