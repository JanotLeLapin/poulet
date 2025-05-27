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
pawn_legal(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s = game->board[ay][ax];
  chess_color_t c = chess_color_from_square(s);
  char direction = c * -2 + 1;

  if (0 == game->board[by][bx]) {
    // implement en passant later
    return bx == ax && (by == ay + direction || ((c == CHESS_COLOR_WHITE ? 6 : 1) == ay && by == ay + direction * 2 && 0 == game->board[ay + direction][ax]));
  } else {
    return c != chess_color_from_square(game->board[by][bx]) && 1 == abs(ax - bx) && by == ay + direction;
  }
}

static int
bishop_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  char vx, vy, i;

  if (abs(bx - ax) != abs(by - ay)) {
    return 0;
  }

  vx = bx < ax ? -1 : 1;
  vy = by < ay ? -1 : 1;
  for (i = 1; i < abs(bx - ax); i++) {
    if (0 != board[ay + vy * i][ax + vx * i]) {
      return 0;
    }
  }

  return 1;
}

static int
knight_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  char dx, dy;

  dx = abs(bx - ax);
  dy = abs(by - ay);

  return 2 == dx && 1 == dy || 2 == dy && 1 == dx;
}

static int
rook_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  char step, i;

  if (bx != ax && by != ay) {
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

static int
queen_legal(chess_board_t board, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  return bishop_legal(board, ax, ay, bx, by) || rook_legal(board, ax, ay, bx, by);
}

static int
king_legal(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  return abs(bx - ax) <= 1 && abs(by - ay) <= 1;
}

static uint16_t
find_king(chess_board_t board, chess_color_t color)
{
  uint16_t res = 0;
  uint8_t i, j;

  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      if (CHESS_PIECE_KING == chess_piece_from_square(board[i][j]) && color == chess_color_from_square(board[i][j])) {
        return (j << 8) | i;
      }
    }
  }

  return 0;
}

int
chess_is_check(chess_game_t *game, chess_color_t color)
{
  char res = 0;
  uint16_t king_pos;
  uint8_t i, j, kx, ky;

  king_pos = find_king(game->board, color);
  kx = (king_pos >> 8) & 0xFF;
  ky = king_pos & 0xFF;

  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      if (0 != game->board[i][j] && color != chess_color_from_square(game->board[i][j]) && chess_legal_move(game, j, i, kx, ky)) {
        return 1;
      }
    }
  }

  return 0;
}

int
chess_legal_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t a, b;

  a = game->board[ay][ax];
  b = game->board[by][bx];

  if (
    (ax > 7 || ay > 7 || bx > 7 || by > 7)
    || ((ax == bx) && (ay == by))
    || (0 != b && chess_color_from_square(b) == chess_color_from_square(a))
  ) {
    return 0;
  }

  switch (chess_piece_from_square(a)) {
  case CHESS_PIECE_PAWN:
    return pawn_legal(game, ax, ay, bx, by);
  case CHESS_PIECE_BISHOP:
    return bishop_legal(game->board, ax, ay, bx, by);
  case CHESS_PIECE_KNIGHT:
    return knight_legal(game->board, ax, ay, bx, by);
  case CHESS_PIECE_ROOK:
    return rook_legal(game->board, ax, ay, bx, by);
  case CHESS_PIECE_QUEEN:
    return queen_legal(game->board, ax, ay, bx, by);
  case CHESS_PIECE_KING:
    return king_legal(game, ax, ay, bx, by);
  default:
    return 0;
  }
}

int
chess_safe_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t a, b;
  chess_color_t c;
  int res;

  if (!chess_legal_move(game, ax, ay, bx, by)) {
    return 0;
  }

  a = game->board[ay][ax];
  b = game->board[by][bx];
  c = chess_color_from_square(a);

  game->board[ay][ax] = 0;
  game->board[by][bx] = a;

  res = !chess_is_check(game, c);
  game->board[ay][ax] = a;
  game->board[by][bx] = b;

  return res;
}
