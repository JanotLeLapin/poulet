#include <stdint.h>
#include <stdlib.h>

#include "game.h"

void
chess_init(chess_game_t *game)
{
  uint8_t i, j;

  game->meta = 0x03 << 5;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      game->board[i * 7][j * 7] = chess_new_square(CHESS_PIECE_ROOK, (chess_color_t) i);
      game->board[i * 7][j * 5 + 1] = chess_new_square(CHESS_PIECE_KNIGHT, (chess_color_t) i);
      game->board[i * 7][j * 3 + 2] = chess_new_square(CHESS_PIECE_BISHOP, (chess_color_t) i);
    }
    game->board[i * 7][3] = chess_new_square(CHESS_PIECE_QUEEN, (chess_color_t) i);
    game->board[i * 7][4] = chess_new_square(CHESS_PIECE_KING, (chess_color_t) i);
  }

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 8; j++) {
      game->board[i * 5 + 1][j] = chess_new_square(CHESS_PIECE_PAWN, (chess_color_t) i);
    }
  }

  for (i = 2; i < 6; i++) {
    for (j = 0; j < 8; j++) {
      game->board[i][j] = 0;
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
    if (bx == chess_get_enpassant_file(game->meta) && (int) c != chess_get_enpassant_color(game->meta) && by == (CHESS_COLOR_WHITE == c ? 2 : 5)) {
      return 1 == abs(ax - bx) && by == ay + direction ? CHESS_MOVE_TAKE_ENPASSANT : CHESS_MOVE_ILLEGAL;
    }
    return bx == ax && (by == ay + direction || ((c == CHESS_COLOR_WHITE ? 6 : 1) == ay && by == ay + direction * 2 && 0 == game->board[ay + direction][ax]));
  } else {
    return 1 == abs(ax - bx) && by == ay + direction;
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

  return (2 == dx && 1 == dy) || (2 == dy && 1 == dx);
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

static chess_move_t
king_legal(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t s, rook;
  chess_color_t c;
  char step;
  uint8_t i, can_castle = CHESS_MOVE_CASTLE;

  if (abs(bx - ax) <= 1 && abs(by - ay) <= 1) {
    return CHESS_MOVE_LEGAL;
  }

  s = game->board[ay][ax];
  c = chess_color_from_square(s);
  if (abs(bx - ax) == 2 && by == ay && chess_get_castling_rights(game->meta, c) && !chess_is_check(game, c)) {
    step = bx < ax ? -1 : 1;
    rook = game->board[ay][step == -1 ? 0 : 7];

    if (0 == rook || c != chess_color_from_square(rook) || CHESS_PIECE_ROOK != chess_piece_from_square(rook)) {
      return CHESS_MOVE_ILLEGAL;
    }

    for (i = 1; i <= 2; i++) {
      if (0 != game->board[ay][ax + step * i]) {
        can_castle = CHESS_MOVE_ILLEGAL;
        break;
      }

      game->board[ay][ax + step] = game->board[ay][ax + step - 1];
      game->board[ay][ax + step - 1] = 0;
      if (chess_is_check(game, c)) {
        can_castle = CHESS_MOVE_ILLEGAL;
        break;
      }
    }

    game->board[by][bx] = 0;
    game->board[ay][ax] = s;

    return can_castle;
  }

  return CHESS_MOVE_ILLEGAL;
}

static uint16_t
find_king(chess_board_t board, chess_color_t color)
{
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
  uint16_t king_pos;
  uint8_t i, j, kx, ky;

  king_pos = find_king(game->board, color);
  kx = (king_pos >> 8) & 0xFF;
  ky = king_pos & 0xFF;

  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      if (0 != game->board[i][j] && CHESS_PIECE_KING != chess_piece_from_square(game->board[i][j]) && color != chess_color_from_square(game->board[i][j]) && chess_legal_move(game, j, i, kx, ky)) {
        return 1;
      }
    }
  }

  return 0;
}

chess_move_t
chess_legal_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_square_t a, b;
  chess_move_t res;

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
    res = pawn_legal(game, ax, ay, bx, by);
    break;
  case CHESS_PIECE_BISHOP:
    res = bishop_legal(game->board, ax, ay, bx, by);
    break;
  case CHESS_PIECE_KNIGHT:
    res = knight_legal(game->board, ax, ay, bx, by);
    break;
  case CHESS_PIECE_ROOK:
    res = rook_legal(game->board, ax, ay, bx, by);
    break;
  case CHESS_PIECE_QUEEN:
    res = queen_legal(game->board, ax, ay, bx, by);
    break;
  case CHESS_PIECE_KING:
    res = king_legal(game, ax, ay, bx, by);
    break;
  default:
    return 0;
  }

  if (CHESS_MOVE_LEGAL == res && 0 != b) {
    return CHESS_MOVE_TAKE;
  }

  return res;
}

chess_move_t
chess_safe_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_move_t move;
  chess_square_t a, b;
  chess_color_t c;
  int is_check;
  uint8_t enpassant_y;

  move = chess_legal_move(game, ax, ay, bx, by);
  if (CHESS_MOVE_ILLEGAL == move) {
    return CHESS_MOVE_ILLEGAL;
  }

  a = game->board[ay][ax];
  b = game->board[by][bx];
  c = chess_color_from_square(a);

  switch (move) {
  case CHESS_MOVE_LEGAL:
  case CHESS_MOVE_TAKE:
    game->board[ay][ax] = 0;
    game->board[by][bx] = a;
    is_check = chess_is_check(game, c);
    game->board[ay][ax] = a;
    game->board[by][bx] = b;
    break;
  case CHESS_MOVE_TAKE_ENPASSANT:
    enpassant_y = by + (CHESS_COLOR_WHITE == c ? 1 : -1);
    b = game->board[enpassant_y][bx];
    game->board[ay][ax] = 0;
    game->board[by][bx] = a;
    game->board[enpassant_y][bx] = 0;
    is_check = chess_is_check(game, c);
    game->board[ay][ax] = a;
    game->board[by][bx] = 0;
    game->board[enpassant_y][bx] = b;
    break;
  case CHESS_MOVE_CASTLE:
  case CHESS_MOVE_UNSAFE:
  case CHESS_MOVE_ILLEGAL:
    return move;
  }

  return is_check ? CHESS_MOVE_UNSAFE : move;
}

void
chess_do_move(chess_game_t *game, uint8_t ax, uint8_t ay, uint8_t bx, uint8_t by)
{
  chess_move_t move;
  chess_color_t c;
  chess_piece_t p;
  uint8_t enpassant_y, rook_src_x, rook_dest_x;

  c = chess_color_from_square(game->board[ay][ax]);
  p = chess_piece_from_square(game->board[ay][ax]);

  if (CHESS_PIECE_KING == p) {
    game->meta = game->meta & ~(0x01 << (5 + c));
  }

  if ((int) c == chess_get_enpassant_color(game->meta)) {
    game->meta = game->meta & ~(0x1F);
  }

  if (CHESS_PIECE_PAWN == p && 2 == abs(ay - by)) {
    game->meta = (game->meta & (~0x1F)) | 0x01 | (c << 1) | (ax << 2);
  }

  move = chess_safe_move(game, ax, ay, bx, by);
  switch (move) {
  case CHESS_MOVE_LEGAL:
  case CHESS_MOVE_TAKE:
    game->board[by][bx] = game->board[ay][ax];
    game->board[ay][ax] = 0;
    break;
  case CHESS_MOVE_TAKE_ENPASSANT:
    enpassant_y = by + (CHESS_COLOR_WHITE == c ? 1 : -1);
    game->board[by][bx] = game->board[ay][ax];
    game->board[ay][ax] = 0;
    game->board[enpassant_y][bx] = 0;
    break;
  case CHESS_MOVE_CASTLE:
    if (bx < ax) {
      rook_src_x = 0;
      rook_dest_x = 3;
    } else {
      rook_src_x = 7;
      rook_dest_x = 5;
    }
    game->board[by][bx] = game->board[ay][ax];
    game->board[by][rook_dest_x] = game->board[by][rook_src_x];
    game->board[by][rook_src_x] = 0;
    break;
  case CHESS_MOVE_ILLEGAL:
  case CHESS_MOVE_UNSAFE:
    return;
  }

  if (CHESS_PIECE_PAWN == p && ((CHESS_COLOR_WHITE == c && 0 == by) || (CHESS_COLOR_BLACK == c && 7 == by))) {
    game->board[by][bx] = chess_new_square(CHESS_PIECE_QUEEN, c); // automatically promote to queen
  }
}
