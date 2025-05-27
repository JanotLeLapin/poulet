#ifndef _CHESS_GAME_H
#define _CHESS_GAME_H

#include <stdint.h>

typedef enum {
  CHESS_PIECE_PAWN = 1,
  CHESS_PIECE_BISHOP,
  CHESS_PIECE_KNIGHT,
  CHESS_PIECE_ROOK,
  CHESS_PIECE_QUEEN,
  CHESS_PIECE_KING,
} chess_piece_t;

typedef enum {
  CHESS_COLOR_WHITE = 0,
  CHESS_COLOR_BLACK = 1,
} chess_color_t;

typedef uint8_t chess_square_t;
typedef chess_square_t chess_board_t[8][8];

inline static chess_square_t
chess_new_square(chess_piece_t piece, chess_color_t color)
{
  chess_square_t res = 0;
  res |= piece;
  res |= color << 3;
  return res;
}

inline static const char *
chess_stringify_piece(chess_piece_t piece)
{
  switch (piece) {
  case CHESS_PIECE_PAWN:
    return "pawn";
  case CHESS_PIECE_BISHOP:
    return "bishop";
  case CHESS_PIECE_KNIGHT:
    return "knight";
  case CHESS_PIECE_ROOK:
    return "rook";
  case CHESS_PIECE_QUEEN:
    return "queen";
  case CHESS_PIECE_KING:
    return "king";
  }
}

inline static const char *
chess_stringify_color(chess_color_t color)
{
  switch (color) {
  case CHESS_COLOR_WHITE:
    return "white";
  case CHESS_COLOR_BLACK:
    return "black";
  }
}

void chess_init_board(chess_board_t board);

#endif
