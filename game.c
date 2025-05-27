#include <stdint.h>

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
