#include "poulet.h"

#define EXPORT_SIZE(type_name) \
  size_t \
  w_##type_name##_size() \
  { \
    return sizeof(type_name); \
  }

EXPORT_SIZE(ai_layer_t)
EXPORT_SIZE(ai_brain_t)
EXPORT_SIZE(chess_color_t)
EXPORT_SIZE(chess_move_t)
EXPORT_SIZE(chess_game_t)
