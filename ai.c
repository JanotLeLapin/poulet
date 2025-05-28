#include "ai.h"

static float
relu(float v)
{
  return v < 0 ? 0 : v;
}
