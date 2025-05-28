#include <math.h>
#include <stdlib.h>

#include "ai.h"

static float
relu(float v)
{
  return v < 0 ? 0 : v;
}

int
ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size)
{
  float scale;
  size_t i;

  layer->input_size = input_size;
  layer->output_size = output_size;

  layer->weights = malloc(sizeof(float) * input_size * output_size);
  if (0 == layer->weights) {
    return -1;
  }
  layer->biases = malloc(sizeof(float) * output_size);
  if (0 == layer->biases) {
    free(layer->weights);
    return -1;
  }
  layer->outputs = malloc(sizeof(float) * output_size);
  if (0 == layer->outputs) {
    free(layer->weights);
    free(layer->biases);
    return -1;
  }

  scale = sqrtf(2.0f / (input_size + output_size));
  for (i = 0; i < input_size * output_size; i++) {
    layer->weights[i] = scale * ((float) rand() / RAND_MAX * 2 - 1);
  }
  for (i = 0; i < output_size; i++) {
    layer->biases[i] = 0.0f;
  }

  return 0;
}

int
ai_brain_init(ai_brain_t *brain, size_t layer_count)
{
  brain->layer_count = layer_count;
  brain->layers = malloc(sizeof(ai_layer_t) * layer_count);

  return brain->layers == 0 ? -1 : 0;
}

void
ai_layer_free(ai_layer_t *layer)
{
  free(layer->weights);
  free(layer->biases);
  free(layer->outputs);
}

void
ai_brain_free(ai_brain_t *brain)
{
  size_t i;

  for (i = 0; i < brain->layer_count; i++) {
    ai_layer_free(&brain->layers[i]);
  }

  free(brain->layers);
}
