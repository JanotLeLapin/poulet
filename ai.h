#include <stdlib.h>

typedef float (*activation_func)(float);

typedef struct {
  size_t input_size;
  size_t output_size;
  float *weights;
  float *biases;
  float *outputs;
  activation_func activation;
} ai_layer_t;

typedef struct {
  size_t layer_count;
  ai_layer_t *layers;
} ai_brain_t;

int ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size);
int ai_brain_init(ai_brain_t *brain, size_t layer_count);

void ai_layer_free(ai_layer_t *layer);
void ai_brain_free(ai_brain_t *brain);
