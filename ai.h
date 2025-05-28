#include <stdlib.h>

typedef float (*activation_func)(float);

typedef struct {
  size_t input_size;
  size_t output_size;
  float *weights;
  float *biases;
  float *outpus;
  activation_func activation;
} ai_layer_t;

typedef struct {
  size_t layer_count;
  ai_layer_t *layers;
} ai_brain_t;

void ai_layer_init(ai_layer_t *layer);
void ai_brain_init(ai_brain_t *brain);
