#include <stdlib.h>

typedef enum {
  AI_ACTIVATION_RELU = 0,
  AI_ACTIVATION_SOFTMAX,
} ai_activation_type_t;

typedef struct {
  ai_activation_type_t type;
  union {
    struct {
      float temperature;
    } softmax;
  } data;
} ai_activation_t;

typedef struct {
  size_t input_size;
  size_t output_size;
  float *weights;
  float *biases;
  float *outputs;
  ai_activation_t activation;
} ai_layer_t;

typedef struct {
  size_t layer_count;
  ai_layer_t *layers;
} ai_brain_t;

int ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size, ai_activation_t activation);
void ai_layer_offspring(ai_layer_t *child, ai_layer_t *a, ai_layer_t *b);
void ai_layer_forward(ai_layer_t *layer, float *input);
void ai_layer_free(ai_layer_t *layer);

int ai_brain_init(ai_brain_t *brain, size_t layer_count);
void ai_brain_offspring(ai_brain_t *a, ai_brain_t *b, ai_brain_t *child);
void ai_brain_forward(ai_brain_t *brain, float *input);
int ai_brain_save(ai_brain_t *brain, const char *filename);
int ai_brain_load(ai_brain_t *brain, const char *filename);
void ai_brain_free(ai_brain_t *brain);
