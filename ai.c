#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ai.h"

static float
gaussian_noise()
{
  float u1 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
  float u2 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);

  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}


int
ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size, activation_func activation)
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

  scale = sqrtf(2.0f / input_size);
  for (i = 0; i < input_size * output_size; i++) {
    layer->weights[i] = scale * ((float) rand() / RAND_MAX * 2 - 1);
  }
  for (i = 0; i < output_size; i++) {
    layer->biases[i] = 0.0f;
  }

  layer->activation = activation;

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
ai_layer_offspring(ai_layer_t *child, ai_layer_t *a, ai_layer_t *b)
{
  size_t i;

  for (i = 0; i < child->input_size * child->output_size; i++) {
    child->weights[i] = ((rand() % 2) == 0 ? a : b)->weights[i];
    if (((float) rand() / RAND_MAX) < 0.01f) {
      child->weights[i] += 0.1f * gaussian_noise();
    }
  }

  for (i = 0; i < child->output_size; i++) {
    child->biases[i] = ((rand() % 2) == 0 ? a : b)->biases[i];
    if (((float) rand() / RAND_MAX) < 0.01f) {
      child->biases[i] += 0.1f * gaussian_noise();
    }
  }

  child->activation = a->activation;
}

void
ai_brain_offspring(ai_brain_t *a, ai_brain_t *b, ai_brain_t *child)
{
  size_t i;

  for (i = 0; i < child->layer_count; i++) {
    ai_layer_offspring(&child->layers[i], &a->layers[i], &b->layers[i]);
  }
}

void
ai_layer_forward(ai_layer_t *layer, float *input)
{
  size_t i, j;
  float sum;

  for (i = 0; i < layer->output_size; i++) {
    sum = layer->biases[i];
    for (j = 0; j < layer->input_size; j++) {
      sum += layer->weights[i * layer->input_size + j] * input[j];
    }
    layer->outputs[i] = layer->activation(sum);
  }
}

void
ai_brain_forward(ai_brain_t *brain, float *input)
{
  size_t i;
  float *current_input = input;

  for (i = 0; i < brain->layer_count; i++) {
    ai_layer_forward(&brain->layers[i], current_input);
    current_input = brain->layers[i].outputs;
  }
}

int
ai_brain_save(ai_brain_t *brain, const char *filename)
{
  FILE *f;
  size_t i;
  ai_layer_t l;

  f = fopen(filename, "wb");

  fwrite(&brain->layer_count, sizeof(size_t), 1, f);
  for (i = 0; i < brain->layer_count; i++) {
    l = brain->layers[i];
    fwrite(&l.input_size, sizeof(size_t), 1, f);
    fwrite(&l.output_size, sizeof(size_t), 1, f);
    fwrite(l.weights, sizeof(float), l.input_size * l.output_size, f);
    fwrite(l.biases, sizeof(float), l.output_size, f);
  }

  return 0;
}

int
ai_brain_load(ai_brain_t *brain, const char *filename)
{
  FILE *f;
  size_t i;
  ai_layer_t *l;

  f = fopen(filename, "rb");

  fread(&brain->layer_count, sizeof(size_t), 1, f);
  brain->layers = malloc(sizeof(ai_layer_t) * brain->layer_count);
  for (i = 0; i < brain->layer_count; i++) {
    l = &brain->layers[i];
    fread(&l->input_size, sizeof(size_t), 1, f);
    fread(&l->output_size, sizeof(size_t), 1, f);
    l->weights = malloc(sizeof(float) * l->input_size * l->output_size);
    l->biases = malloc(sizeof(float) * l->output_size);
    l->outputs = malloc(sizeof(float) * l->output_size);
    l->activation = ai_activation_relu;
    fread(l->weights, sizeof(float), l->input_size * l->output_size, f);
    fread(l->biases, sizeof(float), l->output_size, f);
  }

  return 0;
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
