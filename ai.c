#include <math.h>
#include <stdio.h>
#include <string.h>

#include "poulet.h"

#define BURST_CHANCE 0.05f
#define BURST_AMPLITUDE 0.5f

#define BUF_READ(dst, src, size) \
  memcpy(&(dst), src, size); \
  src += size

static float
gaussian_noise()
{
  float u1 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
  float u2 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);

  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void
act_softmax(float *logits, size_t logit_count, float temperature)
{
  size_t i;
  float max, sum_exp;

  for (i = 0; i < logit_count; i++) {
    logits[i] = logits[i] / temperature;
  }

  max = logits[0];
  for (i = 0; i < logit_count; i++) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }

  sum_exp = 0.0f;
  for (i = 0; i < logit_count; i++) {
    logits[i] = expf(logits[i] - max);
    sum_exp += logits[i];
  }

  for (i = 0; i < logit_count; i++) {
    logits[i] /= sum_exp;
  }
}

int
ai_layer_init(ai_layer_t *layer, size_t input_size, size_t output_size, ai_activation_t activation)
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

static void
crossover(float *dst, float *a, float *b, size_t size)
{
  size_t i;
  float alpha, r;

  for (i = 0; i < size; i++) {
    alpha = 0.4f + ((float) rand() / RAND_MAX) * 0.2f;
    dst[i] = alpha * a[i] + (1 - alpha) * b[i];

    r = ((float) rand() / RAND_MAX) < 0.01f;

    if (r < 0.01f) {
      dst[i] += 0.1f * gaussian_noise();
    } else if (r < 0.01 + BURST_CHANCE) {
      dst[i] += BURST_AMPLITUDE * gaussian_noise();
    }
  }
}

void
ai_layer_offspring(ai_layer_t *child, ai_layer_t *a, ai_layer_t *b)
{
  crossover(child->weights, a->weights, b->weights, child->input_size * child->output_size);
  crossover(child->biases, a->biases, b->biases, child->output_size);
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
    layer->outputs[i] = sum;
  }

  switch (layer->activation.type) {
  case AI_ACTIVATION_RELU:
    for (i = 0; i < layer->output_size; i++) {
      layer->outputs[i] = layer->outputs[i] > 0 ? layer->outputs[i] : 0;
    }
    break;
  case AI_ACTIVATION_SOFTMAX:
    act_softmax(layer->outputs, layer->output_size, layer->activation.data.softmax.temperature);
    break;
  default:
    break;
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
    fwrite(&l.activation, sizeof(ai_activation_t), 1, f);
    fwrite(&l.input_size, sizeof(size_t), 1, f);
    fwrite(&l.output_size, sizeof(size_t), 1, f);
    fwrite(l.weights, sizeof(float), l.input_size * l.output_size, f);
    fwrite(l.biases, sizeof(float), l.output_size, f);
  }

  fclose(f);

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
    fread(&l->activation, sizeof(ai_activation_t), 1, f);
    fread(&l->input_size, sizeof(size_t), 1, f);
    fread(&l->output_size, sizeof(size_t), 1, f);
    l->weights = malloc(sizeof(float) * l->input_size * l->output_size);
    l->biases = malloc(sizeof(float) * l->output_size);
    l->outputs = malloc(sizeof(float) * l->output_size);
    fread(l->weights, sizeof(float), l->input_size * l->output_size, f);
    fread(l->biases, sizeof(float), l->output_size, f);
  }

  fclose(f);

  return 0;
}

int
ai_brain_load_from_buffer(ai_brain_t *brain, const uint8_t *buffer, size_t buffer_length)
{
  const uint8_t *p;
  size_t i;
  ai_layer_t *l;

  p = buffer;

  BUF_READ(brain->layer_count, p, sizeof(size_t));
  for (i = 0; i < brain->layer_count; i++) {
    l = &brain->layers[i];
    BUF_READ(l->activation, p, sizeof(ai_activation_t));
    BUF_READ(l->input_size, p, sizeof(size_t));
    BUF_READ(l->output_size, p, sizeof(size_t));
    l->weights = malloc(sizeof(float) * l->input_size * l->output_size);
    l->biases = malloc(sizeof(float) * l->output_size);
    l->outputs = malloc(sizeof(float) * l->output_size);
    BUF_READ(l->weights, p, sizeof(float) * l->input_size * l->output_size);
    BUF_READ(l->biases, p, sizeof(float) * l->output_size);
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
