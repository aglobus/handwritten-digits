#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>

#define HEADER_LEN_ 21

#define N_INPUT_NEURONS_ 1024
#define N_HIDDEN_NEURONS_ 32
#define N_OUTPUT_NEURONS_ 10

#define MIN_INPUT_ -1.0
#define MAX_INPUT_ 1.0

#define MIN_WEIGHT_ -1.0
#define MAX_WEIGHT_ 1.0

#define N_EPOCHS_ 100

#define N_TRAIN_INPUTS_ 1934
#define N_TEST_INPUTS_ 943

#define MOMENTUM_ 0.07
#define ETA_ 0.5

typedef struct {
  std::vector<float> inputs;
  std::vector<float> expected;
} Test_t;

typedef struct Neuron {
  float input_sum; //summation
  size_t num_inputs; // number of inputs this neuron takes
  std::vector<float> weights; // vector of weights, sizes matches num_inputs
  std::vector<float> previous_weights;
  std::vector<float> delta_weights;
Neuron(int num_inputs) : num_inputs(num_inputs), input_sum(0) {
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<float> distribution(MIN_WEIGHT_, MAX_WEIGHT_);
    for (size_t i = 0; i < num_inputs; i++)
      weights.push_back(distribution(engine));
    delta_weights = std::vector<float>(num_inputs, 0);
    previous_weights = std::vector<float>(num_inputs, 0);

  }
  ~Neuron() {
  }
} Neuron_t;

typedef struct Layer {
  size_t num_neurons; // number of neurons in the layer
  std::vector<Neuron_t *> neurons; // vector of neurons in this layer
  std::vector<float> activated_summation;
  size_t input_count;
  Layer(int inputs_per_neuron, int num_neurons) : num_neurons(num_neurons) {
    input_count = inputs_per_neuron * num_neurons;
    activated_summation = std::vector<float>(num_neurons);
    for (size_t i = 0; i < num_neurons; i++)
      neurons.push_back(new Neuron_t(inputs_per_neuron));
  }
  ~Layer() {
    for (Neuron_t *n : neurons)
      delete n;
  }
} Layer_t;
