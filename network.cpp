#include "network.h"

double sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

double sigmoid_derivative(float x) {
  double fx = sigmoid(x);
  return fx * (1 - fx);
}

Network::Network(int numi, int numh, int numo)
  :
  input_layer(new Layer_t(1,numi)),
  hidden_layer(new Layer_t(numi,numh)),
  output_layer(new Layer_t(numh,numo))
{
  layers.push_back(input_layer);
  layers.push_back(hidden_layer);
  layers.push_back(output_layer);
}

Network::~Network() {
  for (Layer_t *l : layers)
    delete l;
}

float Network::calculate_total_error(const std::vector<float> &expected) {
  float sum = 0;
  for (size_t i = 0; i < output.size(); i++) {
    sum += 0.5f * (output[i] - expected[i]) * (output[i] - expected[i]);
  }
  return sum;
}


void Network::train(const std::vector<float> &inputs, const std::vector<float> &desired) {
  propagate_forward(inputs);

  std::vector<float> errors;

  /* update hidden layer to output layer weights */
  for (size_t i = 0; i < output_layer->num_neurons; i++) {
    Neuron *n = output_layer->neurons[i];

    double node_delta = -(desired[i] - output[i]) * sigmoid_derivative(n->input_sum);

    errors.push_back(node_delta);

    for (size_t j = 0; j < n->num_inputs; j++) {
      if (n->previous_weights[j] == 0)
        n->previous_weights[j] = n->weights[j];

       n->delta_weights[j] = n->weights[j] - n->previous_weights[j];
       n->previous_weights[j] = n->weights[j];

       double weight_change = sigmoid(hidden_layer->neurons[j]->input_sum) * node_delta + MOMENTUM_ * n->delta_weights[j];

       n->weights[j] -= ETA_ * weight_change;
    }
  }

  /* Update input layer to hidden layer weights */
  for (size_t j = 0; j < hidden_layer->num_neurons; j++) {
    Neuron *hn = hidden_layer->neurons[j];
    double sum_output = 0;

    for (size_t i = 0; i < output_layer->num_neurons; i++) {
      sum_output += output_layer->neurons[i]->weights[j] * errors[i];
    }

    for (size_t k = 0; k < hn->num_inputs; k++) {
      if (hn->previous_weights[k] == 0)
        hn->previous_weights[k] = hn->weights[k];

      hn->delta_weights[k] = hn->weights[k] - hn->previous_weights[k];
      hn->previous_weights[k] = hn->weights[k];

      double weight_change = sigmoid(input_layer->neurons[k]->input_sum) * sigmoid_derivative(hn->input_sum) * sum_output + MOMENTUM_ * hn->delta_weights[k];

      hn->weights[k] -= ETA_ * weight_change;
    }
  }
}

void Network::debug_output() {
  debug("\tOutput vector\n");
  for (size_t i = 0; i < output_layer->num_neurons; i++) {
    debug("\t %f",output[i]);
  }
  debug("\n");
}

void Network::propagate_forward(const std::vector<float> &inputs) {
  std::vector<float> working_set;

  for (int i = 0; i < layers.size(); i++) {
    working_set.clear();
    Layer_t *l = layers[i];

    if (i > 0) {
      working_set = output;
    } else {
      working_set = inputs;
    }

    output.clear();

    for (size_t j = 0; j < l->num_neurons; j++) {
      Neuron_t *n = l->neurons[j];

      double net_input = 0;

      if (i > 0) {
        for (size_t k = 0; k < working_set.size(); k++) {
          net_input += n->weights[k] * working_set[k];
        }
      } else {
        net_input = working_set[j];
      }

      n->input_sum = net_input;
      output.push_back(sigmoid(n->input_sum));
    }
  }
}
