#include "debug.h"
#include "types.h"

class Network {
 public:
  Network(int, int, int);
  ~Network();

  float calculate_total_error(const std::vector<float> &expected);
  void propagate_forward(const std::vector<float> &inputs);
  void train(const std::vector<float> &inputs, const std::vector<float> &desired);
  void debug_output();
  std::vector<float> output;

 private:
  Layer_t *input_layer, *hidden_layer, *output_layer;
  std::vector<Layer_t *> layers;
};
