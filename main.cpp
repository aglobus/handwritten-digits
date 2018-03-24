#include <fstream>

#include "network.h"

namespace {
  Network network(N_INPUT_NEURONS_, N_HIDDEN_NEURONS_, N_OUTPUT_NEURONS_);

  void skip_lines(std::istream& pStream, size_t pLines) {
    std::string s;
    for (; pLines; --pLines)
      std::getline(pStream, s);
  }

  int biggest_index(std::vector<float> &vals) {
    auto biggest = std::max_element(std::begin(vals), std::end(vals));
    return std::distance(std::begin(vals), biggest);
  }

  void read_data(std::string &filename, std::vector<Test_t> &set) {
    std::ifstream file(filename, std::ios::in);

    // skip the first few lines
    skip_lines(file, HEADER_LEN_);

    Test_t t;

    int i = 1;
    for (std::string line; std::getline(file, line); i++) {
      if (i % 33 == 0) {
        //marks start of expected
        t.expected = std::vector<float>(N_OUTPUT_NEURONS_, 0);
        t.expected[stoi(line)] = 1.0f;

        set.push_back(t);
        t.inputs.clear();
      } else {
        //row of data
        for (int x = 0; x < line.size() - 1 /*\r at the end */; x++) { // line.size === 32
          float r = line[x] == '1' ? MAX_INPUT_ : MIN_INPUT_;
          t.inputs.push_back(r);
        }
      }
    }
  }

  void train() {
    srand(time(0));

    std::vector<Test_t> set;
    std::string trainingfile = "training-original.txt";
    read_data(trainingfile, set);
    int i = 0;
    int j = 0;
    size_t total = 0;
    size_t num_correct = 0;
    float sum_total_error = 0;
    while (true) {
      for (auto item : set) {
        network.train(item.inputs, item.expected);
        i++;

        int biggest_expected_index = biggest_index(item.expected);
        int biggest_output_index = biggest_index(network.output);

        if (biggest_output_index == biggest_expected_index) {
          num_correct++;
        }
        total++;

        // for (size_t t= 0; t < item.expected.size(); t++)
        //   debug("\t %f", item.expected[t]);
        // network.debug_output();

        // std::cout << network.calculate_total_error(item.expected) << "\n";
        sum_total_error += network.calculate_total_error(item.expected);

        std::cout << "\r" << i << "/" << N_EPOCHS_ * N_TRAIN_INPUTS_;
      }
      j++;
      if (j == N_EPOCHS_)
        break;
    }
    std::cout << "...done" << "\n";
    std::cout << "Correct" << "\t" << "Total" << "\n";
    std::cout << num_correct << "\t" << total << "\n";
    std::cout << "Average Total Error" << "\t" << sum_total_error / total << "\n";
  }

  void test() {
    std::vector<Test_t> set;
    std::string trainingfile = "testing-original.txt";
    read_data(trainingfile, set);

    size_t total = 0;
    size_t num_correct = 0;
    float sum_total_error = 0;
    int i = 0;
    for (auto item : set) {
      // network.train(item.inputs, item.expected);
      network.propagate_forward(item.inputs);

      int biggest_expected_index = biggest_index(item.expected);
      int biggest_output_index = biggest_index(network.output);

      if (biggest_output_index == biggest_expected_index) {
        num_correct++;
      }

      sum_total_error += network.calculate_total_error(item.expected);

      total++;
      i++;
      std::cout << "\r" << i << "/" << N_TEST_INPUTS_;
    }
    std::cout << "...done" << "\n";
    std::cout << "Correct" << "\t" << "Total" << "\n";
    std::cout << num_correct << "\t" << total << "\n";
    std::cout << "Average Total Error" << "\t" << sum_total_error / total << "\n";
  }
}

int main(void) {
  std::cout.setf(std::ios::unitbuf);
  std::cout << "Starting Training" << "\n";
  train();
  std::cout << "Starting Testing" << "\n";
  test();
  return 0;
}

