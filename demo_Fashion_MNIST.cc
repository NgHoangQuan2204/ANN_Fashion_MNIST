#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

#include "src/layer.h"
#include "src/layer/dense.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

#include "src/layer/cuda_utilities.h"
#include "config.h"
#include <thread>
#include <vector>
#include <numeric>

const bool IS_TRAINING = true;
const bool IS_CREATING_TEST_CASES = false;

namespace config 
{
  int currentVersion = 1;
  int startVersion = 2;
  int endVersion = 2;
  bool runAllVersion = false;
  float forwardTime = 0;
}

float testing(Network& ann, MNIST& dataset, int epoch) {
  ann.forward(dataset.test_data);
   
  float acc = compute_accuracy(ann.output(), dataset.test_labels);
  std::cout << "Test acc: " << acc << std::endl;
  std::cout << std::endl;
  return acc;
}

int main(int argc, char** argv) {
  config::startVersion = std::stoi(argv[1]);
  config::endVersion = std::stoi(argv[2]);
  config::runAllVersion = std::stoi(argv[3]);

  // data
  MNIST dataset("../data/fashion-mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  
  // ann
  Network ann;
  Layer* fc1 = new Dense(784, 128);
  Layer* fc2 = new Dense(128, 128);
  Layer* fc3 = new Dense(128, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* softmax = new Softmax;
  
  ann.add_layer(fc1);
  ann.add_layer(relu1);
  ann.add_layer(fc2);
  ann.add_layer(relu2);
  ann.add_layer(fc3);
  ann.add_layer(softmax);
  
  // loss
  Loss* loss = new CrossEntropy;
  ann.add_loss(loss);
  // train & test
  SGD opt(0.0001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 3;
  const int batch_size = 100;

  Matrix previous_weight = ann.get_weight_from_network();

  for (int v = config::startVersion; v <= config::endVersion; v++)
  {
    config::currentVersion = v;
    std::cout << "\nCurrent version: " << config::currentVersion << "\n\n";
    float avg_acc = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < n_epoch; epoch ++) 
    {
      for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) 
      {
        int ith_batch = start_idx / batch_size;
        Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                      std::min(batch_size, n_train - start_idx));
        Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                      std::min(batch_size, n_train - start_idx));
        Matrix target_batch = one_hot_encode(label_batch, 10);
        if (false && ith_batch % 10 == 1) {
          std::cout << ith_batch << "-th grad: " << std::endl;
          ann.check_gradient(x_batch, target_batch, 10);
        }
        
        ann.forward(x_batch);

        ann.backward(x_batch, target_batch);

        // display
        if (ith_batch % 100 == 0) {
          std::cout << ith_batch << "-th batch, loss: " << ann.get_loss() << std::endl;
        }
        
        // optimize
        ann.update(opt);
      }
    // test
    avg_acc += testing(ann, dataset, epoch);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Average accuracy: " << avg_acc/n_epoch << std::endl;
    std::cout << "Train time: " << duration.count() << " s" << std::endl;
    ann.print_average_times();
  }
  
  return 0;
}

