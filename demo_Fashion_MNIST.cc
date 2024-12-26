#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "src/layer.h"
#include "src/layer/dense.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
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

void testing(Network& dnn, MNIST& dataset, int epoch) {
  startTimer();    
  dnn.forward(dataset.test_data);
  std::cout << "Test time: " << stopTimer() << std::endl;
   
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << "Test acc: " << acc << std::endl;
  std::cout << std::endl;
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
  
  // dnn
  Network dnn;
  Layer* fc1 = new Dense(784, 128);
  Layer* fc2 = new Dense(128, 128);
  Layer* fc3 = new Dense(128, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* softmax = new Softmax;
  
  dnn.add_layer(fc1);
  dnn.add_layer(relu1);
  dnn.add_layer(fc2);
  dnn.add_layer(relu2);
  dnn.add_layer(fc3);
  dnn.add_layer(softmax);
  
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  SGD opt(0.0002, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 4;
  const int batch_size = 100;

  if (!IS_TRAINING) {
    for (int v = config::startVersion; v <= config::endVersion; v++)
    {
      config::currentVersion = v;
      std::cout << "\nCurrent version: " << config::currentVersion << "\n\n";
      std::cout << "\n\n";

      // Run on the test set
      testing(dnn, dataset, 0);
      std::cout << "------------------------------------------\n" << std::endl;

      if (!config::runAllVersion)
        break;
    }

    return 0;
  }

  Matrix previous_weight = dnn.get_weight_from_network();

  for (int v = config::startVersion; v <= config::endVersion; v++)
  {
    config::currentVersion = v;
    std::cout << "\nCurrent version: " << config::currentVersion << "\n";
    startTimer();
    for (int epoch = 0; epoch < n_epoch; epoch ++) 
    {
      shuffle_data(dataset.train_data, dataset.train_labels);
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
          dnn.check_gradient(x_batch, target_batch, 10);
        }
        
        dnn.forward(x_batch);

        dnn.backward(x_batch, target_batch);

        // display
        if (ith_batch % 100 == 0) {
          std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
        }
        
        // optimize
        dnn.update(opt);
      }
    // test
    testing(dnn, dataset, epoch);
    }
    std::cout << "Train time: " << stopTimer() << std::endl;
  }
  
  return 0;
}

