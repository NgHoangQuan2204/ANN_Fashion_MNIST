#include "./network.h"
#include "./layer/dense.h"
#include "./layer/cuda_utilities.h"

void Network::forward(const Matrix& input) {
  if (layers.empty())
    return;
  layers[0]->forward(input);
  if (forward_times.empty()) {
    forward_times.resize(layers.size(), 0.0f);
  }
  forward_count++;
  for (int i = 1; i < layers.size(); i++) {
    startTimer();
    layers[i]->forward(layers[i-1]->output());
    float elapsed = stopTimer();
    forward_times[i] += elapsed;
  }
}

void Network::backward(const Matrix& input, const Matrix& target) {
  int n_layer = layers.size();
  if (n_layer <= 0)
    return;
  loss->evaluate(layers[n_layer-1]->output(), target);
  if (backward_times.empty()) {
    backward_times.resize(layers.size(), 0.0f);
  }
  backward_count++;
  if (n_layer == 1) {
    startTimer();
    layers[0]->backward(input, loss->back_gradient());
    float elapsed = stopTimer();
    backward_times[0] += elapsed;
    return;
  }
  layers[n_layer-1]->backward(layers[n_layer-2]->output(), loss->back_gradient());
  for (int i = n_layer-2; i > 0; i--) {
    startTimer();
    layers[i]->backward(layers[i-1]->output(), layers[i+1]->back_gradient());
    float elapsed = stopTimer();
    backward_times[i] += elapsed;
  }
  layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(Optimizer& opt) {
  for (int i = 0; i < layers.size(); i++) {
    layers[i]->update(opt);
  }
}

std::vector<std::vector<float> > Network::get_parameters() const {
  const int n_layer = layers.size();
  std::vector< std::vector<float> > res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++) {
    res.push_back(layers[i]->get_parameters());
  }
  return res;
}

void Network::set_parameters(const std::vector< std::vector<float> >& param) {
  const int n_layer = layers.size();
  if (static_cast<int>(param.size()) != n_layer)
      throw std::invalid_argument("Parameter size does not match");
  for (int i = 0; i < n_layer; i++) {
    layers[i]->set_parameters(param[i]);
  }
}

std::vector<std::vector<float> > Network::get_derivatives() const {
  const int n_layer = layers.size();
  std::vector< std::vector<float> > res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++) {
    res.push_back(layers[i]->get_derivatives());
  }
  return res;
}

void Network::check_gradient(const Matrix& input, const Matrix& target,
                             int n_points, int seed) {
  if (seed > 0)
    std::srand(seed);

  this->forward(input);
  this->backward(input, target);
  std::vector< std::vector<float> > param = this->get_parameters();
  std::vector< std::vector<float> > deriv = this->get_derivatives();

  const float eps = 1e-4;
  const int n_layer = deriv.size();
  for (int i = 0; i < n_points; i++) {
    // Randomly select a layer
    const int layer_id = int(std::rand() / double(RAND_MAX) * n_layer);
    // Randomly pick a parameter, note that some layers may have no parameters
    const int n_param = deriv[layer_id].size();
    if (n_param < 1)  continue;
    const int param_id = int(std::rand() / double(RAND_MAX) * n_param);
    // Turbulate the parameter a little bit
    const float old = param[layer_id][param_id];

    param[layer_id][param_id] -= eps;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_pre = loss->output();

    param[layer_id][param_id] += eps * 2;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_post = loss->output();

    const float deriv_est = (loss_post - loss_pre) / eps / 2;

    std::cout << "[layer " << layer_id << ", param " << param_id <<
    "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
    ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;

    param[layer_id][param_id] = old;
  }

  // Restore original parameters
  this->set_parameters(param);
}

Matrix Network::get_weight_from_network(){
  for (Layer* layer : layers) {  // Duyệt qua các layer
        // Kiểm tra nếu lớp hiện tại là Dense
        Dense* fc_layer = dynamic_cast<Dense*>(layer);
        if (fc_layer) {
            return fc_layer->get_weight();  // Lấy trọng số từ layer Dense đầu tiên
        }
    }
    throw std::runtime_error("No Dense layer found in the network.");
}

void Network::print_average_times() const {
    std::cout << "Total forward times:" << std::endl;
    for (int i = 0; i < forward_times.size(); i++) {
        std::cout << "Layer " << i+1 << ": " << (forward_times[i]) << " ms" << std::endl;
    }

    std::cout << "Total backward times:" << std::endl;
    for (int i = 0; i < backward_times.size(); i++) {
        std::cout << "Layer " << i+1 << ": " << (backward_times[i]) << " ms" << std::endl;
    }
}
