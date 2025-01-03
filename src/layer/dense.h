#ifndef SRC_DENSE_H_
#define SRC_DENSE_H_

#include <vector>
#include "../layer.h"

class Dense : public Layer {
 private:
  const int dim_in;
  const int dim_out;

  Matrix weight;  // weight parameter
  Vector bias;  // bias paramter
  Matrix grad_weight;  // gradient w.r.t weight
  Vector grad_bias;  // gradient w.r.t bias

  void init();

 public:
  Dense(const int dim_in, const int dim_out) :
                 dim_in(dim_in), dim_out(dim_out)
  { init(); }

  void forward(const Matrix& bottom);
  void forwardVersion_Host(const Matrix& bottom); // Sequential
  void forwardVersion_Device(const Matrix& bottom, int version); // Parallel
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void backwardVersion_Host(const Matrix& bottom, const Matrix& grad_top); // Sequential
  void backwardVersion_Device(const Matrix& bottom, const Matrix& grad_top, int version); // Parallel
  void update(Optimizer& opt);
  int output_dim() { return dim_out; }
  std::vector<float> get_parameters() const;
  std::vector<float> get_derivatives() const;
  void set_parameters(const std::vector<float>& param);
  const Matrix& get_weight() const {return weight;}
};

#endif // SRC_DENSE_H_
