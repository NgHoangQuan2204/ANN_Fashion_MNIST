#include "./dense.h"
#include "../../config.h"
#include "cuda_utilities.h"

void Dense::init() 
{
  weight.resize(dim_in, dim_out);
  bias.resize(dim_out);
  grad_weight.resize(dim_in, dim_out);
  grad_bias.resize(dim_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.008);
  set_normal_random(bias.data(), bias.size(), 0, 0.008);
}

Matrix HostMatrixMultiplication(const Matrix& A, const Matrix& B) 
{
    // Kiểm tra kích thước hợp lệ
    if (A.cols() != B.rows()) 
    {
      throw std::invalid_argument("Matrix dimensions do not match");
    }
  
    // Khởi tạo ma trận kết quả với các giá trị 0
    Matrix result = Matrix::Zero(A.rows(), B.cols());
  
    // Tính toán nhân ma trận
    for (int i = 0; i < A.rows(); ++i) 
    {      
      for (int j = 0; j < B.cols(); ++j) 
      {   
        float sum = 0.0f;                 
        for (int k = 0; k < A.cols(); ++k) 
        {
          sum += A(i, k) * B(k, j);
        }
        result(i, j) = sum;                
      }
    }
    return result; 
}

void Dense::forward(const Matrix& bottom) 
{
  // z = w' * x + b
  switch (config::currentVersion)
  {
  case 1:
    Dense::forwardVersion_Host(bottom);
    break;
  case 2:
    Dense::forwardVersion_Device(bottom, 1);
    break;
  case 3:
    Dense::forwardVersion_Device(bottom, 2);
    break;
  case 4:
    Dense::forwardVersion_Device(bottom, 3);
    break;
  case 5:
    Dense::forwardVersion_Device(bottom, 4);
    break;
  
  default:
    Dense::forwardVersion_Host(bottom);
    break;
  }
}

// Sequential Version
void Dense::forwardVersion_Host(const Matrix& bottom)
{
  // z = w' * x + b

  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample);

  // Sử dụng HostMatrixMultiplication để tính toán nhân ma trận
  top = HostMatrixMultiplication(weight.transpose(), bottom);

  // Cộng bias vào mỗi cột của ma trận kết quả
  top.colwise() += bias;
}

// Parallel Version
void Dense::forwardVersion_Device(const Matrix& bottom, int version)
{
  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample);    

  // Chuẩn bị dữ liệu trên CPU và GPU
  float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));
  float* h_result = (float*)calloc(dim_out * n_sample, sizeof(float));  // Ma trận kết quả

  for (int i = 0; i < n_sample; i++) 
  {
    // Trích xuất cột của bottom
    float* columnData = const_cast<float*>(bottom.col(i).data());

    // Chuyển đổi cột sang hàng
    Matrix columnMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(columnData, dim_in, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrix.data(), dim_in * sizeof(float));

    // Thực hiện phép nhân ma trận trên GPU
    matrixMultiplicationGPUWrapper(weight.transpose().data(), h_bottom + i * dim_in, h_result + i * dim_out,
                                   dim_out, dim_in, 1, version);

    // Cộng bias vào kết quả GPU
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_result + i * dim_out, dim_out, 1).colwise() += bias;
  }

  // Chuyển kết quả từ GPU về Eigen Matrix
  Matrix result = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_result, dim_out, n_sample);
  top = result;

  free(h_bottom);
  free(h_result);
}

void Dense::backward(const Matrix& bottom, const Matrix& grad_top) 
{

  switch (config::currentVersion)
  {
  case 1:
    Dense::backwardVersion_Host(bottom, grad_top);
    break;
  case 2:
    Dense::backwardVersion_Device(bottom, grad_top, 1);
    break;
  case 3:
    Dense::backwardVersion_Device(bottom, grad_top, 2);
    break;
  case 4:
    Dense::backwardVersion_Device(bottom, grad_top, 3);
    break;
  case 5:
    Dense::backwardVersion_Device(bottom, grad_top, 4);
    break;
  
  default:
    Dense::backwardVersion_Host(bottom, grad_top);
    break;
  }
}

// Sequential Version
void Dense::backwardVersion_Host(const Matrix& bottom, const Matrix& grad_top) 
{
  const int n_sample = bottom.cols();

  // Tính grad_weight = bottom * grad_top.transpose()
  grad_weight = bottom * grad_top.transpose();

  // Tính grad_bias = \sum{ d(L)/d(z_i) }
  grad_bias = grad_top.rowwise().sum();

  // Tính grad_bottom = weight * grad_top sử dụng HostMatrixMultiplication
  grad_bottom = HostMatrixMultiplication(weight, grad_top);
}

// Parallel Version
void Dense::backwardVersion_Device(const Matrix& bottom, const Matrix& grad_top, int version) 
{
    const int n_sample = bottom.cols();

    float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));
    float* h_grad_top = (float*)malloc(dim_out * n_sample * sizeof(float));
    float* h_grad_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));

    grad_weight = bottom * grad_top.transpose();

    // Tính grad_bias = \sum(d(L)/d(z))
    grad_bias.resize(dim_out, 1);
    for (int i = 0; i < dim_out; ++i) 
    {
        grad_bias(i, 0) = grad_top.row(i).sum();
    }

    // Tính grad_bottom = weight * grad_top
    for (int i = 0; i < n_sample; i++) 
    {
        // Trích xuất cột của grad_top
        float* columnData = const_cast<float*>(grad_top.col(i).data());

        // Chuyển đổi cột sang hàng
        Matrix columnMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(columnData, dim_out, 1);
        std::memcpy(h_grad_top + i * dim_out, columnMatrix.data(), dim_out * sizeof(float));

        // Thực hiện phép nhân ma trận trên GPU
        matrixMultiplicationGPUWrapper(weight.data(), h_grad_top + i * dim_out, h_grad_bottom + i * dim_in,
                                       dim_in, dim_out, 1, version);
    }

    // Chuyển kết quả từ GPU về Eigen Matrix
    grad_bottom = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_grad_bottom, dim_in, n_sample);

    free(h_bottom);
    free(h_grad_top);
    free(h_grad_bottom);

}

void Dense::update(Optimizer& opt) 
{
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());

  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Dense::get_parameters() const 
{
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(),
            res.begin() + weight.size());
  return res;
}

void Dense::set_parameters(const std::vector<float>& param)
{
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Dense::get_derivatives() const 
{
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(),
            res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
