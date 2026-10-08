/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <opencv2/core/core.hpp>
#include <torch/torch.h>


namespace okvis {

/// Convert a grayscale OpenCV image to a torch (byte) tensor.
/**
 * \param img   OpenCV image as cv::Mat.
 * \return      Corresponding torch (byte) tensor.
 */
inline torch::Tensor cvMatToTensor(const cv::Mat& img) {
  return torch::from_blob(img.data, {img.rows, img.cols}, at::kByte);
}

/// Convert a torch (byte) tensor to a grayscale OpenCV image.
/**
 * \param tensor  Torch tensor to convert.
 * \return        Corresponding OpenCV image as cv::Mat.
 */
inline cv::Mat tensorToCvMatByte(const torch::Tensor& tensor) {
  int width = tensor.sizes()[0];
  int height = tensor.sizes()[1];
  cv::Mat output_mat(cv::Size{height, width}, CV_8UC1, tensor.data_ptr<uchar>());  
  return output_mat.clone();
}

/// Convert a torch (float) tensor to a grayscale OpenCV image.
/**
 * \param tensor  Torch tensor to convert.
 * \return        Corresponding OpenCV image as cv::Mat.
 */
inline cv::Mat tensorToCvMatFloat(const torch::Tensor& tensor) {
  int width = tensor.sizes()[0];
  int height = tensor.sizes()[1];
  cv::Mat output_mat(cv::Size{height, width}, CV_32F, tensor.data_ptr<float>());  
  return output_mat.clone();
}

inline std::shared_ptr<float> tensorToSharedPtr(const torch::Tensor& tensor) {
  std::shared_ptr<torch::Tensor> shared_tensor(new torch::Tensor(tensor.permute({1, 2, 0}).contiguous()));
  return std::shared_ptr<float>(shared_tensor, shared_tensor->data_ptr<float>());

}

inline torch::Tensor eigenMatrixToTorch(const Eigen::MatrixXf& e) {
  auto t = torch::zeros({e.cols(), e.rows()}, torch::kFloat);
  auto* data = t.data_ptr<float>();
  Eigen::Map<Eigen::MatrixXf> ef(data, t.size(1), t.size(0));
  ef = e.cast<float>();
  return t.transpose(0, 1);
}

inline cv::Mat tensorToDepthVis(const torch::Tensor& tensor) {
  torch::Tensor visTensor = ((tensor/6.0).clamp(0.0, 1.0f) * 255.0f).to(torch::kU8);
  cv::Mat visMat = tensorToCvMatByte(visTensor.detach().cpu());
  cv::Mat visRgb;
  cv::applyColorMap(visMat, visRgb, cv::COLORMAP_INFERNO);

  return visRgb.clone();
}

inline cv::Mat tensorToiDepthVis(const torch::Tensor& tensor) {
  torch::Tensor iDepth = 1.0/tensor;
  torch::Tensor visTensor = ((iDepth/2.0).clamp(0.0, 1.0f) * 255.0f).to(torch::kU8);
  cv::Mat visMat = tensorToCvMatByte(visTensor.detach().cpu());
  cv::Mat visRgb;
  cv::applyColorMap(visMat, visRgb, cv::COLORMAP_INFERNO);

  return visRgb.clone();
}

inline cv::Mat tensorToiSigmaVis(const torch::Tensor& depth, const torch::Tensor& sigma) {
  torch::Tensor iSigma = (1.0/(depth*depth)) * sigma;
  torch::Tensor visTensor = ((iSigma/1.0).clamp(0.0, 1.0f) * 255.0f).to(torch::kU8);
  cv::Mat visMat = tensorToCvMatByte(visTensor.detach().cpu());
  cv::Mat visRgb;
  cv::applyColorMap(visMat, visRgb, cv::COLORMAP_JET);

  return visRgb.clone();
}
}