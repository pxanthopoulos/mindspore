/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ConvGradInputGpuBkwKernel : public GpuKernel {
 public:
  ConvGradInputGpuBkwKernel()
      : cudnn_handle_(nullptr),
        w_desc_(nullptr),
        conv_desc_(nullptr),
        dy_desc_(nullptr),
        dx_desc_(nullptr),
        padded_descriptor_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        group_(1),
        is_null_input_(false),
        dy_size_(0),
        w_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~ConvGradInputGpuBkwKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *w = GetDeviceAddress<T>(inputs, 1);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    T *work_space = nullptr;
    if (workspace_size_ != 0) {
      work_space = GetDeviceAddress<T>(workspace, 0);
    }

    const float alpha = 1;
    const float beta = 0;
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);

      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta, padded_descriptor_, padded),
        "ConvolutionBackwardData failed");
      if (data_format_ == "NHWC") {
        CalPadGradNHWC(output_size_ / sizeof(T), padded, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                       old_width_ + pad_width_, pad_top_, pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CalPadGrad(output_size_ / sizeof(T), padded, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
                   old_width_ + pad_width_, pad_top_, pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
      }
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta, dx_desc_, dx),
        "ConvolutionBackwardData failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(dy_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "ConvGradInputGpuBkwKernel input is null.";
      InitSizeLists();
      return true;
    }

    std::vector<size_t> input_shape;
    GetInputShape(kernel_node, &input_shape);
    if (data_format_ == "NHWC") {
      compute_format_ = CUDNN_TENSOR_NHWC;
      ShapeNCHW2NHWC(&input_shape);
    }
    SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    Set4DDesc(dy_shape, input_shape, filter_shape);

    group_ = GetAttr<int>(kernel_node, "group");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionGroupCount(conv_desc_, group_), "cudnnSetConvGroupCount failed");

    auto pad_list = GetAttr<std::vector<int>>(kernel_node, "pad_list");
    pad_height_ = pad_list[0];
    pad_width_ = pad_list[2];
    auto symmetry_pad = (pad_height_ == pad_list[1]) && (pad_width_ == pad_list[3]);
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t dx_desc_real = nullptr;
    int padA[2];
    int strideA[2] = {stride_[0], stride_[1]};
    int dilaA[2] = {dilation_[0], dilation_[1]};
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase || !symmetry_pad) {
      pad_height_ = pad_list[0] + pad_list[1];
      pad_width_ = pad_list[2] + pad_list[3];
      pad_top_ = pad_list[0];
      pad_left_ = pad_list[2];
      if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
        use_pad_ = false;
      }
      int dimA[4];
      int strideApadded[4];
      if (data_format_ == "NCHW" || data_format_ == "DefaultFormat") {
        auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_height_ + pad_height_),
                             IntToSize(old_width_ + pad_width_)};
        SetDimA(padded_shape, dimA, data_format_);
        SetStrideA(padded_shape, strideApadded, data_format_);
      } else if (data_format_ == "NHWC") {
        auto padded_shape = {IntToSize(n_), IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_),
                             IntToSize(c_)};
        SetDimA(padded_shape, dimA, data_format_);
        SetStrideA(padded_shape, strideApadded, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, 4, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      if (use_pad_) {
        padA[0] = 0;
        padA[1] = 0;
      } else {
        padA[0] = pad_top_;
        padA[1] = pad_left_;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor failed");
      dx_desc_real = use_pad_ ? padded_descriptor_ : dx_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_height_;
      padA[1] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      dx_desc_real = dx_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(dx_desc_real);
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dx_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&padded_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateFilterDescriptor(&w_desc_), "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dy_desc_, &dy_size_), "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetFilterSizeInBytes(w_desc_, &w_size_), "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dx_desc_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(w_size_);
    output_size_list_.push_back(output_size_);

    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, padded_descriptor_,
                                                     algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetConvolutionBackwardDataWorkspaceSize(
                                      cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_, algo_, &workspace_size_),
                                    "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyFilterDescriptor(w_desc_), "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(padded_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dx_desc_), "cudnnDestroyTensorDescriptor failed");
  }
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ConvGradInput needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ConvGradInput needs 1 output.";
      return false;
    }
    return true;
  }
  void SetPad(const std::vector<int> &input_shape, const CNodePtr &kernel_node) {
    auto pad_list = GetAttr<std::vector<int>>(kernel_node, "pad_list");
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t dx_desc_real) {
    if (group_ > 1 || CUDNN_MAJOR < 7) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_real,
                                                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &algo_),
        "cudnnGetConvolutionBackwardDataAlgorithm failed");
    } else {
      constexpr int requested_algo_count = 1;
      int returned_algo_count;
      cudnnConvolutionBwdDataAlgoPerf_t perf_results;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_real,
                                                    requested_algo_count, &returned_algo_count, &perf_results),
        "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
      algo_ = perf_results.algo;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    }
  }
  void GetInputShape(const CNodePtr &kernel_node, std::vector<size_t> *input_shape) {
    auto shp_tuple_x = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("input_sizes")->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*input_shape),
                         [](const ValuePtr &e) -> size_t { return e->cast<Int32ImmPtr>()->value(); });
  }
  void Set4DDesc(const std::vector<size_t> &dy_shape, const std::vector<size_t> &input_shape,
                 const std::vector<size_t> &filter_shape) {
    int nbDims = 4;
    int dimA[4];
    int strideAin[4];
    int dimAdy[4];
    int strideAdy[4];
    int filterDimA[4];
    SetDimA(input_shape, dimA, data_format_);
    SetStrideA(input_shape, strideAin, data_format_);
    SetDimA(dy_shape, dimAdy, data_format_);
    SetStrideA(dy_shape, strideAdy, data_format_);
    SetDimA(filter_shape, filterDimA, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, nbDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, compute_format_, nbDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(dx_desc_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }
  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    stride_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "stride");
    dilation_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "dilation");
    if (stride_.size() != 2) {
      MS_LOG(EXCEPTION) << "ConvGradInputGpuBkwKernel's stride must be 2d!";
    }
    if (dilation_.size() != 4) {
      MS_LOG(EXCEPTION) << "ConvGradInputGpuBkwKernel's dilation must be 4d!";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "ConvGradInputGpuBkwKernel dilation only support 1 in N axis and C axis!";
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = "NCHW";
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  size_t dy_size_;
  size_t w_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
