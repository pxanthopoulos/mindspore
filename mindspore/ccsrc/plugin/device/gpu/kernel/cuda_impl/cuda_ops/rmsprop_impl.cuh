/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RMSPROP_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RMSPROP_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements, const T *learning_rate,
                             const T decay, const T momentum, const T epsilon, T *variable, T *mean_square, T *moment,
                             T *gradients, const size_t size, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements, const T *learning_rate,
                                   const T *decay, const T *momentum, const T *epsilon, T *variable, T *mean_gradients,
                                   T *mean_square, T *moment, T *gradients, const size_t size,
                                   cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RMSPROP_IMPL_CUH_
