/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_VOLUME_PATCHES_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_VOLUME_PATCHES_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const T *input, T *output,
  cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_VOLUME_PATCHES_IMPL_CUH_
