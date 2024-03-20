/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/all_finite_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void AllFiniteAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  for (auto input : inputs) {
    GetWorkspaceForResize(input, outputs[kIndex0]);
  }
}

bool AllFiniteAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  aclrtMemsetAsync(outputs[kIndex0]->device_ptr(), outputs[kIndex0]->size(), 0, outputs[kIndex0]->size(), stream_ptr);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const bool use_huge_pages = true;
    auto res = GEN_EXECUTOR_CUST(op_type_, use_huge_pages, inputs[i], outputs[kIndex0]);
    executor_ = std::get<kIndex1>(res);
    RunOp(stream_ptr, workspace);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AllFinite, AllFiniteAscend);
}  // namespace kernel
}  // namespace mindspore
