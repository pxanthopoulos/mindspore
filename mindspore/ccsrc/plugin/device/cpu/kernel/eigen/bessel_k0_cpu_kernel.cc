/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include <map>
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/kernel/eigen/bessel_k0_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_k0.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselK0InputsNum = 1;
constexpr size_t kBesselK0OutputsNum = 1;
}  // namespace

bool BesselK0CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK0>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For 'BesselK0CpuKernelMod', BaseOperatorPtr can not dynamic cast to BesselK0 before initialize!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK0InputsNum || outputs.size() != kBesselK0OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK0InputsNum << " and "
                  << kBesselK0OutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_dtype_ = inputs[0]->GetDtype();

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK0CpuKernelMod::LaunchKernel<Eigen::half>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK0 kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK0CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());

  return 0;
}

template <typename T>
bool BesselK0CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  int block_size = 1000;
  size_t tensor_size = inputs[0]->size / sizeof(T);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> input(static_cast<T *>(inputs[0]->addr), tensor_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> output(static_cast<T *>(outputs[0]->addr), tensor_size);

  auto task = [this, &input, &output](size_t start, size_t end) {
    Eigen::array<Eigen::Index, 1> offsets = {static_cast<int64_t>(start)};
    Eigen::array<Eigen::Index, 1> extends = {static_cast<int64_t>(end - start)};
    output.slice(offsets, extends) = input.slice(offsets, extends).bessel_k0();
  };
  ParallelLaunch(task, input_size_, block_size);

  return true;
}

std::vector<KernelAttr> BesselK0CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK0, BesselK0CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
