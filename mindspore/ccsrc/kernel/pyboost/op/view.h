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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_VIEW_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_VIEW_H_

#include "kernel/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT View : public pyboost::Op {
 public:
  View() = default;
  ~View() = default;
  void CastInput() override;

  virtual tensor::TensorPtr Call(const tensor::TensorPtr &input, const std::vector<Int64ImmPtr> &shape);
  void PyboostProcessView(const tensor::TensorPtr &input, const std::vector<Int64ImmPtr> &shape,
                          const std::string &device_target);
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_VIEW_H_
