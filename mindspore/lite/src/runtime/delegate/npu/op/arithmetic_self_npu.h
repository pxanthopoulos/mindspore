/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_ARITHMETIC_SELF_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_ARITHMETIC_SELF_NPU_H_
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/graph/op/math_defs.h"
#include "src/runtime/delegate/npu/op/npu_op.h"

namespace mindspore::lite {
class ArithmeticSelfNPUOp : public NPUOp {
 public:
  ArithmeticSelfNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, name) {}

  ~ArithmeticSelfNPUOp() override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override {
    return RET_OK;
  }

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors,
                   const std::vector<ge::Operator *> &npu_inputs) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors, const std::vector<ge::Operator *> &npu_inputs,
                   const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) override;

  ge::Operator *GetNPUOp() override;

 private:
  ge::Operator *op_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_ARITHMETICSELF_NPU_H_
