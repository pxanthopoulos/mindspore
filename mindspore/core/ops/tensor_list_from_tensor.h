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

#ifndef MINDSPORE_CORE_OPS_TENSOR_LIST_FROM_TENSOR_H_
#define MINDSPORE_CORE_OPS_TENSOR_LIST_FROM_TENSOR_H_
#include <memory>
#include <vector>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTensorListFromTensor = "TensorListFromTensor";
class MS_CORE_API TensorListFromTensor : public PrimitiveC {
 public:
  TensorListFromTensor() : PrimitiveC(kNameTensorListFromTensor) {}
  ~TensorListFromTensor() = default;
  MS_DECLARE_PARENT(TensorListFromTensor, PrimitiveC);
  void Init(const int64_t element_dtype, const int64_t shape_type);
  void set_element_dtype(const int64_t element_dtype);
  void set_shape_type(const int64_t shape_type);
  int64_t get_element_dtype() const;
  int64_t get_shape_type() const;
};
AbstractBasePtr TensorListFromTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args);
using PrimTensorListFromTensorPtr = std::shared_ptr<TensorListFromTensor>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_LIST_FROM_TENSOR_H_
