/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed t*o in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/ops_func_impl/avg_pool_grad.h"

#include <vector>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr AvgPoolGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x = input_args[kInputIndex0];
  auto x_shape = x->GetShape();

  // Retain 'x_from_tensor' temporarily, and remove it after ascend is switched to the GE.
  constexpr auto kXFromTensor = "x_from_tensor";
  if (primitive->HasAttr(kXFromTensor) && GetValue<bool>(primitive->GetAttr(kXFromTensor))) {
    // The value of the first input is the real "x_origin" shape when 'x_from_tensor' exists.
    auto x_value = x->GetValue();
    if (x_value->isa<tensor::Tensor>()) {
      auto x_value_array_opt = GetArrayValue<int32_t>(x_value);
      if (!x_value_array_opt.has_value()) {
        return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
      }
      auto x_value_array = x_value_array_opt.value();
      ShapeVector value_shape;
      for (size_t i = 0; i < x_value_array.size(); i++) {
        if (x_value_array.IsValueUnknown(i)) {
          value_shape.push_back(abstract::Shape::kShapeDimAny);
        } else {
          value_shape.push_back(static_cast<int64_t>(x_value_array[i]));
        }
      }
      return std::make_shared<abstract::Shape>(std::move(value_shape));
    } else {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name
                               << "], the input argument 'x_origin' must be a tensor, but got " << x_value->ToString();
    }
  }

  const auto &shape = x_shape->GetShapeVector();
  if (!IsDynamicRank(shape)) {
    const size_t x_rank = 4;
    MS_CHECK_VALUE(shape.size() == x_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", SizeToLong(shape.size()), kEqual,
                                                               SizeToLong(x_rank), primitive));
  }
  return x_shape->Clone();
}

TypePtr AvgPoolGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  return x_type->Clone();
}

}  // namespace ops
}  // namespace mindspore
