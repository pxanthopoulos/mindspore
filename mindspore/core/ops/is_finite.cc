/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/is_finite.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(IsFinite, BaseOperator);
class IsFiniteInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, 1L, primitive->name());
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    return std::make_shared<abstract::Shape>(x_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, 1L, primitive->name());
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    CheckAndConvertUtils::CheckTensorTypeValid(
      "x", input_args[0]->BuildType(),
      {kBool, kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kUInt32, kUInt64},
      primitive->name());
    return std::make_shared<TensorType>(kBool);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IsFinite, prim::kPrimIsFinite, IsFiniteInfer, false);
}  // namespace ops
}  // namespace mindspore
