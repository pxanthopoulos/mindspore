/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/lp_norm_reduce_v2.h"

namespace mindspore {
namespace ops {
namespace {

void InferShapeWhenAxisBiggerThanOne(const PrimitivePtr &primitive, ShapeVector *output_shape) {
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  constexpr int64_t place_holder = INT64_MAX;
  auto prim_name = primitive->name();

  for (size_t i = 0; i < axis.size(); ++i) {
    auto temp = axis;
    auto idx = std::find(temp.begin(), temp.end(), axis[i]);
    (void)temp.erase(idx);
    auto re_idx = std::find(temp.begin(), temp.end(), axis[i]);
    if (re_idx != temp.end()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the element of the axis must be different, but got axis: " << axis << ".";
    }
    if (!keep_dims) {
      // Here, we need a placeholder for infer shape, but dynamic shape using -1, so just change to INT64_MAX.
      (*output_shape)[LongToSize(axis[i])] = place_holder;
    } else {
      (*output_shape)[LongToSize(axis[i])] = 1;
    }
  }
  if (!keep_dims) {
    for (auto iter = output_shape->begin(); iter != output_shape->end(); ++iter) {
      if (*iter == place_holder) {
        iter = output_shape->erase(iter);
        iter -= 1;
      }
    }
  }
}

abstract::ShapePtr LpNormReduceV2InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  ShapeVector input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto input_rank = SizeToLong(input_shape.size());
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  if (input_rank == 0) {
    (void)CheckAndConvertUtils::CheckInteger("axis size", SizeToLong(axis.size()), kEqual, input_rank + 1, prim_name);
    return std::make_shared<abstract::Shape>(input_shape);
  } else {
    CheckAndConvertUtils::CheckInRange("axis size", axis.size(), kIncludeNeither, {0, input_rank + 1}, prim_name);
  }
  for (int64_t &axi : axis) {
    CheckAndConvertUtils::CheckInRange("axis value", axi, kIncludeLeft, {-input_rank, input_rank}, prim_name);
    if (axi < 0) {
      axi += input_rank;
    }
  }
  bool invalid_axis = std::any_of(axis.begin(), axis.end(), [&input_rank](int64_t axis) { return axis >= input_rank; });
  if (invalid_axis) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the value of axis is out of range (-" << input_rank << ", "
                             << input_rank << ").";
  }
  ShapeVector output_shape = input_shape;
  if (axis.size() > 1) {
    InferShapeWhenAxisBiggerThanOne(primitive, &output_shape);
  } else {
    if (!keep_dims) {
      (void)output_shape.erase(output_shape.begin() + axis[0]);
    } else {
      output_shape[LongToSize(axis[0])] = 1;
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr LpNormReduceV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LpNormReduceV2, BaseOperator);
AbstractBasePtr LpNormReduceV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = LpNormReduceV2InferType(primitive, input_args);
  auto infer_shape = LpNormReduceV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void LpNormReduceV2::Init(const std::vector<int64_t> &axis, const float p, const bool keep_dims, const float epsilon) {
  this->set_axis(axis);
  this->set_p(p);
  this->set_keep_dims(keep_dims);
  this->set_epsilon(epsilon);
}

void LpNormReduceV2::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> LpNormReduceV2::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void LpNormReduceV2::set_p(const float p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

float LpNormReduceV2::get_p() const {
  auto value_ptr = this->GetAttr(kP);
  return GetValue<float>(value_ptr);
}

void LpNormReduceV2::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }

bool LpNormReduceV2::get_keep_dims() const {
  auto value_ptr = this->GetAttr(kKeepDims);
  return GetValue<bool>(value_ptr);
}

void LpNormReduceV2::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float LpNormReduceV2::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

// AG means auto generated
class MIND_API AGLpNormReduceV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LpNormReduceV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LpNormReduceV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LpNormReduceV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LpNormReduceV2, prim::kPrimLpNormReduceV2, AGLpNormReduceV2Infer, false);
}  // namespace ops
}  // namespace mindspore
