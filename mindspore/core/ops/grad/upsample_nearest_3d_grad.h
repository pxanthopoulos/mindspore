/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRAD_UPSAMPLE_NEAREST_3D_GRAD_H
#define MINDSPORE_CORE_OPS_GRAD_UPSAMPLE_NEAREST_3D_GRAD_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUpsampleNearest3DGrad = "UpsampleNearest3DGrad";
class MIND_API UpsampleNearest3DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UpsampleNearest3DGrad);
  UpsampleNearest3DGrad() : BaseOperator(kNameUpsampleNearest3DGrad) { InitIOName({"dy"}, {"dx"}); }
  std::vector<int64_t> get_out_spatial_size() const;
  std::vector<int64_t> get_grad_spatial_size() const;
  std::vector<float> get_scale_factors() const;
};

MIND_API abstract::AbstractBasePtr UpsampleNearest3DGradInfer(const abstract::AnalysisEnginePtr &,
                                                              const PrimitivePtr &primitive,
                                                              const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimUpsampleNearest3DGrad = std::shared_ptr<UpsampleNearest3DGrad>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_GRAD_UPSAMPLE_TRILINEAR_3D_GRAD_H
