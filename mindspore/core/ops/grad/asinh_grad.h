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

#ifndef MINDSPORE_CORE_OPS_ASINH_GRAD_H_
#define MINDSPORE_CORE_OPS_ASINH_GRAD_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAsinhGrad = "AsinhGrad";

class AsinhGrad : public PrimitiveC {
 public:
  AsinhGrad() : PrimitiveC(kNameAsinhGrad) { InitIOName({"y", "dy"}, {"z"}); }
  ~AsinhGrad() = default;

  MS_DECLARE_PARENT(AsinhGrad, PrimitiveC);
};

AbstractBasePtr AsinhGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimAsinhGradPtr = std::shared_ptr<AsinhGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ASINH_GRAD_H_
