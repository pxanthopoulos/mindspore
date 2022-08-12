/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_
#define MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonMaxSuppressionWithOverlaps = "NonMaxSuppressionWithOverlaps";
class MIND_API NonMaxSuppressionWithOverlaps : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NonMaxSuppressionWithOverlaps);
  NonMaxSuppressionWithOverlaps() : BaseOperator("NonMaxSuppressionWithOverlaps") {}
};
abstract::AbstractBasePtr NonMaxSuppressionWithOverlapsInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimNonMaxSuppressionWihtOverlapsPtr = std::shared_ptr<NonMaxSuppressionWithOverlaps>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_
