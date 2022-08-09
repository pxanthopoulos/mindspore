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

#ifndef MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_GRAD_H_
#define MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_GRAD_H_
#include <map>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeNearestNeighborGrad = "ResizeNearestNeighborGrad";
class MIND_API ResizeNearestNeighborGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ResizeNearestNeighborGrad);
  ResizeNearestNeighborGrad() : BaseOperator(kNameResizeNearestNeighborGrad) {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_GRAD_H_
