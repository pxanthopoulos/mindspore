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

#ifndef MINDSPORE_CORE_OPS_GRAD_LAYER_NORM_GRAD_V3_H_
#define MINDSPORE_CORE_OPS_GRAD_LAYER_NORM_GRAD_V3_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormGradV3 = "LayerNormGradV3";
class MIND_API LayerNormGradV3 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LayerNormGradV3);
  LayerNormGradV3() : BaseOperator(kNameLayerNormGradV3) {}
  explicit LayerNormGradV3(const std::string k_name) : BaseOperator(k_name) {}
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1);
  void set_begin_norm_axis(const int64_t begin_norm_axis);
  void set_begin_params_axis(const int64_t begin_params_axis);
  int64_t get_begin_norm_axis() const;
  int64_t get_begin_params_axis() const;
  float get_epsilon() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_LAYER_NORM_GRAD_H_
