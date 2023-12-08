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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LP_NORM_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LP_NORM_FISSION_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class LpNormFission : public PatternProcessPass {
 public:
  explicit LpNormFission(bool multigraph = true) : PatternProcessPass("lp_norm_fission", multigraph) {}
  ~LpNormFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateLpNormReduceV2(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                  const AnfNodePtr &cast_node) const;
  AnfNodePtr CreateLpNormUpdateV2(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                  const AnfNodePtr &lp_norm_reduce_v2_outputs) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LP_NORM_FISSION_H_
