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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_FLATTEN_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_FLATTEN_H_

#include <memory>
#include <vector>
#include "src/expression/node.h"

namespace mindspore {
namespace lite {
class FlattenM : public Node {
 public:
  FlattenM() = default;
  explicit FlattenM(int dummy);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_FLATTEN_H_
