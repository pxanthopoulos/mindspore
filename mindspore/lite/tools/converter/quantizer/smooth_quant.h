/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_SMOOTH_QUANT_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_SMOOTH_QUANT_H
#include "ir/anf.h"

namespace mindspore::lite::quant {
class SmoothQuant {
 public:
  SmoothQuant() = default;

  ~SmoothQuant() = default;

  int Run(const FuncGraphPtr &func_graph, double smooth_alpha);

 private:
  int LinearSmooth(const FuncGraphPtr &func_graph, const CNodePtr &cnode, double alpha);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_SMOOTH_QUANT_H
