/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_H

#include <memory>
#include <vector>
#include "backend/optimizer/common/optimizer.h"
#include "schema/inner/model_generated.h"
#include "tools/common/storage.h"
#include "tools/converter/converter_flags.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
class AnfTransform {
 public:
  AnfTransform();
  virtual ~AnfTransform();
  FuncGraphPtr Transform(const FuncGraphPtr &old_graph, const converter::Flags *config = nullptr);

 private:
  STATUS GetAllFuncGraph(const FuncGraphPtr &main_graph, FuncGraphPtrList *subgraphs,
                         std::vector<ValueNodePtr> *vnodes);
  FuncGraphPtr TransformSingleFuncGraph(const FuncGraphPtr &old_graph, const converter::Flags *config = nullptr);
  std::unique_ptr<quant::Quantizer> mQuantizer = nullptr;

  int AddFusionPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config);

  int AddGraphPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config);

  int AddConvertPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config);

  int AddConstFoldPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config);

  int RunAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config);

  int RunMindirAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config);

  int RunOnnxAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config);

  int RunTFAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config);

  int DoQuantize(const FuncGraphPtr &old_graph, const converter::Flags *config, const FuncGraphPtr &new_graph);

  int DoHuffmanEncode(const converter::Flags *config, const FuncGraphPtr &new_graph, bool enableHuffmanCode);
};
}  // namespace lite
}  // namespace mindspore

#endif
