/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/switch_not_cut.h"

#include <memory>
#include <vector>
#include <utility>
#include "ops/other_ops.h"
#include "ops/framework_ops.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool IsValidInlinePartial(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if(!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if(cnode->size()<=kPartialGraphIndex) {
    return false;
  }
  auto sub_graph = common::AnfAlgo::GetValueNodeFuncGraph(cnode->input(kIndex1));
  if(sub_graph == nullptr || sub_graph->return_node()==nullptr || sub_graph->return_node()->size()<=1) {
    return false;
  }
  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(sub_graph->return_node()->input(1));
  if(std::any_of(outputs.begin(), outputs.end(), [](const std::pair<AnfNodePtr, int64_t> &pair){
    return pair.first !=nullptr && pair.first->isa<ValueNode>();})){
    return false;
  }
  return true;
}

bool IsValidInlineSwitch(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != kSwitchInputSize) {
    MS_LOG(DEBUG) << "Invalid switch node" << cnode->DebugString();
    return false;
  }
  if((!IsValidInlinePartial(cnode->input(kSwitchTrueBranchIndex))) ||
         (!IsValidInlinePartial(cnode->input(kSwitchFalseBranchIndex)))){
    return false;
  }
  return true;
}

bool IsAbstractIncludeDynamicLen(const abstract::AbstractBasePtr &abstract) {
  if(abstract == nullptr || (!abstract->isa<abstract::AbstractSequence>())) {
    return false;
  }
  const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if(sequence_abs->dynamic_len()) {
    return true;
  }
  for(const auto &sub_abs : sequence_abs->elements()) {
    if(IsAbstractIncludeDynamicLen(sub_abs)) {
      return true;
    }
  }
  return false;
}

bool SwitchNotCut::Run(const FuncGraphPtr &func_graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const bool is_enable_ge = (context->backend_policy() == "ge");
  if (!is_enable_ge) {
    // only support ge backend
    return false;
  }
  static const auto is_enable_switch_inline = (common::GetEnv("MS_ENABLE_SWITCH_INLINE") == "1");
  if (!is_enable_switch_inline) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : all_nodes) {
    if(!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if(cnode->inputs().empty()) {
      continue;
    }
    if(IsAbstractIncludeDynamicLen(cnode->abstract())){
      continue;
    }
    auto primitive_input = cnode->input(kAnfPrimitiveIndex);
    if (!IsPrimitiveCNode(primitive_input, prim::kPrimSwitch) || (!IsValidInlineSwitch(primitive_input))) {
      continue;
    }
    cnode->AddPrimalAttr(kAttrNotCut, MakeValue(true));
    const auto &switch_node = primitive_input->cast<CNodePtr>();
    switch_node->AddPrimalAttr(kAttrNotCut, MakeValue(true));
    const auto &true_partial_node = switch_node->input(kSwitchTrueBranchIndex)->cast<CNodePtr>();
    true_partial_node->AddPrimalAttr(kAttrNotCut, MakeValue(true));
    auto true_partial_graph = true_partial_node->input(kIndex1);
    auto true_sub_graph = common::AnfAlgo::GetValueNodeFuncGraph(true_partial_graph);
    MS_EXCEPTION_IF_NULL(true_sub_graph);
    true_sub_graph->set_flag(kFlagSwitchInline, true);
    const auto &false_partial_node = switch_node->input(kSwitchFalseBranchIndex)->cast<CNodePtr>();
    false_partial_node->AddPrimalAttr(kAttrNotCut, MakeValue(true));
    auto false_partial_graph = false_partial_node->input(kIndex1);
    auto false_sub_graph = common::AnfAlgo::GetValueNodeFuncGraph(false_partial_graph);
    MS_EXCEPTION_IF_NULL(false_sub_graph);
    false_sub_graph->set_flag(kFlagSwitchInline, true);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
